using Test
using MiniGPT

@testset "neural bigram objectives" begin
    @test isapprox(logsumexp([0.0, 0.0, 0.0]), log(3))
    @test isapprox(cross_entropy([0.0, 0.0, 0.0], 1), log(3))

    stable_loss = cross_entropy([1000.0, 0.0, 0.0], 1)
    @test isfinite(stable_loss)
    @test isapprox(stable_loss, 0.0; atol = 1e-10)

    logits_matrix = [0.0 1000.0; 0.0 0.0; 0.0 0.0]
    @test isapprox(mean_cross_entropy(logits_matrix, [2, 1]), log(3) / 2; atol = 1e-10)
    @test isapprox(perplexity_from_loss(log(3)), 3)

    @test_throws ArgumentError logsumexp(Float64[])
    @test_throws ArgumentError cross_entropy([0.0, 0.0], 0)
    @test_throws ArgumentError mean_cross_entropy(zeros(2, 2), [1])
end

@testset "neural bigram model" begin
    model = NeuralBigramModel(47; seed = 7)
    same_seed = NeuralBigramModel(47; seed = 7)
    different_seed = NeuralBigramModel(47; seed = 8)

    @test size(model.logits_table) == (47, 47)
    @test vocab_size(model) == 47
    @test parameter_count(model) == 2209
    @test model.logits_table == same_seed.logits_table
    @test model.logits_table != different_seed.logits_table
    @test size(logits(model, 1)) == (47,)

    batch = [1 2; 3 4]
    batch_logits = logits(model, batch)
    @test size(batch_logits) == (47, 2, 2)
    @test batch_logits[:, 1, 1] == model.logits_table[:, 1]
    @test batch_logits[:, 2, 1] == model.logits_table[:, 3]

    zero_model = NeuralBigramModel(5; init = :zeros)
    @test isapprox(mean_cross_entropy(zero_model, [1, 2, 3], [2, 3, 4]), log(5))

    @test_throws ArgumentError NeuralBigramModel(0)
    @test_throws ArgumentError NeuralBigramModel(3; seed = -1)
    @test_throws ArgumentError NeuralBigramModel(3; init = :bad)
    @test_throws ArgumentError logits(model, 0)
    @test_throws ArgumentError logits(model, [1, 48])
end

@testset "neural bigram gradient and SGD" begin
    model = NeuralBigramModel(3; init = :zeros)
    inputs = [1, 1, 2]
    targets = [2, 2, 3]

    loss_before = mean_cross_entropy(model, inputs, targets)
    gradient = neural_bigram_gradient(model, inputs, targets)

    @test size(gradient) == size(model.logits_table)
    @test all(isfinite, gradient)
    @test gradient_norm(gradient) > 0
    @test isapprox(sum(gradient[:, 1]), 0.0; atol = 1e-7)
    @test isapprox(sum(gradient[:, 2]), 0.0; atol = 1e-7)

    original = copy(model.logits_table)
    step_result = train_step!(model, inputs, targets; learning_rate = 0.5)

    @test isfinite(step_result.loss)
    @test step_result.gradient_norm > 0
    @test model.logits_table != original
    @test mean_cross_entropy(model, inputs, targets) < loss_before

    finite_difference_model = NeuralBigramModel(3; seed = 3, dtype = Float64)
    fd_inputs = [1, 2]
    fd_targets = [2, 3]
    fd_gradient = neural_bigram_gradient(finite_difference_model, fd_inputs, fd_targets)
    epsilon = 1e-5
    row = 2
    column = 1

    plus = NeuralBigramModel(copy(finite_difference_model.logits_table))
    minus = NeuralBigramModel(copy(finite_difference_model.logits_table))
    plus.logits_table[row, column] += epsilon
    minus.logits_table[row, column] -= epsilon

    numerical = (
        mean_cross_entropy(plus, fd_inputs, fd_targets) -
        mean_cross_entropy(minus, fd_inputs, fd_targets)
    ) / (2epsilon)

    @test isapprox(fd_gradient[row, column], numerical; atol = 1e-6)
end

@testset "neural bigram evaluation, overfit, generation, checkpoint" begin
    repeated = [1, 2, 1, 2, 1, 2, 1, 2]
    model = NeuralBigramModel(2; init = :zeros)
    initial_loss = evaluate(model, repeated).nll

    for _ in 1:200
        train_step!(
            model,
            repeated[1:(end - 1)],
            repeated[2:end];
            learning_rate = 0.5,
        )
    end

    report = evaluate(model, repeated)
    @test report.nll < initial_loss / 4
    @test report.predictions == length(repeated) - 1
    @test report.correct == report.predictions
    @test isapprox(report.accuracy, 1.0)
    @test argmax(logits(model, 1)) == 2
    @test argmax(logits(model, 2)) == 1

    greedy = generate_ids(model, [1], 5; strategy = :greedy)
    @test greedy == [1, 2, 1, 2, 1, 2]

    sample_a = generate_ids(model, [1], 10; seed = 11, temperature = 1.0)
    sample_b = generate_ids(model, [1], 10; seed = 11, temperature = 1.0)
    @test sample_a == sample_b
    @test length(sample_a) == 11
    @test all(id -> id in 1:2, sample_a)

    @test generate_ids(model, [1, 2], 0; strategy = :greedy) == [1, 2]
    @test_throws ArgumentError generate_ids(model, Int[], 1)
    @test_throws ArgumentError generate_ids(model, [1], -1)
    @test_throws ArgumentError generate_ids(model, [1], 1; temperature = 0.0)
    @test_throws ArgumentError generate_ids(model, [1], 1; strategy = :topk)

    path = joinpath(mktempdir(), "neural_bigram_checkpoint.bin")
    save_checkpoint(
        path,
        model;
        step = 200,
        config = Dict("learning_rate" => 0.5),
        best_validation_loss = report.nll,
        seed = 123,
    )
    loaded = load_checkpoint(path)

    @test loaded.step == 200
    @test loaded.vocab_size == 2
    @test loaded.config["learning_rate"] == 0.5
    @test isapprox(loaded.best_validation_loss, report.nll)
    @test loaded.model.logits_table == model.logits_table
    @test logits(loaded.model, 1) == logits(model, 1)
    @test generate_ids(loaded.model, [1], 10; seed = 9) ==
          generate_ids(model, [1], 10; seed = 9)
end

@testset "neural bigram training loop" begin
    train_tokens = [1, 2, 1, 2, 1, 2, 1, 2]
    validation_tokens = [1, 2, 1, 2]
    model = NeuralBigramModel(2; init = :zeros)
    state = train!(
        model,
        train_tokens,
        validation_tokens;
        learning_rate = 0.4,
        batch_size = 4,
        max_steps = 20,
        eval_interval = 5,
        seed = 5,
    )

    @test state.step == 20
    @test state.tokens_seen == 80
    @test isfinite(state.best_validation_loss)
    @test state.best_step > 0
    @test state.best_logits_table !== nothing
    @test length(state.history) >= 4
    @test state.history[end].validation_nll <= state.history[1].validation_nll
end
