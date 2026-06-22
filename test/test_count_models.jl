using Test
using MiniGPT

@testset "count models" begin
    @testset "unigram counts" begin
        model = fit_unigram([1, 2, 1, 3], 3; alpha = 0.0)

        @test model isa UnigramCountModel
        @test model.counts == [2, 1, 1]
        @test sum(model.counts) == 4
        @test model.prob ≈ [0.5, 0.25, 0.25]
        @test sum(model.prob) ≈ 1.0

        report = evaluate(model, [1, 2, 3])
        @test report.prediction == 3
        @test isfinite(report.nll)
        @test isfinite(report.perplexity)
    end

    @testset "unigram smoothing" begin
        model = fit_unigram([1, 1, 2], 4; alpha = 0.5)

        @test model.prob[3] > 0.0
        @test model.prob[4] > 0.0
        @test sum(model.prob) ≈ 1.0
        @test !any(isnan, model.prob)
        @test !any(isnan, model.log_prob)
    end

    @testset "bigram counts" begin
        model = fit_bigram([1, 2, 1, 2, 3], 3; alpha = 0.0)

        @test model isa BigramCountModel
        @test model.counts[1, 2] == 2
        @test model.counts[2, 1] == 1
        @test model.counts[2, 3] == 1
        @test sum(model.counts) == 4
        @test model.probs[1, 2] ≈ 1.0
        @test model.probs[2, 1] ≈ 0.5
        @test model.probs[2, 3] ≈ 0.5
    end

    @testset "bigram row normalization" begin
        model = fit_bigram([1, 2, 1, 2, 3], 4; alpha = 0.25)

        @test size(model.probs) == (4, 4)
        @test all(p -> p > 0.0, model.probs)
        @test !any(isnan, model.probs)
        @test !any(isnan, model.logprobs)

        for previous in axes(model.probs, 1)
            @test sum(@view model.probs[previous, :]) ≈ 1.0
        end
    end

    @testset "bigram zero-outgoing backoff" begin
        tokens = [1, 2, 1]
        model = fit_bigram(tokens, 3; alpha = 0.0, backoff_alpha = 0.5)
        backoff = fit_unigram(tokens, 3; alpha = 0.5)

        @test sum(@view model.probs[3, :]) ≈ 1.0
        @test !any(isnan, @view model.probs[3, :])
        @test model.probs[3, :] ≈ backoff.prob
    end

    @testset "evaluation correctness" begin
        unigram = fit_unigram([1, 1, 2], 2; alpha = 0.0)
        unigram_expected = -(log(2 / 3) + log(1 / 3)) / 2
        unigram_report = evaluate(unigram, [1, 2])

        @test unigram_report.prediction == 2
        @test unigram_report.nll ≈ unigram_expected
        @test unigram_report.perplexity ≈ exp(unigram_expected)

        bigram = fit_bigram([1, 2, 2], 2; alpha = 0.0)
        bigram_report = evaluate(bigram, [1, 2, 2])

        @test bigram_report.prediction == 2
        @test bigram_report.nll ≈ 0.0
        @test bigram_report.perplexity ≈ 1.0
    end

    @testset "unseen events" begin
        unsmoothed_unigram = fit_unigram([1, 1], 2; alpha = 0.0)
        smoothed_unigram = fit_unigram([1, 1], 2; alpha = 0.1)
        unsmoothed_bigram = fit_bigram([1, 2, 1], 2; alpha = 0.0)
        smoothed_bigram = fit_bigram([1, 2, 1], 2; alpha = 0.1)

        @test evaluate(unsmoothed_unigram, [2]).nll == Inf
        @test evaluate(unsmoothed_unigram, [2]).perplexity == Inf
        @test evaluate(unsmoothed_bigram, [1, 1]).nll == Inf
        @test evaluate(unsmoothed_bigram, [1, 1]).perplexity == Inf

        @test isfinite(evaluate(smoothed_unigram, [2]).nll)
        @test isfinite(evaluate(smoothed_unigram, [2]).perplexity)
        @test isfinite(evaluate(smoothed_bigram, [1, 1]).nll)
        @test isfinite(evaluate(smoothed_bigram, [1, 1]).perplexity)
        @test !isnan(evaluate(smoothed_unigram, [2]).nll)
        @test !isnan(evaluate(smoothed_bigram, [1, 1]).nll)
    end

    @testset "generation reproducibility" begin
        unigram = fit_unigram([1, 2, 1, 3], 3; alpha = 0.1)
        bigram = fit_bigram([1, 2, 1, 3, 1, 2], 3; alpha = 0.1)

        first_unigram = generate_ids(unigram, 10; seed = 7)
        second_unigram = generate_ids(unigram, 10; seed = 7)
        first_bigram = generate_ids(bigram, [1], 8; seed = 7)
        second_bigram = generate_ids(bigram, [1], 8; seed = 7)

        @test first_unigram == second_unigram
        @test first_bigram == second_bigram
        @test length(first_unigram) == 10
        @test length(first_bigram) == 9
        @test all(id -> id in 1:3, first_unigram)
        @test all(id -> id in 1:3, first_bigram)
        @test generate_ids(bigram, [1, 2], 0; seed = 7) == [1, 2]
    end

    @testset "most common bigrams" begin
        model = fit_bigram([1, 2, 1, 2, 2, 1, 3], 3; alpha = 0.1)
        entries = most_common_bigrams(model; k = 10)

        @test all(entry -> entry[3] > 0, entries)
        @test entries[1][1:3] == (1, 2, 2)
        @test entries[2][1:3] == (2, 1, 2)
        @test entries[3][1:3] == (1, 3, 1)
        @test entries[4][1:3] == (2, 2, 1)
        @test most_common_bigrams(model; k = 0) == Tuple{Int,Int,Int,Float64}[]
        @test length(entries) == 4
    end

    @testset "input validation" begin
        unigram = fit_unigram([1, 2], 2; alpha = 0.1)
        bigram = fit_bigram([1, 2, 1], 2; alpha = 0.1)

        @test_throws ArgumentError fit_unigram([1], 0)
        @test_throws ArgumentError fit_bigram([1, 2], 0)
        @test_throws ArgumentError fit_unigram([1], 2; alpha = -0.1)
        @test_throws ArgumentError fit_bigram([1, 2], 2; alpha = -0.1)
        @test_throws ArgumentError fit_unigram([1], 2; alpha = Inf)
        @test_throws ArgumentError fit_bigram([1, 2], 2; alpha = Inf)
        @test_throws ArgumentError fit_unigram(Int[], 2)
        @test_throws ArgumentError fit_bigram([1], 2)
        @test_throws ArgumentError fit_unigram([0], 2)
        @test_throws ArgumentError fit_bigram([1, 0], 2)
        @test_throws ArgumentError fit_unigram([3], 2)
        @test_throws ArgumentError fit_bigram([1, 3], 2)
        @test_throws ArgumentError evaluate(unigram, Int[])
        @test_throws ArgumentError evaluate(bigram, [1])
        @test_throws ArgumentError generate_ids(unigram, -1)
        @test_throws ArgumentError generate_ids(bigram, Int[], 1)
        @test_throws ArgumentError generate_ids(bigram, [3], 1)
    end
end
