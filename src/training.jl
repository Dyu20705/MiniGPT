mutable struct NeuralBigramTrainingState
    step::Int
    tokens_seen::Int
    best_validation_loss::Float64
    best_step::Int
    best_logits_table::Any
    seed::Int
    history::Vector{NamedTuple}
end

function NeuralBigramTrainingState(; seed::Integer = 42)
    seed >= 0 ||
        throw(ArgumentError("seed must be non-negative"))

    return NeuralBigramTrainingState(
        0,
        0,
        Inf,
        0,
        nothing,
        Int(seed),
        NamedTuple[],
    )
end

function _evaluate_token_subset(
    model::NeuralBigramModel,
    tokens::AbstractVector{<:Integer},
    max_predictions::Integer,
)
    max_predictions > 0 ||
        throw(ArgumentError("max_predictions must be positive"))

    if length(tokens) - 1 <= max_predictions
        return evaluate(model, tokens)
    end

    return evaluate(model, tokens[1:(max_predictions + 1)])
end

function train!(
    model::NeuralBigramModel,
    train_tokens::AbstractVector{<:Integer},
    validation_tokens::AbstractVector{<:Integer};
    learning_rate::Real = 0.1,
    batch_size::Integer = 64,
    max_steps::Integer = 2000,
    eval_interval::Integer = 100,
    seed::Integer = 42,
    eval_max_predictions::Integer = typemax(Int),
)
    length(train_tokens) >= 2 ||
        throw(ArgumentError("train_tokens must contain at least two tokens"))
    length(validation_tokens) >= 2 ||
        throw(ArgumentError("validation_tokens must contain at least two tokens"))
    batch_size >= 1 ||
        throw(ArgumentError("batch_size must be positive"))
    max_steps >= 1 ||
        throw(ArgumentError("max_steps must be positive"))
    eval_interval >= 1 ||
        throw(ArgumentError("eval_interval must be positive"))

    rng = MersenneTwister(seed)
    state = NeuralBigramTrainingState(; seed)

    for _ in 1:max_steps
        inputs, targets = get_batch(
            train_tokens;
            context_length = 1,
            batch_size,
            rng,
        )

        step_result = train_step!(
            model,
            inputs,
            targets;
            learning_rate,
        )

        state.step += 1
        state.tokens_seen += length(targets)

        if state.step == 1 || state.step % eval_interval == 0 || state.step == max_steps
            train_report = _evaluate_token_subset(model, train_tokens, eval_max_predictions)
            validation_report =
                _evaluate_token_subset(model, validation_tokens, eval_max_predictions)

            if validation_report.nll < state.best_validation_loss
                state.best_validation_loss = validation_report.nll
                state.best_step = state.step
                state.best_logits_table = copy(model.logits_table)
            end

            push!(
                state.history,
                (
                    step = state.step,
                    tokens_seen = state.tokens_seen,
                    batch_loss = Float64(step_result.loss),
                    gradient_norm = Float64(step_result.gradient_norm),
                    train_nll = train_report.nll,
                    train_perplexity = train_report.perplexity,
                    train_accuracy = train_report.accuracy,
                    validation_nll = validation_report.nll,
                    validation_perplexity = validation_report.perplexity,
                    validation_accuracy = validation_report.accuracy,
                ),
            )
        end
    end

    return state
end
