using LinearAlgebra
using Random

"""
Neural bigram language model.

`logits_table[next, current]` stores the logit for predicting `next` after
seeing `current`. This is intentionally the transpose of `BigramCountModel`,
which stores count-model probabilities as `probs[previous, next]`.
"""
struct NeuralBigramModel{T<:AbstractMatrix{<:AbstractFloat}}
    logits_table::T

    function NeuralBigramModel{T}(logits_table::T) where {T<:AbstractMatrix{<:AbstractFloat}}
        _validate_square_logits_table(logits_table)

        return new{T}(logits_table)
    end
end

function NeuralBigramModel(
    vocab_size::Integer;
    seed::Integer = 42,
    scale::Real = 0.01,
    dtype::Type{<:AbstractFloat} = Float32,
    init::Symbol = :normal,
)
    V = _validate_vocab_size(vocab_size)
    seed >= 0 ||
        throw(ArgumentError("seed must be non-negative"))
    scale >= 0 ||
        throw(ArgumentError("scale must be non-negative"))

    rng = MersenneTwister(seed)

    logits_table = if init == :normal
        dtype.(float(scale) .* randn(rng, V, V))
    elseif init == :zeros
        zeros(dtype, V, V)
    else
        throw(ArgumentError("init must be :normal or :zeros"))
    end

    return NeuralBigramModel(logits_table)
end

vocab_size(model::NeuralBigramModel) = size(model.logits_table, 1)

function parameter_count(model::NeuralBigramModel)
    V = vocab_size(model)
    return V * V
end

function _validate_square_logits_table(logits_table::AbstractMatrix)
    size(logits_table, 1) == size(logits_table, 2) ||
        throw(ArgumentError("logits_table must have shape (vocab_size, vocab_size)"))

    size(logits_table, 1) > 0 ||
        throw(ArgumentError("logits_table must not be empty"))

    return nothing
end

function NeuralBigramModel(logits_table::AbstractMatrix{<:AbstractFloat})
    return NeuralBigramModel{typeof(logits_table)}(logits_table)
end

function _validate_temperature(temperature::Real)
    temperature_value = float(temperature)

    isfinite(temperature_value) ||
        throw(ArgumentError("temperature must be finite"))
    temperature_value > 0 ||
        throw(ArgumentError("temperature must be positive"))

    return temperature_value
end

function _validate_model_token_id(token::Integer, model::NeuralBigramModel)
    return _validate_target_id(token, vocab_size(model))
end

"""
Return logits for a single current token as a vector of length `vocab_size`.
"""
function logits(model::NeuralBigramModel, token::Integer)
    token_id = _validate_model_token_id(token, model)

    return @view(model.logits_table[:, token_id])
end

"""
Forward pass for token ids of any shape.

For input ids with shape `dims...`, the output shape is
`(vocab_size, dims...)`. Flattening the non-vocabulary axes yields the standard
`(vocab_size, predictions)` layout used by the loss functions.
"""
function logits(
    model::NeuralBigramModel,
    input_ids::AbstractArray{<:Integer},
)
    isempty(input_ids) &&
        throw(ArgumentError("input_ids must not be empty"))

    V = vocab_size(model)
    _validate_token_id(vec(input_ids), V)

    flattened_logits = model.logits_table[:, vec(Int.(input_ids))]
    output_shape = (V, size(input_ids)...)

    return reshape(flattened_logits, output_shape)
end

(model::NeuralBigramModel)(token::Integer) = logits(model, token)
(model::NeuralBigramModel)(input_ids::AbstractArray{<:Integer}) = logits(model, input_ids)

function mean_cross_entropy(
    model::NeuralBigramModel,
    inputs::AbstractArray{<:Integer},
    targets::AbstractArray{<:Integer},
)
    size(inputs) == size(targets) ||
        throw(ArgumentError("inputs and targets must have the same shape"))

    flattened_logits = reshape(logits(model, inputs), vocab_size(model), :)

    return mean_cross_entropy(flattened_logits, vec(Int.(targets)))
end

function _softmax_column!(
    probabilities::AbstractVector{<:AbstractFloat},
    logits_vector::AbstractVector{<:Real},
)
    normalizer = logsumexp(logits_vector)

    @inbounds for id in eachindex(probabilities)
        probabilities[id] = exp(float(logits_vector[id]) - normalizer)
    end

    return probabilities
end

"""
Analytic gradient of mean cross-entropy with respect to `logits_table`.

For each prediction, `dL/dz = softmax(z) - one_hot(target)`.
"""
function neural_bigram_gradient(
    model::NeuralBigramModel,
    inputs::AbstractArray{<:Integer},
    targets::AbstractArray{<:Integer},
)
    size(inputs) == size(targets) ||
        throw(ArgumentError("inputs and targets must have the same shape"))

    V = vocab_size(model)
    input_vector = vec(Int.(inputs))
    target_vector = vec(Int.(targets))
    _validate_token_id(input_vector, V)
    _validate_token_id(target_vector, V)

    isempty(input_vector) &&
        throw(ArgumentError("inputs must not be empty"))

    gradient = zeros(eltype(model.logits_table), size(model.logits_table))
    probabilities = Vector{eltype(model.logits_table)}(undef, V)
    scale = one(eltype(model.logits_table)) / length(input_vector)

    @inbounds for position in eachindex(input_vector)
        current = input_vector[position]
        target = target_vector[position]

        _softmax_column!(probabilities, @view(model.logits_table[:, current]))

        for next in 1:V
            gradient[next, current] += probabilities[next] * scale
        end

        gradient[target, current] -= scale
    end

    return gradient
end

function gradient_norm(gradient::AbstractArray{<:Real})
    return norm(vec(float.(gradient)))
end

function sgd_step!(
    model::NeuralBigramModel,
    gradient::AbstractMatrix{<:Real};
    learning_rate::Real,
)
    size(gradient) == size(model.logits_table) ||
        throw(ArgumentError("gradient shape must match logits_table shape"))

    learning_rate_value = float(learning_rate)
    isfinite(learning_rate_value) ||
        throw(ArgumentError("learning_rate must be finite"))
    learning_rate_value > 0 ||
        throw(ArgumentError("learning_rate must be positive"))

    model.logits_table .-= learning_rate_value .* gradient

    return model
end

function train_step!(
    model::NeuralBigramModel,
    inputs::AbstractArray{<:Integer},
    targets::AbstractArray{<:Integer};
    learning_rate::Real,
)
    loss = mean_cross_entropy(model, inputs, targets)
    gradient = neural_bigram_gradient(model, inputs, targets)
    norm_value = gradient_norm(gradient)
    sgd_step!(model, gradient; learning_rate)

    return (loss = loss, gradient_norm = norm_value)
end

struct NeuralLMReport
    nll::Float64
    perplexity::Float64
    predictions::Int
    correct::Int
    accuracy::Float64
end

function _neural_predictions(model::NeuralBigramModel, tokens::AbstractVector{<:Integer})
    length(tokens) >= 2 ||
        throw(ArgumentError("neural bigram evaluation requires at least two tokens"))

    inputs = Int.(tokens[1:(end - 1)])
    targets = Int.(tokens[2:end])

    return inputs, targets
end

function evaluate(
    model::NeuralBigramModel,
    tokens::AbstractVector{<:Integer},
)
    inputs, targets = _neural_predictions(model, tokens)
    logits_matrix = reshape(logits(model, inputs), vocab_size(model), :)
    loss = mean_cross_entropy(logits_matrix, targets)

    correct = 0

    @inbounds for position in eachindex(targets)
        predicted = argmax(@view(logits_matrix[:, position]))
        correct += predicted == targets[position] ? 1 : 0
    end

    predictions = length(targets)
    accuracy = correct / predictions

    return NeuralLMReport(
        Float64(loss),
        perplexity_from_loss(loss),
        predictions,
        correct,
        accuracy,
    )
end

function _argmax_token(logits_vector::AbstractVector{<:Real})
    best_id = firstindex(logits_vector)
    best_value = logits_vector[best_id]

    @inbounds for id in eachindex(logits_vector)
        if logits_vector[id] > best_value
            best_id = id
            best_value = logits_vector[id]
        end
    end

    return Int(best_id)
end

"""
Generate ids from a neural bigram model.

`strategy = :sample` uses categorical sampling. `strategy = :greedy` appends the
highest-logit token at each step. The returned ids always include the prompt.
"""
function generate_ids(
    model::NeuralBigramModel,
    prompt_ids::AbstractVector{<:Integer},
    max_new_tokens::Integer;
    seed::Integer = 42,
    temperature::Real = 1.0,
    strategy::Symbol = :sample,
)
    isempty(prompt_ids) &&
        throw(ArgumentError("neural bigram generation requires a prompt"))
    max_new_tokens >= 0 ||
        throw(ArgumentError("max_new_tokens must be non-negative"))
    seed >= 0 ||
        throw(ArgumentError("seed must be non-negative"))

    V = vocab_size(model)
    _validate_token_id(prompt_ids, V)
    temperature_value = _validate_temperature(temperature)

    strategy in (:sample, :greedy) ||
        throw(ArgumentError("strategy must be :sample or :greedy"))

    rng = MersenneTwister(seed)
    output = Int[Int(id) for id in prompt_ids]

    for _ in 1:max_new_tokens
        current_logits = logits(model, output[end])

        next = if strategy == :greedy
            _argmax_token(current_logits)
        else
            probabilities = _softmax_probabilities(current_logits; temperature = temperature_value)
            _sample_categorical(rng, probabilities)
        end

        push!(output, next)
    end

    return output
end
