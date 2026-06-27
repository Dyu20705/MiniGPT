"""
Numerically stable log-sum-exp for a vector of logits.
"""
function logsumexp(logits::AbstractVector{<:Real})
    isempty(logits) &&
        throw(ArgumentError("logits must not be empty"))

    max_logit = maximum(logits)

    isfinite(max_logit) ||
        return float(max_logit)

    total = 0.0

    @inbounds for logit in logits
        total += exp(float(logit) - max_logit)
    end

    return max_logit + log(total)
end

function _validate_target_id(target::Integer, vocab_size::Integer)
    1 <= target <= vocab_size ||
        throw(ArgumentError(
            "target id $target is out of bounds. Valid range is [1, $vocab_size].",
        ))

    return Int(target)
end

"""
Cross-entropy for one target id and one vector of logits.
"""
function cross_entropy(
    logits::AbstractVector{<:Real},
    target::Integer,
)
    target_id = _validate_target_id(target, length(logits))

    return logsumexp(logits) - float(logits[target_id])
end

"""
Mean cross-entropy for a `(vocab_size, predictions)` logits matrix.

Each column is one prediction position; `targets[i]` is the target token id for
`logits[:, i]`.
"""
function mean_cross_entropy(
    logits::AbstractMatrix{<:Real},
    targets::AbstractVector{<:Integer},
)
    size(logits, 2) == length(targets) ||
        throw(ArgumentError(
            "number of logit columns must match number of targets",
        ))

    isempty(targets) &&
        throw(ArgumentError("targets must not be empty"))

    total = 0.0

    @inbounds for position in eachindex(targets)
        total += cross_entropy(@view(logits[:, position]), targets[position])
    end

    return total / length(targets)
end

function perplexity_from_loss(loss::Real)
    loss_value = float(loss)

    isfinite(loss_value) ||
        return Inf

    loss_value >= log(floatmax(Float64)) &&
        return Inf

    return exp(loss_value)
end

function _softmax_probabilities(
    logits::AbstractVector{<:Real};
    temperature::Real = 1.0,
)
    temperature_value = _validate_temperature(temperature)
    scaled = Float64[float(logit) / temperature_value for logit in logits]
    normalizer = logsumexp(scaled)

    return exp.(scaled .- normalizer)
end
