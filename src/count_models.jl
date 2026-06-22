
using Random

abstract type AbstractCountLanguageModel end

"""
LMReport: holds the results of a count-based language model evaluation.
- nll: negative log-likelihood of the model on the evaluation data
- perplexity: perplexity of the model on the evaluation data
- prediction: the number of correct predictions made by the model on the evaluation data
"""

struct LMReport
    nll::Float64
    perplexity::Float64
    prediction::Int
end

"""
UnigramCountModel: a count-based language model that uses unigram counts to estimate probabilities of words in a vocabulary.
- counts[id]: the count of the word with the given id in the training data
- prob[id]: the probability of the word with the given id in the training data
- log_prob[id]: the log probability of the word with the given id in the training data
- alpha: the smoothing parameter used to avoid zero probabilities for unseen words
"""

struct UnigramCountModel <: AbstractCountLanguageModel
    counts::Vector{Int}
    prob::Vector{Float64}
    log_prob::Vector{Float64}
    alpha::Float64
end

const UnigramCountLanguageModel = UnigramCountModel

"""
Bigram count model.

counts[previous, next]   = số transition previous → next.
probs[previous, next]    = P(next | previous).
logprobs[previous, next] = log(P(next | previous)).
"""
struct BigramCountModel <: AbstractCountLanguageModel
    counts::Matrix{Int}
    probs::Matrix{Float64}
    logprobs::Matrix{Float64}
    backoff_probs::Vector{Float64}
    alpha::Float64
end

"""
validate_vocab_size: function to validate the size of the vocabulary used in the language model.
- vocab_size: the size of the vocabulary to be validated
- Throws an error if the vocabulary size is not > 0.
- Returns the validated vocabulary size if it is valid.
"""

function _validate_vocab_size(vocab_size::Integer)
    vocab_size > 0 ||
        throw(ArgumentError("Vocabulary size must be greater than 0."))

    return Int(vocab_size)
end

"""
validate_alpha: function to validate the smoothing parameter alpha used in the language model.
- alpha: the smoothing parameter to be validated
- Throws an error if alpha is not finite or is less than 0.
- Returns the validated alpha if it is valid.
"""

function _validate_alpha(alpha::Real)
    α = float(alpha)

    isfinite(α) ||
        throw(ArgumentError("Alpha must be finite."))

    α >= 0 ||
        throw(ArgumentError("Alpha must be greater than or equal to 0."))

    return α
end

"""
validate_token_id: function to validate a token id used in the language model.
- tokens: an array of token ids to be validated
- vocab_size: the size of the vocabulary used in the language model
- Throws an error if any token id is not within the valid range of [1, vocab_size].
- Returns nothing if all token ids are valid.
"""

function _validate_token_id(
    tokens::AbstractVector{<:Integer},
    vocab_size::Integer
)
    @inbounds for (position, raw_id) in pairs(tokens)
        token = Int(raw_id)

        1 <= token <= vocab_size ||
            throw(ArgumentError(
                "Token id $token at position $position is out of bounds. " *
                "Valid range is [1, $vocab_size]."
            ))
    end

    return nothing
end

"""
safe_log: function to compute the logarithm of a value, returning -Inf for non-positive values.
- probability: the value for which to compute the logarithm
- Returns:
+ log(probability) if probability > 0
+ otherwise return -Inf.
"""

@inline function _safe_log(probability::Float64)
    return probability > 0.0 ? log(probability) : -Inf
end

"""
perplexity: function to compute the perplexity of a language model given the negative log-likelihood and the number of tokens.
- nll: negative log-likelihood of the model on the evaluation data
- return Inf if nll is Inf or NaN
- to avoid overflow in exp, compare nll with log(floatmax(Float64)) and return Inf if nll is greater than that value
- return exp(nll)
"""

function _perplexity(nll::Float64)
    isfinite(nll) ||
        return Inf

    nll >= log(floatmax(Float64)) &&
        return Inf

    return exp(nll)
end

"""
Sample a categorical distribution given a vector of probabilities, no need StatsBase, just use the built-in rand function.
"""

function _sample_categorical(
    rng::AbstractRNG,
    probabilities::AbstractVector{<:Real},
)
    threshold = rand(rng)
    cumulative = 0.0

    @inbounds for id in eachindex(probabilities)
        cumulative += probabilities[id]

        if threshold < cumulative
            return Int(id)
        end
    end

    # protect against floating point errors, return the last index if threshold is not reached
    return Int(lastindex(probabilities))
end

"""
Fit unigram model from token IDs

Theoretically,

Unigram assumes: each token is independent of the others.

That is: P(x1, x2, ..., xn) = P(x1) * P(x2) * ... * P(xn)

Where xi is a token in the vocabulary.

Probability of a token is estimated as:
    P(a) = count(a) + alpha / N + alpha * V

alpha = 0.0: pure MLE, no smoothing
alpha > 0.0: additive smoothing
"""

function fit_unigram(
    tokens::AbstractVector{<:Integer},
    vocab_size::Integer;
    alpha::Real = 0.1,
)
    V = _validate_vocab_size(vocab_size)
    α = _validate_alpha(alpha)

    isempty(tokens) &&
        throw(ArgumentError("Cannot fit a unigram model on an empty token array."))
    
    _validate_token_id(tokens, V)

    counts = zeros(Int, V)

    @inbounds for raw_id in tokens
        counts[Int(raw_id)] += 1
    end

    denominator = length(tokens) + α * V

    denominator > 0.0 ||
        throw(ArgumentError("Denominator must be greater than 0. Check your alpha and token counts."))

    probs = Vector{Float64}(undef, V)
    logprobs = Vector{Float64}(undef, V)

    @inbounds for id in 1:V
        probability = (counts[id] + α) / denominator
        probs[id] = probability
        logprobs[id] = _safe_log(probability)
    end

    return UnigramCountLanguageModel(
        counts,
        probs,
        logprobs,
        α
    )
end

"""
Fit bigram model on transitions tokens[t] → tokens[t + 1]

Bigram assumes: each token depends only on the previous token.

That is: P(x1, x2, ..., xn) = P(x1) * P(x2 | x1) * P(x3 | x2) * ... * P(xn | xn-1)

Probability of a transition is estimated as:
    P(a | b) = (count(b → a) + alpha) / (count(b) + alpha * V)

When alpha == 0 and a token has never had an outgoing transition,
the model uses the unigram distribution as backoff.
"""
function fit_bigram(
    tokens::AbstractVector{<:Integer},
    vocab_size::Integer;
    alpha::Real = 0.1,
    backoff_alpha::Real = 0.1,
)
    V = _validate_vocab_size(vocab_size)
    α = _validate_alpha(alpha)
    β = _validate_alpha(backoff_alpha)

    length(tokens) >= 2 ||
        throw(ArgumentError("bigram model requires at least two tokens"))

    _validate_token_id(tokens, V)

    counts = zeros(Int, V, V)

    @inbounds for position in 1:(length(tokens) - 1)
        previous = Int(tokens[position])
        next = Int(tokens[position + 1])

        counts[previous, next] += 1
    end

    backoff = fit_unigram(
        tokens,
        V;
        alpha = β,
    )

    probs = Matrix{Float64}(undef, V, V)
    logprobs = Matrix{Float64}(undef, V, V)

    @inbounds for previous in 1:V
        row_total = 0

        for next in 1:V
            row_total += counts[previous, next]
        end

        if row_total == 0 && α == 0.0
            # Token exist but never had an outgoing transition.
            # Use the unigram distribution as backoff.
            for next in 1:V
                probability = backoff.prob[next]
                probs[previous, next] = probability
                logprobs[previous, next] = _safe_log(probability)
            end

            continue
        end

        denominator = row_total + α * V

        @inbounds for next in 1:V
            probability =
                (counts[previous, next] + α) / denominator

            probs[previous, next] = probability
            logprobs[previous, next] = _safe_log(probability)
        end
    end

    return BigramCountModel(
        counts,
        probs,
        logprobs,
        copy(backoff.prob),
        α,
    )
end

"""
Evaluate a unigram model on a set of token IDs
"""

function evaluate(
    model::UnigramCountModel,
    tokens::AbstractVector{<:Integer},
)
    isempty(tokens) &&
        throw(ArgumentError("Cannot evaluate a unigram model on an empty token array."))
    
    V = length(model.counts)
    _validate_token_id(tokens, V)

    total_nll = 0.0

    @inbounds for raw_id in tokens
        id = Int(raw_id)
        log_prob = model.log_prob[id]
        
        if !isfinite(log_prob)
            return LMReport(
                Inf,
                Inf,
                length(tokens)
            )
        end

        total_nll -= log_prob
    end

    mean_nll = total_nll / length(tokens)
    perplexity = _perplexity(mean_nll)

    return LMReport(
        mean_nll,
        perplexity,
        length(tokens)
    )
end

"""
Evaluate a bigram model on a set of token IDs.

A sequence of T tokens creates T - 1 predictions, since the first token has no previous token to condition on.
"""
function evaluate(
    model::BigramCountModel,
    tokens::AbstractVector{<:Integer},
)
    length(tokens) >= 2 ||
        throw(ArgumentError(
            "bigram evaluation requires at least two tokens",
        ))

    V = size(model.counts, 1)
    _validate_token_id(tokens, V)

    predictions = length(tokens) - 1
    total_negative_log_likelihood = 0.0

    @inbounds for position in 1:predictions
        previous = Int(tokens[position])
        next = Int(tokens[position + 1])

        log_probability = model.logprobs[previous, next]

        if !isfinite(log_probability)
            return LMReport(
                Inf,
                Inf,
                predictions,
            )
        end

        total_negative_log_likelihood -= log_probability
    end

    mean_nll =
        total_negative_log_likelihood / predictions

    return LMReport(
        mean_nll,
        _perplexity(mean_nll),
        predictions,
    )
end

"""
Create independently n_tokens from a unigram distribution, given a fitted UnigramCountLanguageModel.
"""

function generate_ids(
    model::UnigramCountModel,
    n_tokens::Integer;
    seed::Integer = 42,
)
    n_tokens > 0 ||
        throw(ArgumentError("Number of tokens to generate must be greater than 0."))

    rng = MersenneTwister(seed)
    output = Vector{Int}(undef, n_tokens)

    @inbounds for position in 1:n_tokens
        output[position] =
            _sample_categorical(rng, model.prob)
    end

    return output
end

"""
Create max_new_tokens by bigram model
Results include the initial prompt.
"""
function generate_ids(
    model::BigramCountModel,
    prompt_ids::AbstractVector{<:Integer},
    max_new_tokens::Integer;
    seed::Integer = 42,
)
    isempty(prompt_ids) &&
        throw(ArgumentError("bigram generation requires a prompt"))

    max_new_tokens >= 0 ||
        throw(ArgumentError(
            "max_new_tokens must be non-negative",
        ))

    V = size(model.counts, 1)
    _validate_token_id(prompt_ids, V)

    rng = MersenneTwister(seed)
    output = Int[Int(id) for id in prompt_ids]

    for _ in 1:max_new_tokens
        previous = output[end]

        next = _sample_categorical(
            rng,
            @view(model.probs[previous, :]),
        )

        push!(output, next)
    end

    return output
end


"""
Return bigrams with the highest counts.
Each bigram is represented as a tuple of (previous_id, next_id, count, probability).
"""
function most_common_bigrams(
    model::BigramCountModel;
    k::Integer = 20,
)
    k >= 0 ||
        throw(ArgumentError("k must be non-negative"))

    V = size(model.counts, 1)
    entries = Tuple{Int,Int,Int,Float64}[]

    @inbounds for previous in 1:V
        for next in 1:V
            count = model.counts[previous, next]

            count == 0 && continue

            push!(
                entries,
                (
                    previous,
                    next,
                    count,
                    model.probs[previous, next],
                ),
            )
        end
    end

    sort!(
        entries;
        by = entry -> (
            -entry[3],
            entry[1],
            entry[2],
        ),
    )

    result_length = min(k, length(entries))

    return entries[1:result_length]
end
