struct TokenSplits
    train::Vector{Int}
    validation::Vector{Int}
    test::Vector{Int}
end

function split_token_ids(
    ids::AbstractVector{<:Integer};
    train_ratio::Real = 0.8,
    validation_ratio::Real = 0.1,
)
    0 <= train_ratio <= 1 ||
        throw(ArgumentError("train_ratio must be between 0 and 1"))
    0 <= validation_ratio <= 1 ||
        throw(ArgumentError("validation_ratio must be between 0 and 1"))
    train_ratio + validation_ratio <= 1 ||
        throw(ArgumentError("train_ratio + validation_ratio must be at most 1"))

    total = length(ids)
    train_count = floor(Int, train_ratio * total)
    validation_count = floor(Int, validation_ratio * total)
    validation_end = train_count + validation_count

    train = Int.(ids[1:train_count])
    validation = Int.(ids[(train_count + 1):validation_end])
    test = Int.(ids[(validation_end + 1):end])

    length(train) + length(validation) + length(test) == total ||
        throw(AssertionError("split sizes must add up to the input length"))

    return TokenSplits(train, validation, test)
end
