function _seeded_starts(seed::Integer, max_start::Integer, batch_size::Integer)
    seed >= 0 || throw(ArgumentError("seed must be non-negative"))

    starts = Vector{Int}(undef, batch_size)
    state = UInt64(seed)

    for i in eachindex(starts)
        state = state * 0x5851f42d4c957f2d + 0x14057b7ef767814f
        starts[i] = Int(mod(state, UInt64(max_start))) + 1
    end

    return starts
end

function next_token_example(
    ids::AbstractVector{<:Integer},
    start::Integer,
    context_length::Integer,
)
    start >= 1 || throw(ArgumentError("start must be at least 1"))
    context_length >= 1 || throw(ArgumentError("context_length must be at least 1"))
    start + context_length <= length(ids) ||
        throw(ArgumentError("not enough tokens for start=$start and context_length=$context_length"))

    x = Int.(ids[start:(start + context_length - 1)])
    y = Int.(ids[(start + 1):(start + context_length)])

    return x, y
end

next_token_example(ids::AbstractVector{<:Integer}, context_length::Integer) =
    next_token_example(ids, 1, context_length)

function get_batch(
    ids::AbstractVector{<:Integer};
    context_length::Integer,
    batch_size::Integer,
    rng = nothing,
    seed = nothing,
)
    context_length >= 1 ||
        throw(ArgumentError("context_length must be at least 1"))
    batch_size >= 1 ||
        throw(ArgumentError("batch_size must be at least 1"))
    length(ids) >= context_length + 1 ||
        throw(ArgumentError("ids must contain at least context_length + 1 tokens"))

    max_start = length(ids) - context_length
    starts = if seed === nothing
        rng === nothing ? rand(1:max_start, batch_size) : rand(rng, 1:max_start, batch_size)
    else
        _seeded_starts(seed, max_start, batch_size)
    end

    x = Matrix{Int}(undef, context_length, batch_size)
    y = Matrix{Int}(undef, context_length, batch_size)

    for (column, start) in enumerate(starts)
        example_x, example_y = next_token_example(ids, start, context_length)
        x[:, column] = example_x
        y[:, column] = example_y
    end

    return x, y
end
