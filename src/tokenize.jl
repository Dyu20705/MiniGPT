struct CharacterTokenizer
    vocabulary::Vector{Char}
    stoi::Dict{Char, Int}
    itos::Dict{Int, Char}
end

function CharacterTokenizer(text::AbstractString)
    chars = sort!(collect(Set(text)))
    stoi = Dict(ch => id for (id, ch) in enumerate(chars))
    itos = Dict(id => ch for (id, ch) in enumerate(chars))
    return CharacterTokenizer(chars, stoi, itos)
end

vocabulary(tokenizer::CharacterTokenizer) = copy(tokenizer.vocabulary)

function encode(tokenizer::CharacterTokenizer, text::AbstractString)
    ids = Vector{Int}(undef, length(text))

    for (i, ch) in enumerate(text)
        id = get(tokenizer.stoi, ch, nothing)
        id === nothing && throw(ArgumentError("character is not in tokenizer vocabulary: $(repr(ch))"))
        ids[i] = id
    end

    return ids
end

function decode(tokenizer::CharacterTokenizer, ids::AbstractVector{<:Integer})
    chars = Vector{Char}(undef, length(ids))

    for (i, id) in enumerate(ids)
        ch = get(tokenizer.itos, Int(id), nothing)
        ch === nothing && throw(ArgumentError("token id is outside tokenizer vocabulary: $id"))
        chars[i] = ch
    end

    return String(chars)
end
