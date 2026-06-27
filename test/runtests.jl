using Test
using MiniGPT
using Unicode

const EXPECTED_TOKENS = 1677
const EXPECTED_VOCAB_SIZE = 47
const EXPECTED_TRAIN_TOKENS = 1341
const EXPECTED_VALIDATION_TOKENS = 167
const EXPECTED_TEST_TOKENS = 169

function corpus_path()
    return normpath(joinpath(@__DIR__, "..", "data", "raw", "tiny_corpus.txt"))
end

function frequency_counts(text)
    counts = Dict{Char, Int}()

    for ch in text
        counts[ch] = get(counts, ch, 0) + 1
    end

    return counts
end

mutable struct CountingRNG
    state::Int
end

function Base.rand(rng::CountingRNG, range::UnitRange{Int}, count::Integer)
    starts = Vector{Int}(undef, count)

    for i in eachindex(starts)
        rng.state += 1
        starts[i] = first(range) + mod(rng.state, length(range))
    end

    return starts
end

@testset "MiniGPT tests" begin
    @test MiniGPT.hello() == "MiniGPT is working"
end

@testset "tiny character corpus fixture" begin
    text = read(corpus_path(), String)
    tokens = collect(text)

    @test length(text) == EXPECTED_TOKENS
    @test length(Set(text)) == EXPECTED_VOCAB_SIZE
    @test Unicode.normalize(text, :NFC) == text
    @test !startswith(text, "\ufeff")
    @test !occursin('\r', text)

    train_text = String(tokens[1:EXPECTED_TRAIN_TOKENS])
    validation_text = String(tokens[(EXPECTED_TRAIN_TOKENS + 1):(EXPECTED_TRAIN_TOKENS + EXPECTED_VALIDATION_TOKENS)])
    test_text = String(tokens[(EXPECTED_TRAIN_TOKENS + EXPECTED_VALIDATION_TOKENS + 1):end])

    @test length(train_text) == EXPECTED_TRAIN_TOKENS
    @test length(validation_text) == EXPECTED_VALIDATION_TOKENS
    @test length(test_text) == EXPECTED_TEST_TOKENS
    @test length(train_text) + length(validation_text) + length(test_text) == length(text)

    train_vocab = Set(train_text)
    validation_vocab = Set(validation_text)
    test_vocab = Set(test_text)

    @test train_vocab == Set(text)
    @test isempty(setdiff(validation_vocab, train_vocab))
    @test isempty(setdiff(test_vocab, train_vocab))

    frequencies = frequency_counts(text)
    @test all(count >= 2 for count in values(frequencies))

    tokenizer = CharacterTokenizer(text)
    @test length(vocabulary(tokenizer)) == EXPECTED_VOCAB_SIZE
    @test decode(tokenizer, encode(tokenizer, text)) == text
end

@testset "reusable token data pipeline" begin
    text = read(corpus_path(), String)
    tokenizer = CharacterTokenizer(text)
    ids = encode(tokenizer, text)
    splits = split_token_ids(ids)

    @test splits isa TokenSplits
    @test length(splits.train) == EXPECTED_TRAIN_TOKENS
    @test length(splits.validation) == EXPECTED_VALIDATION_TOKENS
    @test length(splits.test) == EXPECTED_TEST_TOKENS
    @test vcat(splits.train, splits.validation, splits.test) == ids

    x_example, y_example = next_token_example(ids, 1, 16)
    @test length(x_example) == 16
    @test length(y_example) == 16
    @test x_example[2:end] == y_example[1:end-1]
    @test decode(tokenizer, x_example) == String(collect(text)[1:16])
    @test decode(tokenizer, y_example) == String(collect(text)[2:17])

    x, y = get_batch(splits.train; context_length = 16, batch_size = 4, seed = 42)
    @test size(x) == (16, 4)
    @test size(y) == (16, 4)
    @test x[2:end, :] == y[1:end-1, :]

    x_again, y_again = get_batch(splits.train; context_length = 16, batch_size = 4, seed = 42)
    @test x == x_again
    @test y == y_again

    rng = CountingRNG(0)
    first_x, first_y = get_batch(splits.train; context_length = 16, batch_size = 4, rng = rng)
    second_x, second_y = get_batch(splits.train; context_length = 16, batch_size = 4, rng = rng)

    reset_rng = CountingRNG(0)
    reset_x, reset_y = get_batch(splits.train; context_length = 16, batch_size = 4, rng = reset_rng)

    @test first_x == reset_x
    @test first_y == reset_y
    @test rng.state == 8
    @test first_x != second_x || first_y != second_y

    @test_throws ArgumentError split_token_ids(ids; train_ratio = 0.9, validation_ratio = 0.2)
    @test_throws ArgumentError next_token_example(ids, 0, 16)
    @test_throws ArgumentError next_token_example(ids, 1, length(ids))
    @test_throws ArgumentError get_batch(ids; context_length = 0, batch_size = 4)
    @test_throws ArgumentError get_batch(ids; context_length = 16, batch_size = 0)
    @test_throws ArgumentError get_batch(ids[1:16]; context_length = 16, batch_size = 4)
end

include("test_count_models.jl")
include("test_neural_bigram.jl")
