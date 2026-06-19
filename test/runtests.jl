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
