using Printf
using SHA
using Unicode

const CORPUS_RELATIVE_PATH = joinpath("data", "raw", "tiny_corpus.txt")
const EXPECTED_TOKENS = 1677
const EXPECTED_VOCAB_SIZE = 47
const TRAIN_RATIO = 0.8
const VALIDATION_RATIO = 0.1
const EXPECTED_TRAIN_TOKENS = 1341
const EXPECTED_VALIDATION_TOKENS = 167
const EXPECTED_TEST_TOKENS = 169

function project_root()
    return normpath(joinpath(@__DIR__, ".."))
end

function corpus_path()
    return abspath(joinpath(project_root(), CORPUS_RELATIVE_PATH))
end

function fail(message)
    println(stderr, "Error: $message")
    exit(1)
end

function warn!(warnings, message)
    push!(warnings, message)
    return nothing
end

function validate_file(path)
    ispath(path) || fail("Corpus path does not exist: $path")
    isfile(path) || fail("Corpus path is not a file: $path")
    filesize(path) > 0 || fail("Corpus file is empty: $path")

    try
        open(path, "r") do _
        end
    catch err
        fail("Corpus file cannot be opened for reading: $path ($(typeof(err)))")
    end

    return nothing
end

function read_utf8(path)
    bytes = read(path)

    try
        return bytes, String(copy(bytes))
    catch err
        if isa(err, ArgumentError) || isa(err, UnicodeError)
            fail("Corpus is not valid UTF-8: $path")
        end

        rethrow(err)
    end
end

function char_tokens(text)
    return collect(text)
end

function deterministic_vocabulary(tokens)
    return sort!(collect(Set(tokens)))
end

function split_tokens(tokens)
    total = length(tokens)
    train_end = floor(Int, TRAIN_RATIO * total)
    validation_count = floor(Int, VALIDATION_RATIO * total)
    validation_end = train_end + validation_count

    train = tokens[1:train_end]
    validation = tokens[(train_end + 1):validation_end]
    test = tokens[(validation_end + 1):end]

    length(train) + length(validation) + length(test) == total ||
        fail("Split sizes do not add up to total token count.")

    return train, validation, test
end

function build_tokenizer(vocabulary)
    stoi = Dict(ch => id for (id, ch) in enumerate(vocabulary))
    itos = Dict(id => ch for (id, ch) in enumerate(vocabulary))

    length(stoi) == length(vocabulary) ||
        fail("Vocabulary is not injective: at least two characters share a token id.")
    length(itos) == length(vocabulary) ||
        fail("ID map is not injective: at least two token ids share a character.")

    return stoi, itos
end

function encode(tokens, stoi)
    ids = Vector{Int}(undef, length(tokens))

    for (i, ch) in enumerate(tokens)
        id = get(stoi, ch, nothing)
        id === nothing && fail("Encode failed: character $(describe_char(ch)) is not in vocabulary.")
        ids[i] = id
    end

    return ids
end

function decode(ids, itos)
    chars = Vector{Char}(undef, length(ids))

    for (i, id) in enumerate(ids)
        ch = get(itos, id, nothing)
        ch === nothing && fail("Decode failed: token id $id is outside the vocabulary.")
        chars[i] = ch
    end

    return String(chars)
end

function describe_char(ch)
    if ch == ' '
        label = "Space"
    elseif ch == '\n'
        label = "Newline"
    elseif ch == '\r'
        label = "Carriage return"
    elseif ch == '\t'
        label = "Tab"
    elseif ch == '\ufeff'
        label = "BOM"
    elseif ch == '\ufffd'
        label = "Replacement character"
    elseif iscntrl(ch)
        label = "Control"
    else
        label = string(ch)
    end

    return @sprintf("%s U+%04X", label, Int(ch))
end

function count_crlf(tokens)
    pairs = 0

    for i in 1:(length(tokens) - 1)
        if tokens[i] == '\r' && tokens[i + 1] == '\n'
            pairs += 1
        end
    end

    return pairs
end

function count_lines(tokens)
    isempty(tokens) && return 0

    newline_count = count(==('\n'), tokens)
    return newline_count + (last(tokens) == '\n' ? 0 : 1)
end

function frequency_table(tokens)
    counts = Dict{Char, Int}()

    for ch in tokens
        counts[ch] = get(counts, ch, 0) + 1
    end

    return counts
end

function sorted_frequencies(counts)
    pairs = collect(counts)
    sort!(pairs, by = pair -> (-pair.second, Int(pair.first)))
    return pairs
end

function render_frequencies(pairs; limit = 5)
    isempty(pairs) && return "(none)"

    shown = first(pairs, min(limit, length(pairs)))
    return join(
        ["$(describe_char(pair.first))=$(pair.second)" for pair in shown],
        ", ",
    )
end

function first_roundtrip_mismatch(original_tokens, decoded_tokens)
    limit = min(length(original_tokens), length(decoded_tokens))

    for i in 1:limit
        if original_tokens[i] != decoded_tokens[i]
            return i
        end
    end

    if length(original_tokens) != length(decoded_tokens)
        return limit + 1
    end

    return nothing
end

function context_window(tokens, center; radius = 8)
    start_index = max(1, center - radius)
    end_index = min(length(tokens), center + radius)
    return String(tokens[start_index:end_index])
end

function verify_roundtrip(original_text, original_tokens, decoded_text)
    decoded_tokens = char_tokens(decoded_text)
    mismatch = first_roundtrip_mismatch(original_tokens, decoded_tokens)

    if mismatch !== nothing
        println(stderr, "Round-trip mismatch at token position: $mismatch")

        if mismatch <= length(original_tokens)
            println(stderr, "Original: $(describe_char(original_tokens[mismatch]))")
            println(stderr, "Original context: $(repr(context_window(original_tokens, mismatch)))")
        else
            println(stderr, "Original ended before decoded text.")
        end

        if mismatch <= length(decoded_tokens)
            println(stderr, "Decoded: $(describe_char(decoded_tokens[mismatch]))")
            println(stderr, "Decoded context: $(repr(context_window(decoded_tokens, mismatch)))")
        else
            println(stderr, "Decoded ended before original text.")
        end

        fail("Round-trip check failed.")
    end

    original_text == decoded_text || fail("Round-trip strings are not exactly equal.")
    return true
end

function inspect_unicode(tokens, text, warnings)
    Unicode.normalize(text, :NFC) == text ||
        warn!(warnings, "Corpus is not normalized as Unicode NFC.")

    !isempty(tokens) && first(tokens) == '\ufeff' &&
        warn!(warnings, "Corpus starts with a byte order mark (BOM).")

    count(==('\ufffd'), tokens) == 0 ||
        warn!(warnings, "Corpus contains Unicode replacement characters (U+FFFD).")

    cr_count = count(==('\r'), tokens)
    tab_count = count(==('\t'), tokens)
    abnormal_controls = [
        ch for ch in tokens
        if iscntrl(ch) && !(ch in ('\n', '\r', '\t'))
    ]

    cr_count == 0 ||
        warn!(warnings, "Corpus contains carriage return characters.")
    tab_count == 0 ||
        warn!(warnings, "Corpus contains tab characters.")
    isempty(abnormal_controls) ||
        warn!(warnings, "Corpus contains unusual control characters.")

    return nothing
end

function inspect_splits(train, validation, test, vocabulary)
    train_vocab = Set(train)
    validation_vocab = Set(validation)
    test_vocab = Set(test)

    validation_only = setdiff(validation_vocab, train_vocab)
    test_only = setdiff(test_vocab, train_vocab)
    missing_from_train = setdiff(Set(vocabulary), train_vocab)

    return train_vocab, validation_only, test_only, missing_from_train
end

function check_expected_counts(tokens, vocabulary, train, validation, test, warnings)
    length(tokens) == EXPECTED_TOKENS ||
        warn!(warnings, "Expected $EXPECTED_TOKENS character tokens, found $(length(tokens)).")
    length(vocabulary) == EXPECTED_VOCAB_SIZE ||
        warn!(warnings, "Expected vocabulary size $EXPECTED_VOCAB_SIZE, found $(length(vocabulary)).")
    length(train) == EXPECTED_TRAIN_TOKENS ||
        warn!(warnings, "Expected $EXPECTED_TRAIN_TOKENS train tokens, found $(length(train)).")
    length(validation) == EXPECTED_VALIDATION_TOKENS ||
        warn!(warnings, "Expected $EXPECTED_VALIDATION_TOKENS validation tokens, found $(length(validation)).")
    length(test) == EXPECTED_TEST_TOKENS ||
        warn!(warnings, "Expected $EXPECTED_TEST_TOKENS test tokens, found $(length(test)).")

    return nothing
end

function check_distribution(
    vocabulary,
    train_vocab,
    validation_only,
    test_only,
    missing_from_train,
    singleton_count,
    warnings,
)
    if singleton_count > 0
        warn!(warnings, "Corpus contains $singleton_count singleton vocabulary character(s).")
    end

    if length(train_vocab) != length(vocabulary)
        warn!(
            warnings,
            "Train vocabulary size $(length(train_vocab)) differs from full vocabulary size $(length(vocabulary)).",
        )
    end

    isempty(validation_only) ||
        warn!(
            warnings,
            "Validation split contains characters absent from train: $(render_frequencies([ch => 1 for ch in sort!(collect(validation_only))]))",
        )

    isempty(test_only) ||
        warn!(
            warnings,
            "Test split contains characters absent from train: $(render_frequencies([ch => 1 for ch in sort!(collect(test_only))]))",
        )

    isempty(missing_from_train) ||
        warn!(
            warnings,
            "Full-corpus vocabulary contains characters missing from train: $(render_frequencies([ch => 1 for ch in sort!(collect(missing_from_train))]))",
        )

    return nothing
end

function main()
    path = corpus_path()
    warnings = String[]

    validate_file(path)
    bytes, text = read_utf8(path)
    tokens = char_tokens(text)
    vocabulary = deterministic_vocabulary(tokens)
    train, validation, test = split_tokens(tokens)
    stoi, itos = build_tokenizer(vocabulary)
    encoded = encode(tokens, stoi)

    length(encoded) == length(tokens) ||
        fail("Encoded token count does not match character token count.")
    all(id -> 1 <= id <= length(vocabulary), encoded) ||
        fail("Encoded ids contain values outside the vocabulary range.")
    length(Set(encoded)) == length(vocabulary) ||
        fail("Encoded ids do not cover the full vocabulary.")

    decoded_text = decode(encoded, itos)
    decoded_tokens = char_tokens(decoded_text)

    length(decoded_tokens) == length(tokens) ||
        fail("Decoded character count does not match original token count.")

    roundtrip_correct = verify_roundtrip(text, tokens, decoded_text)

    inspect_unicode(tokens, text, warnings)
    check_expected_counts(tokens, vocabulary, train, validation, test, warnings)

    counts = frequency_table(tokens)
    frequencies = sorted_frequencies(counts)
    singleton_count = count(pair -> pair.second == 1, frequencies)
    newline_count = count(==('\n'), tokens)
    cr_count = count(==('\r'), tokens)
    crlf_count = count_crlf(tokens)
    standalone_cr_count = cr_count - crlf_count
    space_count = count(==(' '), tokens)
    train_vocab, validation_only, test_only, missing_from_train =
        inspect_splits(train, validation, test, vocabulary)

    check_distribution(
        vocabulary,
        train_vocab,
        validation_only,
        test_only,
        missing_from_train,
        singleton_count,
        warnings,
    )

    println("Corpus path: $path")
    println("File size bytes: $(length(bytes))")
    println("SHA-256: $(bytes2hex(sha256(bytes)))")
    println("UTF-8 readable: true")
    println("Unicode NFC: $(Unicode.normalize(text, :NFC) == text)")
    println("Lines: $(count_lines(tokens))")
    println("Newlines LF: $newline_count")
    println("Carriage returns CR: $cr_count")
    println("CRLF pairs: $crlf_count")
    println("Standalone CR: $standalone_cr_count")
    println("Spaces: $space_count")
    println("Tabs: $(count(==('\t'), tokens))")
    println()
    println("Characters/tokens: $(length(tokens))")
    println("Vocabulary size: $(length(vocabulary))")
    println("Train tokens: $(length(train))")
    println("Validation tokens: $(length(validation))")
    println("Test tokens: $(length(test))")
    println("Round-trip correct: $roundtrip_correct")
    println()
    println("Most frequent characters: $(render_frequencies(frequencies))")
    println("Least frequent characters: $(render_frequencies(reverse(frequencies)))")
    println("Singleton tokens: $singleton_count")
    println("Train vocabulary size: $(length(train_vocab))")
    println("Validation-only characters: $(render_frequencies([ch => counts[ch] for ch in sort!(collect(validation_only))]))")
    println("Test-only characters: $(render_frequencies([ch => counts[ch] for ch in sort!(collect(test_only))]))")
    println("Characters missing from train: $(render_frequencies([ch => counts[ch] for ch in sort!(collect(missing_from_train))]))")

    if isempty(warnings)
        println("Warnings: none")
    else
        println("Warnings:")
        for warning in warnings
            println("- $warning")
        end
    end
end

main()
