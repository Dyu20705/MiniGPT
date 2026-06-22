using Printf
using SHA
import MiniGPT

const CORPUS_RELATIVE_PATH = joinpath("data", "raw", "tiny_corpus.txt")
const TRAIN_RATIO = 0.8
const VALIDATION_RATIO = 0.1
const ALPHA = 0.1
const SEED = 42
const SAMPLE_LENGTH = 300
const SWEEP_ALPHAS = (0.0, 0.01, 0.1, 1.0)

function project_root()
    return normpath(joinpath(@__DIR__, ".."))
end

function corpus_path()
    return abspath(joinpath(project_root(), CORPUS_RELATIVE_PATH))
end

function prepare_data()
    path = corpus_path()
    isfile(path) || throw(ArgumentError("corpus file does not exist: $path"))

    bytes = read(path)
    text = String(copy(bytes))
    tokenizer = MiniGPT.CharacterTokenizer(text)
    ids = MiniGPT.encode(tokenizer, text)
    splits = MiniGPT.split_token_ids(
        ids;
        train_ratio = TRAIN_RATIO,
        validation_ratio = VALIDATION_RATIO,
    )

    return (
        path = path,
        sha256 = bytes2hex(sha256(bytes)),
        tokenizer = tokenizer,
        vocab_size = length(MiniGPT.vocabulary(tokenizer)),
        train_ids = splits.train,
        validation_ids = splits.validation,
        test_ids = splits.test,
    )
end

metric(value::Real) = isinf(value) ? "Inf" : @sprintf("%.6f", value)

function token_repr(tokenizer, id::Integer)
    return repr(MiniGPT.decode(tokenizer, [Int(id)]))
end

function assert_count_model_invariants(unigram, bigram, train_ids)
    @assert sum(unigram.counts) == length(train_ids)
    @assert sum(bigram.counts) == length(train_ids) - 1
    @assert isapprox(sum(unigram.prob), 1.0; atol = 1e-12)
    @assert size(bigram.probs) == size(bigram.counts)
    @assert !any(isnan, unigram.prob)
    @assert !any(isnan, unigram.log_prob)
    @assert !any(isnan, bigram.probs)
    @assert !any(isnan, bigram.logprobs)

    for previous in axes(bigram.probs, 1)
        @assert isapprox(sum(@view bigram.probs[previous, :]), 1.0; atol = 1e-12)
    end

    return nothing
end

function fair_reports(unigram, bigram, ids)
    length(ids) >= 2 ||
        throw(ArgumentError("split must contain at least two tokens"))

    unigram_report = MiniGPT.evaluate(unigram, ids[2:end])
    bigram_report = MiniGPT.evaluate(bigram, ids)
    unigram_report.prediction == bigram_report.prediction ||
        throw(AssertionError("unigram and bigram target counts differ"))

    return unigram_report, bigram_report
end

function print_report_table(unigram, bigram, train_ids, validation_ids, test_ids)
    println("Evaluation")
    @printf("%-12s %-8s %12s %14s %11s\n", "split", "model", "mean NLL", "perplexity", "predictions")
    println("-"^65)

    for (name, ids) in (
        ("train", train_ids),
        ("validation", validation_ids),
        ("test", test_ids),
    )
        unigram_report, bigram_report = fair_reports(unigram, bigram, ids)
        @printf(
            "%-12s %-8s %12s %14s %11d\n",
            name,
            "unigram",
            metric(unigram_report.nll),
            metric(unigram_report.perplexity),
            unigram_report.prediction,
        )
        @printf(
            "%-12s %-8s %12s %14s %11d\n",
            name,
            "bigram",
            metric(bigram_report.nll),
            metric(bigram_report.perplexity),
            bigram_report.prediction,
        )
    end

    println()
    return nothing
end

function print_most_common_bigrams(tokenizer, bigram)
    println("Most common bigrams from train")
    @printf("%4s %-12s %-12s %8s %14s\n", "rank", "previous", "next", "count", "P(next|prev)")
    println("-"^58)

    for (rank, (previous, next, count, probability)) in enumerate(MiniGPT.most_common_bigrams(bigram; k = 20))
        @printf(
            "%4d %-12s %-12s %8d %14.8f\n",
            rank,
            token_repr(tokenizer, previous),
            token_repr(tokenizer, next),
            count,
            probability,
        )
    end

    println()
    return nothing
end

function print_generation(tokenizer, unigram, bigram, train_ids)
    prompt = [train_ids[1]]
    unigram_ids = MiniGPT.generate_ids(unigram, SAMPLE_LENGTH; seed = SEED)
    unigram_ids_again = MiniGPT.generate_ids(unigram, SAMPLE_LENGTH; seed = SEED)
    bigram_ids = MiniGPT.generate_ids(bigram, prompt, SAMPLE_LENGTH; seed = SEED)
    bigram_ids_again = MiniGPT.generate_ids(bigram, prompt, SAMPLE_LENGTH; seed = SEED)

    @assert unigram_ids == unigram_ids_again
    @assert bigram_ids == bigram_ids_again

    println("Generation")
    println("Unigram sample ($(SAMPLE_LENGTH) tokens):")
    println("```text")
    println(MiniGPT.decode(tokenizer, unigram_ids))
    println("```")
    println()
    println("Bigram sample (prompt + $(SAMPLE_LENGTH) new tokens):")
    println("```text")
    println(MiniGPT.decode(tokenizer, bigram_ids))
    println("```")
    println()

    return unigram_ids, bigram_ids
end

function finite_status(report)
    return isfinite(report.nll) && isfinite(report.perplexity) ? "finite" : "Inf"
end

function best_finite_row(rows, score)
    best = nothing
    best_score = Inf

    for row in rows
        value = score(row)
        if isfinite(value) && value < best_score
            best = row
            best_score = value
        end
    end

    best === nothing &&
        throw(ArgumentError("no finite validation score was available"))

    return best
end

function smoothing_sweep(train_ids, validation_ids, test_ids, vocab_size)
    rows = NamedTuple[]

    println("Smoothing sweep")
    @printf(
        "%8s %14s %14s %14s %14s %14s %10s\n",
        "alpha",
        "uni val NLL",
        "uni val PPL",
        "bi val NLL",
        "bi val PPL",
        "bi test PPL",
        "status",
    )
    println("-"^96)

    for alpha in SWEEP_ALPHAS
        unigram = MiniGPT.fit_unigram(train_ids, vocab_size; alpha = alpha)
        bigram = MiniGPT.fit_bigram(train_ids, vocab_size; alpha = alpha, backoff_alpha = ALPHA)
        unigram_validation, bigram_validation = fair_reports(unigram, bigram, validation_ids)
        _, bigram_test = fair_reports(unigram, bigram, test_ids)
        status = "$(finite_status(unigram_validation))/$(finite_status(bigram_validation))"

        row = (
            alpha = alpha,
            unigram_validation = unigram_validation,
            bigram_validation = bigram_validation,
            bigram_test = bigram_test,
            status = status,
        )
        push!(rows, row)

        @printf(
            "%8.2f %14s %14s %14s %14s %14s %10s\n",
            alpha,
            metric(unigram_validation.nll),
            metric(unigram_validation.perplexity),
            metric(bigram_validation.nll),
            metric(bigram_validation.perplexity),
            metric(bigram_test.perplexity),
            status,
        )
    end

    unigram_best = best_finite_row(rows, row -> row.unigram_validation.nll)
    bigram_best = best_finite_row(rows, row -> row.bigram_validation.nll)

    println()
    println("Best unigram alpha by validation NLL: $(unigram_best.alpha)")
    println("Best bigram alpha by validation NLL: $(bigram_best.alpha)")
    println()

    return rows
end

function main()
    data = prepare_data()
    train_ids = data.train_ids
    validation_ids = data.validation_ids
    test_ids = data.test_ids
    vocab_size = data.vocab_size

    unigram = MiniGPT.fit_unigram(train_ids, vocab_size; alpha = ALPHA)
    bigram = MiniGPT.fit_bigram(train_ids, vocab_size; alpha = ALPHA, backoff_alpha = ALPHA)

    assert_count_model_invariants(unigram, bigram, train_ids)

    println("Phase 1 count model inspection")
    println("Corpus path: $(data.path)")
    println("SHA-256: $(data.sha256)")
    println("Vocabulary size: $vocab_size")
    println("Train tokens: $(length(train_ids))")
    println("Validation tokens: $(length(validation_ids))")
    println("Test tokens: $(length(test_ids))")
    println("Unigram count sum: $(sum(unigram.counts))")
    println("Bigram count sum: $(sum(bigram.counts))")
    println("Validation bigram targets: $(length(validation_ids) - 1)")
    println("Test bigram targets: $(length(test_ids) - 1)")
    println("Alpha: $ALPHA")
    println("Seed: $SEED")
    println()

    print_report_table(unigram, bigram, train_ids, validation_ids, test_ids)
    print_most_common_bigrams(data.tokenizer, bigram)
    print_generation(data.tokenizer, unigram, bigram, train_ids)
    smoothing_sweep(train_ids, validation_ids, test_ids, vocab_size)

    println("Phase 1 inspection completed successfully.")
end

main()
