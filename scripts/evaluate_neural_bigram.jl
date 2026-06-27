using Printf
using TOML
import MiniGPT

const CONFIG_PATH = normpath(joinpath(@__DIR__, "..", "configs", "neural_bigram.toml"))

function project_root()
    return normpath(joinpath(@__DIR__, ".."))
end

function load_config()
    return TOML.parsefile(CONFIG_PATH)
end

function checkpoint_path(config)
    return abspath(joinpath(
        project_root(),
        config["checkpoint"]["directory"],
        config["checkpoint"]["filename"],
    ))
end

function prepare_splits(config)
    text = read(abspath(joinpath(project_root(), config["data"]["corpus"])), String)
    tokenizer = MiniGPT.CharacterTokenizer(text)
    ids = MiniGPT.encode(tokenizer, text)
    splits = MiniGPT.split_token_ids(
        ids;
        train_ratio = config["data"]["train_ratio"],
        validation_ratio = config["data"]["validation_ratio"],
    )

    return tokenizer, splits
end

metric(value::Real) = isinf(value) ? "Inf" : @sprintf("%.6f", value)

function main()
    config = load_config()
    path = checkpoint_path(config)
    loaded = MiniGPT.load_checkpoint(path)
    _, splits = prepare_splits(config)

    println("Phase 2 neural bigram evaluation")
    println("Checkpoint: $path")
    println("Checkpoint step: $(loaded.step)")
    println("Vocabulary size: $(loaded.vocab_size)")
    println("Orientation: $(loaded.orientation)")
    println()
    @printf("%-12s %12s %14s %10s %10s\n", "split", "NLL", "perplexity", "accuracy", "targets")
    println("-"^64)

    for (name, ids) in (("train", splits.train), ("validation", splits.validation), ("test", splits.test))
        report = MiniGPT.evaluate(loaded.model, ids)
        @printf(
            "%-12s %12s %14s %9.2f%% %10d\n",
            name,
            metric(report.nll),
            metric(report.perplexity),
            100 * report.accuracy,
            report.predictions,
        )
    end
end

main()
