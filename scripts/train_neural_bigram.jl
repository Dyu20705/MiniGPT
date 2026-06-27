using Printf
using SHA
using TOML
import MiniGPT

const CONFIG_PATH = normpath(joinpath(@__DIR__, "..", "configs", "neural_bigram.toml"))

function project_root()
    return normpath(joinpath(@__DIR__, ".."))
end

function load_config()
    return TOML.parsefile(CONFIG_PATH)
end

function prepare_data(config)
    corpus = abspath(joinpath(project_root(), config["data"]["corpus"]))
    bytes = read(corpus)
    text = String(copy(bytes))
    tokenizer = MiniGPT.CharacterTokenizer(text)
    ids = MiniGPT.encode(tokenizer, text)
    splits = MiniGPT.split_token_ids(
        ids;
        train_ratio = config["data"]["train_ratio"],
        validation_ratio = config["data"]["validation_ratio"],
    )

    return (
        corpus = corpus,
        sha256 = bytes2hex(sha256(bytes)),
        tokenizer = tokenizer,
        vocab_size = length(MiniGPT.vocabulary(tokenizer)),
        train = splits.train,
        validation = splits.validation,
        test = splits.test,
    )
end

function dtype_from_config(config)
    name = get(config["model"], "dtype", "Float32")

    if name == "Float32"
        return Float32
    elseif name == "Float64"
        return Float64
    end

    throw(ArgumentError("unsupported dtype: $name"))
end

function init_from_config(config)
    name = get(config["model"], "init", "normal")
    name in ("normal", "zeros") ||
        throw(ArgumentError("unsupported init: $name"))

    return Symbol(name)
end

metric(value::Real) = isinf(value) ? "Inf" : @sprintf("%.6f", value)

function print_reports(model, data)
    println("Evaluation")
    @printf("%-12s %12s %14s %10s %10s\n", "split", "NLL", "perplexity", "accuracy", "targets")
    println("-"^64)

    for (name, ids) in (("train", data.train), ("validation", data.validation), ("test", data.test))
        report = MiniGPT.evaluate(model, ids)
        @printf(
            "%-12s %12s %14s %9.2f%% %10d\n",
            name,
            metric(report.nll),
            metric(report.perplexity),
            100 * report.accuracy,
            report.predictions,
        )
    end

    println()
    return nothing
end

function main()
    config = load_config()
    data = prepare_data(config)

    model = MiniGPT.NeuralBigramModel(
        data.vocab_size;
        seed = config["data"]["seed"],
        scale = config["model"]["init_scale"],
        dtype = dtype_from_config(config),
        init = init_from_config(config),
    )

    initial_train = MiniGPT.evaluate(model, data.train)
    expected_uniform = log(data.vocab_size)

    println("Phase 2 neural bigram training")
    println("Corpus: $(data.corpus)")
    println("SHA-256: $(data.sha256)")
    println("Vocabulary size: $(data.vocab_size)")
    println("Parameter shape: $(size(model.logits_table))")
    println("Parameter count: $(MiniGPT.parameter_count(model))")
    println("Initial train NLL: $(metric(initial_train.nll))")
    println("Expected uniform NLL: $(metric(expected_uniform))")
    println()

    state = MiniGPT.train!(
        model,
        data.train,
        data.validation;
        learning_rate = config["training"]["learning_rate"],
        batch_size = config["data"]["batch_size"],
        max_steps = config["training"]["max_steps"],
        eval_interval = config["training"]["eval_interval"],
        seed = config["data"]["seed"],
        eval_max_predictions = config["training"]["eval_max_predictions"],
    )

    println("Training history")
    @printf("%8s %12s %12s %12s %12s %12s\n", "step", "batch NLL", "grad norm", "train NLL", "val NLL", "val PPL")
    println("-"^76)

    for row in state.history
        @printf(
            "%8d %12s %12s %12s %12s %12s\n",
            row.step,
            metric(row.batch_loss),
            metric(row.gradient_norm),
            metric(row.train_nll),
            metric(row.validation_nll),
            metric(row.validation_perplexity),
        )
    end

    println()
    print_reports(model, data)

    best_model = MiniGPT.NeuralBigramModel(copy(state.best_logits_table))
    checkpoint_dir = abspath(joinpath(project_root(), config["checkpoint"]["directory"]))
    checkpoint_path = joinpath(checkpoint_dir, config["checkpoint"]["filename"])
    MiniGPT.save_checkpoint(
        checkpoint_path,
        best_model;
        tokenizer = data.tokenizer,
        step = state.best_step,
        config = config,
        best_validation_loss = state.best_validation_loss,
        seed = config["data"]["seed"],
    )

    println("Saved best checkpoint from step $(state.best_step): $checkpoint_path")
end

main()
