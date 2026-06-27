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

function prepare_tokenizer(config)
    text = read(abspath(joinpath(project_root(), config["data"]["corpus"])), String)
    return MiniGPT.CharacterTokenizer(text), text
end

function main()
    config = load_config()
    loaded = MiniGPT.load_checkpoint(checkpoint_path(config))
    tokenizer, text = prepare_tokenizer(config)
    prompt_text = length(ARGS) >= 1 ? ARGS[1] : first(split(text, "\n"))
    prompt_ids = MiniGPT.encode(tokenizer, prompt_text)

    sample_ids = MiniGPT.generate_ids(
        loaded.model,
        prompt_ids,
        config["generation"]["max_new_tokens"];
        seed = config["generation"]["seed"],
        temperature = config["generation"]["temperature"],
        strategy = :sample,
    )
    greedy_ids = MiniGPT.generate_ids(
        loaded.model,
        prompt_ids,
        min(120, config["generation"]["max_new_tokens"]);
        strategy = :greedy,
    )

    println("Prompt:")
    println(prompt_text)
    println()
    println("Sample:")
    println(MiniGPT.decode(tokenizer, sample_ids))
    println()
    println("Greedy:")
    println(MiniGPT.decode(tokenizer, greedy_ids))
end

main()
