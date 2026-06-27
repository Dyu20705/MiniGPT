using Serialization

const NEURAL_BIGRAM_CHECKPOINT_VERSION = 1

function _checkpoint_vocabulary(tokenizer)
    tokenizer === nothing &&
        return nothing

    hasproperty(tokenizer, :vocabulary) ||
        throw(ArgumentError("tokenizer must expose a vocabulary field"))

    return copy(tokenizer.vocabulary)
end

function save_checkpoint(
    path::AbstractString,
    model::NeuralBigramModel;
    tokenizer = nothing,
    step::Integer = 0,
    optimizer_state = nothing,
    config = Dict{String, Any}(),
    best_validation_loss::Real = Inf,
    seed::Integer = 42,
)
    step >= 0 ||
        throw(ArgumentError("step must be non-negative"))
    seed >= 0 ||
        throw(ArgumentError("seed must be non-negative"))

    directory = dirname(path)

    if !isempty(directory)
        mkpath(directory)
    end

    payload = Dict{Symbol, Any}(
        :version => NEURAL_BIGRAM_CHECKPOINT_VERSION,
        :model_type => "neural_bigram",
        :orientation => "logits_table[next, current]",
        :logits_table => copy(model.logits_table),
        :vocab_size => vocab_size(model),
        :step => Int(step),
        :optimizer_state => optimizer_state,
        :config => config,
        :best_validation_loss => Float64(best_validation_loss),
        :seed => Int(seed),
        :vocabulary => _checkpoint_vocabulary(tokenizer),
    )

    open(path, "w") do io
        serialize(io, payload)
    end

    return path
end

function load_checkpoint(path::AbstractString)
    payload = open(path, "r") do io
        deserialize(io)
    end

    get(payload, :model_type, nothing) == "neural_bigram" ||
        throw(ArgumentError("checkpoint is not a neural_bigram checkpoint"))
    get(payload, :orientation, nothing) == "logits_table[next, current]" ||
        throw(ArgumentError("checkpoint has an unsupported logits_table orientation"))

    model = NeuralBigramModel(payload[:logits_table])

    return (
        model = model,
        version = payload[:version],
        vocab_size = payload[:vocab_size],
        step = payload[:step],
        optimizer_state = payload[:optimizer_state],
        config = payload[:config],
        best_validation_loss = payload[:best_validation_loss],
        seed = payload[:seed],
        vocabulary = payload[:vocabulary],
        orientation = payload[:orientation],
    )
end
