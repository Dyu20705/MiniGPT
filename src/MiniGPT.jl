module MiniGPT

include("tokenize.jl")
include("data.jl")
include("batch.jl")
include("count_models.jl")
include("objectives.jl")
include("neural_bigram.jl")
include("training.jl")
include("checkpoint.jl")

export CharacterTokenizer, TokenSplits
export decode, encode, get_batch, hello, next_token_example, split_token_ids, vocabulary
export AbstractCountLanguageModel
export LMReport, UnigramCountModel, UnigramCountLanguageModel, BigramCountModel
export fit_unigram, fit_bigram, evaluate, generate_ids, most_common_bigrams
export NeuralBigramModel, NeuralLMReport, NeuralBigramTrainingState
export cross_entropy, gradient_norm, logits, logsumexp, mean_cross_entropy
export neural_bigram_gradient, parameter_count, perplexity_from_loss, sgd_step!, train!, train_step!
export load_checkpoint, save_checkpoint, vocab_size

hello() = "MiniGPT is working"

end
