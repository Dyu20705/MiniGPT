module MiniGPT

include("tokenize.jl")
include("data.jl")
include("batch.jl")
include("count_models.jl")

export CharacterTokenizer, TokenSplits
export decode, encode, get_batch, hello, next_token_example, split_token_ids, vocabulary
export AbstractCountLanguageModel
export LMReport, UnigramCountModel, UnigramCountLanguageModel, BigramCountModel
export fit_unigram, fit_bigram, evaluate, generate_ids, most_common_bigrams

hello() = "MiniGPT is working"

end
