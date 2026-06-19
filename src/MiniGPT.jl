module MiniGPT

include("tokenize.jl")

export CharacterTokenizer, decode, encode, hello, vocabulary

hello() = "MiniGPT is working"

end
