module StressDetection
using Word2Vec
using CSV
using DataFrames
using Statistics
using LinearAlgebra
# Write your package code here.


function map_tokens(tokens::Vector)
    token2index = Dict()
    index2token = []
    index = 1

    for token in tokens
        token = string(token)
        #println(token)
        if haskey(token2index, token)
            continue
        else
            token2index[token] = index
            append!(index2token, [token])
            index += 1
        end
    end
    return token2index, index2token
end

function tokenize_sentence(sentence::String)
    return split(sentence)
end

function tokenize_text(text::String)
    text = lowercase(text)
    text = replace(text, "?" => " ? ", "!" => " ! ", "..." => ".", "\"" => " ", "/" => " ", "[" => " ", "]" => "", "," => "")
    sentences = collect(eachsplit(text, ". "))

    tokens = []
    for sentence in sentences
        sentence_tokens = tokenize_sentence( string(sentence) )
        append!(tokens, sentence_tokens)
    end
    return tokens
end

function tokens_to_indexes(tokens, token2index)
    indexes = get.([token2index],  tokens, 1)
    return indexes
end

function indexes_to_embeddings(indexes; embed_width=5)
    n_tokens = size(tokens)[1]

    embeddings = []
    for i in (embed_width+1):(n_tokens-embed_width)
        #left_embed = indexes[ (i-embed_width) : i-1 ]
        #right_embed = indexes[ (i+1) : (i+embed_width) ]
        #embedding = vcat(left_embed, right_embed)
        embedding = indexes[ (i-embed_width) : (i+embed_width) ]
        embeddings = vcat(embeddings, [embedding])
    end
    return embeddings
end

function word_to_word_matrix(embeddings, index2token)
    n_tokens = size(index2token)[1]
    n_letters = size(embeddings[1])[1]
    mid_index = Int( (size(embeddings[1])[1]+1)/2 )

    matrix = ones(n_tokens, n_tokens)
    for embed in embeddings
        center_word = embed[mid_index]
        for idx in 1:(mid_index-1)
            word = embed[idx]
            matrix[center_word, word] += 1
        end
        for idx in (mid_index+1):n_letters
            word = embed[idx]
            matrix[center_word, word] += 1
        end
    end
    return matrix
end

function find_center(indexes, word2word)
    n_tokens = size(word2word)[1]
    n_indexes = size(indexes)[1]
    center = zeros(n_tokens)

    for index in indexes
        vector = word2word[index, :]
        center += vector
    end
    return center ./ n_indexes
end

function main()
    train_table = CSV.read("train.csv", DataFrame; header = true)  
    
    train_X_raw = train_table[!,"text"]
    train_Y = train_table[!,"label"]
    
    test_table = CSV.read("test.csv", DataFrame; header = true)
    test_X_raw = test_table[!,"text"]
    

    # train matrix
    text = string(train_X_raw) * string(test_X_raw)
    tokens = tokenize_text(text)
    token2index, index2token = map_tokens(tokens)
    indexes = tokens_to_indexes(tokens, token2index)
    embeddings = indexes_to_embeddings(indexes)
    word2word = word_to_word_matrix(embeddings, index2token)

    # get categories' means
    train_stressed = train_X_raw[train_Y.==1]
    train_chill = train_X_raw[train_Y.==0]

    string_stressed = string(train_stressed)
    string_chill = string(train_chill)

    tokens_stressed = tokenize_text(string_stressed)
    tokens_chill = tokenize_text(string_chill)

    indexes_stressed = tokens_to_indexes(tokens_stressed, token2index)
    indexes_chill = tokens_to_indexes(tokens_chill, token2index)

    mean_stressed = find_center(indexes_stressed, word2word)
    mean_chill = find_center(indexes_chill, word2word)

    # get training accuracy
    Y_hat = []
    for post in train_X_raw
        post_tokens = tokenize_text(post)
        post_indexes = tokens_to_indexes(post_tokens, token2index)
        post_mean = find_center(post_indexes, word2word)
        
        dist_stressed = norm(post_mean - mean_stressed)
        dist_chill = norm(post_mean - mean_chill)

        if dist_stressed < dist_chill
            Y_hat = vcat(Y_hat, 1)
        else
            Y_hat = vcat(Y_hat, 0)
        end
    end
    train_accuracy = mean( train_Y .== Y_hat )
    println("train accuracy: ", train_accuracy)

end