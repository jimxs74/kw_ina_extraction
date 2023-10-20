#1. rutin1 import module
import pandas as pd
import os
import sys
import warnings
import matplotlib.pyplot as plt
#from sklearn.metrics.pairwise import cosine_similarity

warnings.simplefilter(action='ignore', category=UserWarning)

#2. rutin2 membuat syspath ke root utk aktifkan __init__.py
repo_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(repo_root)

#3. rutin3 Load the dataset
#dataset_path = os.path.join(repo_root, "notebooks/postager_nlp-id/dataset_ekstraksi_r29_pos_sm.xlsx")
dataset_path = os.path.join(repo_root, "data/dataset_ekstraksi_r29_lg.xlsx")
df = pd.read_excel(dataset_path)
df["text"] = df["judul"] +". "+ df["isi"]
#df_pos = df['pos_sentence_list']

# Preprocess
import re
def preprocess(text):
    text = re.sub(r"\b[a-zA-Z]\b", "", text)
    text = text.replace('.', '. ')
    text = re.sub('[^a-zA-Z.]', ' ', text)
    text = text.lower()
    text = re.sub("(\\d|\\W)+"," ",text)
    text = text.strip()

    return text

df["text"] = df['text'].apply(preprocess)
df["judul"] = df["judul"].apply(preprocess)

import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx

from nltk.util import ngrams

def generate_ngrams(words, n=2):
    """Generate ngrams from a list of words."""
    return [" ".join(gram) for gram in ngrams(words, n)]
'''
def get_phrase_embedding(phrase, w2v_model):
    """Get the averaged word embedding for a phrase."""
    words = phrase.split()
    embeddings = [w2v_model.wv[word] for word in words if word in w2v_model.wv.key_to_index]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return None
'''

def get_weighted_phrase_embedding(phrase, w2v_model, text_corpus):
    """
    Generate a TF-IDF weighted averaged word embedding for a given phrase.
    """
    # Generate TF-IDF dictionary
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text_corpus)
    tfidf_dict = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
    
    # Get phrase embedding
    words = phrase.split()
    embeddings = []
    weights = []
    
    for word in words:
        if word in w2v_model.wv.key_to_index:
            embedding = w2v_model.wv[word]
            tfidf_weight = tfidf_dict.get(word, 1.0)  # Default weight is 1 if word not in tfidf_dict
            embeddings.append(embedding)
            weights.append(tfidf_weight)
    
    if embeddings:
        embeddings = np.array(embeddings)
        weights = np.array(weights)
        weighted_average_embedding = np.average(embeddings, axis=0, weights=weights)
        return weighted_average_embedding
    else:
        return None


from collections import Counter
from nlp_id_local.tokenizer import PhraseTokenizer 
from nlp_id_local.postag import PosTag


model_path = os.path.join(repo_root, "notebooks/nlp-id_retraining/train_tuned.pkl") #add_8

def detect_bigram(text, available_tokens,):
    
    tokenizer = PhraseTokenizer()
    phrases = tokenizer.tokenize(text)
    # Include only bigrams whose individual words are in available_tokens
    bigrams_only = [phrase for phrase in phrases if phrase.count(" ") == 1 and all(word in available_tokens for word in phrase.split())]

    return bigrams_only

def detect_trigram(text, available_tokens):

    tokenizer = PhraseTokenizer()
    phrases = tokenizer.tokenize(text)
    # Include only trigrams whose individual words are in available_tokens
    trigrams_only = [phrase for phrase in phrases if phrase.count(" ") == 2 and all(word in available_tokens for word in phrase.split())]

    return trigrams_only

def get_unique_tokens_pos(all_tokens, model_path):
    """
    Get unique POS tags for tokens.
    """
    postagger = PosTag(model_path)
    pos_tokens = []
    seen_tokens = set()
    
    for token in all_tokens:
        if token not in seen_tokens:
            seen_tokens.add(token)
            tokens_pos = postagger.get_phrase_tag(token)
            pos_tokens.append(tokens_pos)
    return pos_tokens


def flatten_list_of_lists(list_of_lists):
    """
    Flatten a list of lists into a single list.
    """
    return [item for sublist in list_of_lists for item in sublist]


def filter_tokens_by_pos(flat_tokens, pos_filters):
    """
    Filter tokens based on their POS tags and ensure they're unique.
    """
    seen_tokens = set()
    return [token[0] for token in flat_tokens if token[1] in pos_filters and not (token[0] in seen_tokens or seen_tokens.add(token[0]))]

# Function to determine if a token is a unigram, bigram, or trigram
def get_ngram_type(token):
    return len(token.split())

def extract_keyphrases_with_ngrams_graph(text, w2v_model, judul, available_tokens, n=3):
    # Read stopwords from the file
    #stopwords_path = os.path.join(repo_root, "data/all_stop_words.txt") 
    stopwords_path = os.path.join(repo_root, "notebooks/stopwords_tuning/all_stop_words.txt")
    with open(stopwords_path, 'r') as file:
        stopwords = set(file.read().strip().splitlines())

    # Tokenize the text into unigrams
    #unigrams = [word for word in text.split() if word not in stopwords]

    # Tokenize the text into unigrams that are in available_tokens
    unigrams = [word for word in text.split() if word not in stopwords and word in available_tokens]

    # Generate bigrams and trigrams using nlp-id
    bigrams = detect_bigram(text, available_tokens)
    trigrams = detect_trigram(text, available_tokens)
    
    # Combine unigrams, filtered bigrams, and filtered trigrams
    all_tokens = unigrams + bigrams + trigrams

    # Filter tokens only for selected POS
    pos_tokens = get_unique_tokens_pos(all_tokens, model_path)
    flat_pos_tokens = flatten_list_of_lists(pos_tokens)
    selected_pos = {'NN', 'NNP', 'VB', 'NP', 'VP'} # FW di exclude
    filtered_tokens = filter_tokens_by_pos(flat_pos_tokens, selected_pos)

    # Get embeddings for each token (averaging word embeddings for bigrams/trigrams)
    token_embeddings = [get_weighted_phrase_embedding(token, w2v_model, text) for token in filtered_tokens]
    
    # Filter out tokens that don't have embeddings
    tokens, embeddings = zip(*[(token, emb) for token, emb in zip(filtered_tokens, token_embeddings) if emb is not None])
    # todo : masih ada token bahasa asing atau token aneh yg lolos. 

    # Compute the cosine similarity between token embeddings
    cosine_matrix = cosine_similarity(embeddings)
    
    # Create a graph and connect tokens with high similarity
    G = nx.Graph()
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            if cosine_matrix[i][j] > 0.5:  # This threshold can be adjusted
                G.add_edge(tokens[i], tokens[j], weight=cosine_matrix[i][j])
    
    # Create labels dictionary using the tokens
    labels = {token: token for token in tokens}

    # Compute the PageRank scores to rank the tokens
    scores = nx.pagerank(G)

    # Modify scores based on n-gram type
    for token in scores:
        ngram_type = get_ngram_type(token)
        if ngram_type == 1:  # Unigram
            pass  # No change to score
        elif ngram_type == 2:  # Bigram
            scores[token] *= 2  # Double the score
        elif ngram_type == 3:  # Trigram
            scores[token] *= 2  # Double the score

    # Modify scores if token is in title letter
    for token in scores:
        if any(token in title for title in judul):
            scores[token] *= 2

    # Extract top N keyphrases along with their scores
    ranked_tokens = sorted(((scores[token], token) for token in tokens if token in scores), reverse=True)
    
    keyphrases_with_scores = []
    seen_tokens = set()  # Set to keep track of tokens that have already been added

    for score, token in ranked_tokens:
        if token not in seen_tokens:
            keyphrases_with_scores.append((token, score))
            seen_tokens.add(token)  # Mark the token as seen
            if len(keyphrases_with_scores) >= n:
                break  # Stop when the desired number of keyphrases is reached

    return keyphrases_with_scores, G, labels


def visualize_graph(G, labels):

    # Remove self-loops (edges that connect a node to itself)
    G.remove_edges_from(nx.selfloop_edges(G))

    fig = plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G)
    nx.draw(G, pos=pos, with_labels=False, font_weight="bold")
    nx.draw_networkx_labels(G, pos, labels)
    plt.show()

w2v_path = os.path.join(repo_root, "models/w2v_200/idwiki_word2vec_200_new_lower.model")

w2v_model = Word2Vec.load(w2v_path)

# Get available tokens from the Word2Vec model
available_tokens = set(w2v_model.wv.key_to_index)

predict_textrank = pd.DataFrame()
for i in df.index:
    print('Processing index', i, end='...! ')
    text = df["text"][i] # sblm di preprocess
    #text = df_tr[i] # setelah di preprocess
    ls_judul = preprocess(df["judul"][i]).split()
    keyphrases,_,_ = extract_keyphrases_with_ngrams_graph(text, w2v_model, ls_judul, available_tokens, 3)
    df_keyphrases = pd.DataFrame(keyphrases, columns=['Keyword', 'Score'])
    a = pd.DataFrame(df_keyphrases.Keyword).T.reset_index(drop=True)
    b = pd.DataFrame(df_keyphrases.Score).round(3).T.reset_index(drop=True)
    df_keyphrases = pd.concat([a, b], axis=1)

    # Check if there are missing columns and add them with zero values
    missing_columns = 6 - df_keyphrases.shape[1]
    for _ in range(missing_columns):
        df_keyphrases[df_keyphrases.shape[1]] = 0

    df_keyphrases.columns = ['key_1', 'key_2','key_3','score_1', 'score_2','score_3']
    predict_textrank = pd.concat([predict_textrank, df_keyphrases], ignore_index=True)
    print('Done')
predict_textrank.head(3)

# EVALUATION

from utils import eval

targets = df[["k1", "k2", "k3","k4", "k5", "k6","k7"]].values.tolist()
df_targets = pd.DataFrame(targets)
