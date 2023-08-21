import pandas as pd
import numpy as np
import re
import math
from nltk.tokenize import word_tokenize 
from sklearn.feature_extraction.text import CountVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary

# tuning paramater
tuning_multiplier = 0.6  #aktor pengali dari score jika kata tersebut merupakan frase. default = 1
tuning_f_phrase = 3  #score minimum utk bisa disebut frase
m_prediksi = 3  #jumlah top -n keyword prediksi
n_top_phrase = 3   #jumlah frase yg akan di cari dalam fungsi get_top_phrase

def build_graph(vocab_len, processed_text, vocabulary):
    """
    Builds a weighted edge graph based on co-occurrences of words in the text.
    + perlu ada tambahan formula untuk menghitung score kata yg ada dalam title menjadi lebih besar. (1, 1.5, 2)
    """
    weighted_edge = np.zeros((vocab_len, vocab_len), dtype=np.float32)
    score = np.ones((vocab_len), dtype=np.float32)
    window_size = 3  
    covered_coocurrences = []

    for i in range(vocab_len):
        for j in range(vocab_len):
            if j == i:
                weighted_edge[i][j] = 0
            else:
                for window_start in range(len(processed_text) - window_size):
                    window_end = window_start + window_size
                    window = processed_text[window_start:window_end]
                    if (vocabulary[i] in window) and (vocabulary[j] in window):
                        index_of_i = window_start + window.index(vocabulary[i])
                        index_of_j = window_start + window.index(vocabulary[j])
                        if [index_of_i,index_of_j] not in covered_coocurrences:
                            weighted_edge[i][j] += 1 / math.fabs(index_of_i - index_of_j)
                            covered_coocurrences.append([index_of_i, index_of_j])

    inout = np.sum(weighted_edge, axis=1)
  
    MAX_ITERATIONS = 50
    d = 0.85
    threshold = 0.0001
    for _ in range(MAX_ITERATIONS):
        prev_score = np.copy(score)
        for i in range(vocab_len):
            summation = 0
            for j in range(vocab_len):
                if weighted_edge[i][j] != 0:
                    summation += (weighted_edge[i][j] / inout[j]) * score[j]
            score[i] = (1 - d) + d * summation
        if np.sum(np.fabs(prev_score - score)) <= threshold:
            break

    return vocabulary, score

def score_phrases(unique_phrases, vocabulary, score, multiplier=tuning_multiplier):
    """
    Computes the score of each phrase using the given vocabulary, word scores, and multiplier.
    """
    phrase_scores = []
    keywords = []
    for phrase in unique_phrases:
        phrase_score = 0
        keyword = ''
        for word in phrase:
            keyword += str(word) + " "
            phrase_score += score[vocabulary.index(word)]
        phrase_score *= multiplier
        phrase_scores.append(phrase_score)
        keywords.append(keyword.strip())

    return keywords, phrase_scores


def get_top_phrase(corpus, n=n_top_phrase):  #perlu ada improvement karena phrase yg di hasilkan masih blm proper
    vec1 = CountVectorizer(ngram_range=(2,3),  
            max_features=2000).fit([corpus])
    bag_of_words = vec1.transform([corpus])
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    # perlu di buat filter jika pola tidak mengikuti kaidah kata majemuk indonesia di excludekan.
    return words_freq[:n]

def predict_keywords(text, m=5, f_phrase=5, tuning_multiplier=1):
    """
    Predicts the top m keywords and top f_phrase phrases for the given text.
    processed_text = text keseluruhan
    vocabulary = unique word dalam proccesesed_text
    """
    processed_text = word_tokenize(text)
    vocabulary = list(set(processed_text))
    vocab_len = len(vocabulary)
    vocabulary, score = build_graph(vocab_len, processed_text, vocabulary)
    unigram = pd.DataFrame({
        'Keyword': vocabulary,
        'Score': score
    }).nlargest(m, 'Score')
    
    bi_trigram = pd.DataFrame(get_top_phrase(text, n=50), columns=['Phrase', 'Score'])
    bi_trigram = bi_trigram[bi_trigram['Score'] >= f_phrase]
    bi_trigram['Tokens'] = bi_trigram['Phrase'].apply(word_tokenize)
    unique_phrases = bi_trigram['Tokens'].values.tolist()
    keywords, phrase_scores = score_phrases(unique_phrases, vocabulary, score, tuning_multiplier) #BUG_1 not accesed by pylance, krn tidak di gunakan di procss selanjutnya
    # memasukan score ke dalam dataframe
    bi_trigram = pd.DataFrame({
        'Phrase': keywords,
        'Score': phrase_scores
    }).nlargest(m, 'Score')

      # Combine unigram and bi_trigram dataframes
    predict_keywords = pd.concat([unigram, bi_trigram[['Phrase', 'Score']].rename(columns={'Phrase': 'Keyword'})])\
                    .sort_values('Score', ascending=False)\
                    .nlargest(m, 'Score')\
                    .reset_index(drop=True)

    return predict_keywords