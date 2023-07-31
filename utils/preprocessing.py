import re
import nltk

from nltk.tokenize import word_tokenize 
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary

def preprocess(text):
    '''
    fungsi untuk menghilangkan karakter yg tidak bermakna dan menghilangkan stopword.
    referensi stopword: tala + sastrawi + custom
    '''
    text = re.sub(r"\b[a-zA-Z]\b", "", text)
    text = text.replace('.', '. ')
    text = re.sub('[^a-zA-Z.]', ' ', text)
    text = text.lower()
    text = re.sub("(\\d|\\W)+"," ",text)
    text = text.strip()

    with open('./data/stopword_tala_sastrawi.txt', 'r') as f:
        stopword_tala_sastrawi = [line.strip() for line in f]
    
    with open('./result/custom_current_data_stop_word.txt', 'r') as f:
        stopword_custom = [line.strip() for line in f]
    stop_words = stopword_tala_sastrawi + stopword_custom

    dictionary = ArrayDictionary(stop_words)
    str = StopWordRemover(dictionary)
    text = str.remove(text) # 2x cleaning stop word
    text = str.remove(text)
    return text

def preprocess_tokenize(text):
    '''
    fungsi untuk memproses text yg sudah di process di preprocess() menjadi token
    dilakukan 3x preprocessing, karena stopword masih sering lewat kalau hanya 1x
    '''
    text = preprocess(text)
    text = preprocess(text)
    text = preprocess(text)
    tokens = text.split()
    tokens = [token for token in tokens if token]  # remove any empty tokens
    return tokens


def preprocess_corpus_1_surat(doc):
    '''
    fungsi untuk merubah text dalam 1 surat menjadi list sentence.
    hasil akhir sentence yg sudah di preprocessing dan dihilangkan stopwordnya
    '''
    corpus = []
    paragraphs = doc.split('\n\n') # assuming paragraphs are separated by two newline characters
    for paragraph in paragraphs:
        sentences = nltk.sent_tokenize(paragraph)
        corpus.extend(sentences)

    preprocessed_corpus = []
    for text in corpus:
        preprocessed_text = preprocess(text)
        preprocessed_corpus.append(preprocessed_text)
    return preprocessed_corpus

def preprocess_corpus(df):
    corpus = []
    for doc in df:
        paragraphs = doc.split('\n\n') # assuming paragraphs are separated by two newline characters
        for paragraph in paragraphs:
            sentences = nltk.sent_tokenize(paragraph)
            corpus.extend(sentences)

    preprocessed_corpus = []
    for text in corpus:
        preprocessed_text = preprocess(text)
        preprocessed_text = preprocess(preprocessed_text)
        preprocessed_corpus.append(preprocessed_text)
    return preprocessed_corpus