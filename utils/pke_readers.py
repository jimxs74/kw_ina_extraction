import logging

from utils.pke_data_structures import Sentence

class Reader(object):
    """Reader default class."""

    def read(self, path):
        raise NotImplementedError

class PreprocessedReader(Reader):
    """Reader for preprocessed text."""

    def read(self, list_of_sentence_tuples):
        sentences = []
        for sentence_id, sentence in enumerate(list_of_sentence_tuples):
            words = [word for word, pos_tag in sentence]
            pos_tags = [pos_tag for word, pos_tag in sentence]
            shift = 0
            sentences.append(Sentence(
                words=words,
                pos=pos_tags
            ))
            shift += len(' '.join(words))
        return sentences
