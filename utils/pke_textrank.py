import math
import logging
import networkx as nx

from utils.pke_base import LoadFile

class TextRank(LoadFile):
    """TextRank for keyword extraction.

    This model builds a graph that represents the text. A graph based ranking
    algorithm is then applied to extract the lexical units (here the words) that
    are most important in the text.

    In this implementation, nodes are words of certain part-of-speech (nouns
    and adjectives) and edges represent co-occurrence relation, controlled by
    the distance between word occurrences (here a window of 2 words). Nodes
    are ranked by the TextRank graph-based ranking algorithm in its unweighted
    variant.
    """

    def __init__(self):
        """Redefining initializer for TextRank."""

        super(TextRank, self).__init__()

        self.graph = nx.Graph()
        """The word graph."""

    def candidate_selection(self, pos=None):
        """Candidate selection using longest sequences of PoS.

        Args:
            pos (set): set of valid POS tags, defaults to ('NOUN', 'PROPN',
                'ADJ').
        """

        if pos is None:
            pos = {'NOUN', 'PROPN', 'ADJ'}

        # select sequence of adjectives and nouns
        self.longest_pos_sequence_selection(valid_pos=pos)

    def build_word_graph(self, window=2, pos=None):
        """Build a graph representation of the document in which nodes/vertices
        are words and edges represent co-occurrence relation. Syntactic filters
        can be applied to select only words of certain Part-of-Speech.
        Co-occurrence relations can be controlled using the distance between
        word occurrences in the document.

        As the original paper does not give precise details on how the word
        graph is constructed, we make the following assumptions from the example
        given in Figure 2: 1) sentence boundaries **are not** taken into account
        and, 2) stopwords and punctuation marks **are** considered as words when
        computing the window.

        Args:
            window (int): the window for connecting two words in the graph,
                defaults to 2.
            pos (set): the set of valid pos for words to be considered as nodes
                in the graph, defaults to ('NOUN', 'PROPN', 'ADJ').
        """

        if pos is None:
            pos = {'NOUN', 'PROPN', 'ADJ'}

        # flatten document as a sequence of (word, pass_syntactic_filter) tuples
        text = [(word, sentence.pos[i] in pos) for sentence in self.sentences
                for i, word in enumerate(sentence.stems)]

        # add nodes to the graph
        self.graph.add_nodes_from([word for word, valid in text if valid])

        # add edges to the graph
        for i, (node1, is_in_graph1) in enumerate(text):

            # speed up things
            if not is_in_graph1:
                continue

            for j in range(i + 1, min(i + window, len(text))):
                node2, is_in_graph2 = text[j]
                if is_in_graph2 and node1 != node2:
                    self.graph.add_edge(node1, node2)


    def candidate_weighting_ia(self, window=3, pos=None, top_percent=None, normalized=False, top_n=3):
        if pos is None:
            pos = {'NN', 'VP', 'NP', 'FW'} # modification
        self.build_word_graph(window=window, pos=pos)
        w = nx.pagerank(self.graph, alpha=0.85, tol=0.0001, weight=None)
        top_words = {}

        if top_percent is not None:
            logging.warning("Candidates are generated using {}-top".format(top_percent))
        #top_words = sorted(w, key=w.get, reverse=True)
        top_words = sorted(w.items(), key=lambda x: x[1], reverse=True)[:top_n]

        return top_words
    

'''
FAQ : 
- How to extract top_words variable in this function to the main code?
    - via self.instances?
    - via return?
- hasil top_words berupa list, tidak ada scorenya. 
    - perlu lihat debugging isi dari variable w
    - 
'''