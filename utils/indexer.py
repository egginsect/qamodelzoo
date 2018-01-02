from collections import OrderedDict
class Indexer(object):
    def __init__(self, vocabfile, allow_vocab_growth=False):
        """
        Load Vocabulary with 0 reserved for padding and 1 reserved for unknown words
        If allow_vocab_growth then unknown words will be append at the end of the loaded vocabulary
        """
        self.vocab = OrderedDict([('PAD',0), ('UNK', 1)])
        self.allow_vocab_growth = allow_vocab_growth
        with open(vocabfile) as f:
            for line in f:
                self.vocab[line.split()[0]] = len(self.vocab)
        self.idx2word = OrderedDict(zip(self.vocab.values(), self.vocab.keys()))

    def index_word(self, word):
        """Index word to index given dictionary"""
        if self.allow_vocab_growth:
            self.vocab[word] = self.vocab.get(word, len(self.vocab))
            return self.vocab[word]
        else:
            return self.vocab.get(word, 1)

    def index_sentence(self, words):
        """Index word list"""
        return [self.index_word(word) for word in words]

    def decode_sentence(self, idxs):
        """Convert index to string"""
        return ' '.join([self.idx2word[idx] for idx in idxs])
