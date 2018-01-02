from collections import OrderedDict
class Indexer(object):
    def __init__(self, vocabfile, allow_vocab_growth=False):
        self.vocab = OrderedDict([('PAD',0), ('UNK', 1)])
        self.allow_vocab_growth = allow_vocab_growth
        with open(vocabfile) as f:
            for line in f:
                self.vocab[line.split()[0]] = len(self.vocab)
        self.idx2word = OrderedDict(zip(self.vocab.values(), self.vocab.keys()))

    def index_word(self, word):
        if self.allow_vocab_growth:
            self.vocab[word] = self.vocab.get(word, len(self.vocab))
            return self.vocab[word]
        else:
            return self.vocab.get(word, 1)

    def index_sentence(self, words):
        return [self.index_word(word) for word in words]

    def decode_sentence(self, idxs):
        return ' '.join([self.idx2word[idx] for idx in idxs])
