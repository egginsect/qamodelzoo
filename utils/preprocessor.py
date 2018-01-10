from collections import OrderedDict
import string
import re
class Preprocessor(object):
    def __init__(self, vocabfile, allow_vocab_growth=False):
        """
        Load Vocabulary with 0 reserved for padding and 1 reserved for unknown words
        If allow_vocab_growth then unknown words will be append at the end of the loaded vocabulary
        """
        preserved_word=[('PAD', 0), ('UNK', 1)]
        self.vocab = OrderedDict(preserved_word)
        self.allow_vocab_growth = allow_vocab_growth
        with open(vocabfile) as f:
            for line in f:
                self.vocab[line.split()[0]] = len(self.vocab)
        self.idx2word = OrderedDict(zip(self.vocab.values(), self.vocab.keys()))

    def preprocess(self, qapair):
        datum = {}
        question = Preprocessor.normalize_answer(qapair.question)
        passage = Preprocessor.normalize_answer(qapair.passage)
        datum['question'] = self.index_sentence(question.split())
        datum['passage'] = self.index_sentence(passage.split())
        answer = Preprocessor.normalize_answer(qapair.answer)
        char_ans_start = qapair.answer_start
        datum['answer_start'], datum['answer_end']= Preprocessor.get_answer_span(passage, answer, char_ans_start)
        return datum

    def index_word(self, word):
        """Index word to index given dictionary"""
        if self.allow_vocab_growth:
            try:
                return self.vocab[word]

            except KeyError:
                idx = len(self.vocab)
                self.vocab[word] = idx
                self.idx2word[idx] = word
                return idx
        else:
            return self.vocab.get(word, 1)

    def index_sentence(self, words):
        """Index word list"""
        return [self.index_word(word) for word in words]

    def decode_sentence(self, idxs):
        """Convert index to string"""
        return ' '.join([self.idx2word[idx] for idx in idxs])

    @staticmethod
    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @staticmethod
    def get_answer_span(passage, answer, char_ans_start):
        answer_start = len(passage[:char_ans_start].split())
        answer_end = len(passage[:char_ans_start+len(answer)].split())-1
        return answer_start, answer_end
