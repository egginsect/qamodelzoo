import numpy as np
import json
import ipdb
from collections import namedtuple

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

class QAPair(namedtuple('QAPair', 'question answer_start text id')):
    def __new__(cls, question, ansdict, id):

        return super(QAPair, cls).__new__(cls, question, ansdict['answer_start'], ansdict['text'], id)

class SQuADData(object):
    def __init__(self, data):
        self.title = data['title']
        self.data = data['paragraphs']
    def __iter__(self):
        for item in self.data:
            for qapair in item['qas']:
                for answer in qapair['answers']:
                    yield QAPair(qapair['question'], answer, qapair['id'])

class TrainingData(object):
    def __init__(self, infile):
        with open(infile) as f:
            self.data  = json.load(f)
        for item in self.data['data']:
            for qapair in SQuADData(item):
                ipdb.set_trace()
                pass

if __name__=="__main__":
    reader = TrainingData('../data/train-v1.1.json')
