from preprocess import PreProcess
from eval import Evaluation

import numpy as np
import sklearn.feature_extraction.text as fet
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation


class Model:
    def __init__(self, dataset):
        self.dataset = dataset
        self.DICT_LABEL2INT = {
            "LOVE": 0,
            "NATURE": 1,
            "SOCIAL COMMENTARIES": 2,
            "RELIGION": 3,
            "LIVING": 4,
            "RELATIONSHIPS": 5,
            "ACTIVITIES": 6,
            "ARTS & SCIENCES": 7,
            "MYTHOLOGY & FOLKLORE": 8
        }
        self.subjects = ["LOVE", "NATURE", "SOCIAL COMMENTARIES", "RELIGION",
                         "LIVING", "RELATIONSHIPS", "ACTIVITIES", "ARTS & SCIENCES", "MYTHOLOGY & FOLKLORE"]
        self.corpus = [entity['content'] for entity in self.dataset]
        self.labels = [entity['label'] for entity in self.dataset]

    def split(self, rate=0.2, shuffle=True):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.corpus, self.labels, test_size=rate, shuffle=True)

    """
    TODO: Pipeline is a classical class which could extract, select features and train the model with some classifier
    Instruction for train_SVM:
        (1) vectorize the corpus with word counts
        (2) employ tf-idf transformation
        (3) fit and predict with SVM model (which is optimized by SGDOptimizer)
            Remeber to encode the label with +1/-1
    """
    def encoder_binary(self, _label):
        self.labels = [1 if _label in entity['label'] else -1 for entity in self.dataset]

    def train_svm(self, lda=False):
        if lda:
            model_svm = Pipeline([('vectorized', fet.CountVectorizer()),
                                  ('tf-idf', fet.TfidfTransformer()),
                                  ('lda-model', LatentDirichletAllocation(n_components=100)),
                                  ('clf', SGDClassifier(loss='hinge',
                                                        penalty='l2',
                                                        max_iter=1000,
                                                        shuffle=True,
                                                        validation_fraction=0.1))
                                  ])
        else:
            model_svm = Pipeline([('vectorized', fet.CountVectorizer()),
                                  ('tf-idf', fet.TfidfTransformer()),
                                  ('clf', SGDClassifier(loss='hinge',
                                                        penalty='l2',
                                                        max_iter=1000,
                                                        shuffle=True,
                                                        validation_fraction=0.1))
                                  ])
        # Especially for multi-label problem:
        # Encode each subject as a binary classifier
        for subj in self.subjects:
            self.encoder_binary(_label=self.DICT_LABEL2INT[subj])
            self.split(rate=0.2, shuffle=True)

            model_svm.fit(self.X_train, self.y_train)
            predicted = model_svm.predict(self.X_test)

            print("Evaluation report on the subject of " + str(subj))
            metric = Evaluation(self.y_test, predicted)
            metric.output()


if __name__ == '__main__':
    preprocessor = PreProcess(root='./corpus/corpus.json', save='./corpus/altered_corpus.json')
    model = Model(preprocessor.dataset)
    model.train_svm(lda=False)