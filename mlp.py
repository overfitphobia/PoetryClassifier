from preprocess import PreProcess
from eval import Evaluation
from feature import Feature

import sklearn.feature_extraction.text as fet
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

from sklearn.neural_network import MLPClassifier
import random

from collections import Counter

from sklearn.preprocessing import StandardScaler

class MLP:
    def __init__(self, dataset):
        self.dataset = dataset
        self.RAND_SEED = random.randint(0, 100000)
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
            train_test_split(self.corpus, self.labels, test_size=rate, shuffle=shuffle)

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


    def train(self, lda=False):
        feature = Feature(trained=True)

        # classifier = SGDClassifier(loss='hinge', penalty='l2',
        #                            max_iter=1000, shuffle=True, validation_fraction=0.1)

        # do not warm start
        classifier = MLPClassifier(solver='lbfgs',
                            alpha=1e-5,
                            activation='logistic',
                            learning_rate='adaptive',
                            hidden_layer_sizes=(20, ),
                            random_state=self.RAND_SEED
                            )

        if lda:
            model = Pipeline([('vectorized', feature.vector),
                              ('tf-idf', feature.tfidftransform),
                              ('lda', feature.ldatransform),
                              ('scalar', StandardScaler(with_mean = False)),
                              ('clf', classifier)])
        else:
            model = Pipeline([('vectorized', feature.vector),
                              ('tf-idf', feature.tfidftransform),
                              ('scalar', StandardScaler(with_mean = False)),
                              ('clf', classifier)])

        true, predicted = [], []
        for subj in self.subjects:
            # preprocess training and testing set
            self.encoder_binary(_label=self.DICT_LABEL2INT[subj])
            self.split(rate=0.2, shuffle=True)

            # vx = feature.vector.fit_transform(self.X_train)
            # tfidf = feature.tfidftransform.fit_transform(vx)

            # print tfidf.shape

            # train and predict
            model.fit(self.X_train, self.y_train)

            # store true labels and predictions
            true.append(self.y_test)
            predicted.append(model.predict(self.X_test))

        # convert them to sparse matrix (N * L)
        # matrix[i][j] = 1 indicates entry i has label j,
        true_matrix, pred_matrix = np.array(true, int).T, np.array(predicted, int).T
        true_matrix[true_matrix == -1] = 0
        pred_matrix[pred_matrix == -1] = 0

        evaluation = Evaluation(self.subjects)
        evaluation.model_evaluate(true_matrix=true_matrix, pred_matrix=pred_matrix)


if __name__ == '__main__':
    preprocessor = PreProcess(root='./corpus/corpus.json', save='./corpus/unordered_corpus.json')
    model = MLP(preprocessor.dataset)
    model.train(lda=False)
