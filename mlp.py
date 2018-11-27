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

from sklearn.preprocessing import StandardScaler

class MLP:
    def __init__(self, pre):
        self.RAND_SEED = random.randint(0, 100000)
        self.pre = pre
        self.dataset = pre.dataset
        self.corpus = pre.corpus
        self.labels = pre.labels

        # constant variables for the whole collection
        self.DICT_LABEL2INT = pre.DICT_LABEL2INT
        self.subjects = pre.subjects

    def dataset_gen(self, subject, valid=False):
        self.X_train, self.X_test, self.X_dev, \
        self.y_train, self.y_test, self.y_dev = self.pre.dataset_gen(subject, valid)

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
        for i in range(len(self.subjects)):
            # preprocess training and testing set
            self.dataset_gen(subject=self.subjects[i], valid=False)

            # train and predict
            model.fit(self.X_train, self.y_train)

            # store true labels and predictions
            true.append(self.y_test)
            predicted.append(model.predict(self.X_test))

        # convert them to sparse matrix (N * L)
        # convert them to sparse matrix (N * L)
        # matrix[i][j] = 1 indicates entry i has label j,
        true_matrix, pred_matrix = np.array(true, int).T, np.array(predicted, int).T
        true_matrix[true_matrix == -1] = 0
        pred_matrix[pred_matrix == -1] = 0

        evaluation = Evaluation(self.subjects)
        evaluation.model_evaluate(true_matrix=true_matrix, pred_matrix=pred_matrix, model_name='MLP')


if __name__ == '__main__':
    preprocessor = PreProcess(root='./corpus/corpus.json', save='./corpus/corpus_nostopwords.json')
    model = MLP(preprocessor)
    model.train(lda=False)