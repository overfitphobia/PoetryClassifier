#!/usr/bin/env python3
"""
Created on Sat Nov 17 02:02:18 2018

@author: katieliu
"""

from preprocess import PreProcess
from eval import Evaluation
from feature import Feature

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

import numpy as np


class gbt:
    def __init__(self, pre):
        self.RAND_SEED = 16383
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
        """
        trains a sklearn GradientBoostingClassifier on each subject.
        """
        feature = Feature(trained=True)
# =============================================================================
#         classifier = SGDClassifier(loss='hinge', penalty='l2',
#                                    max_iter=1000, shuffle=True, validation_fraction=0.1)
# =============================================================================
        
        """
        TODO:
            1) tune sklearn gbt classifier
            2) implement xgboost xgbclassifier
        """
        
        # employs early stopping 
        classifier = GradientBoostingClassifier(n_estimators=500, 
                                                validation_fraction=0.1,
                                                n_iter_no_change=5, tol=0.01,
                                                random_state=self.RAND_SEED)

        if lda:
            model = Pipeline([('vectorized', feature.vector),
                              ('tf-idf', feature.tfidftransform),
                              ('lda', feature.ldatransform),
                              ('clf', classifier)])
        else:
            model = Pipeline([('vectorized', feature.vector),
                              ('tf-idf', feature.tfidftransform),
                              ('clf', classifier)])

        true, predicted = [], []
        for subj in self.subjects:
            # preprocess training and testing set
            self.dataset_gen(subject=subj, valid=False)

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
    preprocessor = PreProcess(root='./corpus/corpus.json', save='./corpus/corpus_nostopwords.json')
    model = gbt(preprocessor)
    model.train(lda=False)