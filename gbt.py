#!/usr/bin/env python3
"""
Created on Sat Nov 17 02:02:18 2018

@author: katieliu
"""

from preprocess import PreProcess
from eval import Evaluation
from feature import Feature

import sklearn.feature_extraction.text as fet
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import GradientBoostingClassifier


class gbt:

    def __init__(self, dataset):
        self.RAND_SEED = 16383
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

    def split(self, train_rate=0.7, valid_rate=0.0, shuffle=True):
        """
        1) train-test split 
        2) if valid_rate provided: test further split into validation-test.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.corpus, self.labels, test_size=(1 - train_rate), shuffle=shuffle)
        if valid_rate:
            self.X_test, self.X_valid, self.y_test, self.y_valid = \
                train_test_split(self.X_test, self.y_test, test_size=valid_rate, shuffle=shuffle)

    def encoder_binary(self, _label):
        self.labels = [1 if _label in entity['label'] else -1 for entity in self.dataset]

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

        for subj in self.subjects:
            # preprocess training and testing set
            self.encoder_binary(_label=self.DICT_LABEL2INT[subj])
            self.split(train_rate=0.8, shuffle=True)
            # train and predict
            model.fit(self.X_train, self.y_train)

            predicted = model.predict(self.X_test)

            # Evaluate
            print("Evaluation report on the subject of " + str(subj))
            print("model score = " + str(model.score(self.X_test, self.y_test)))
            metric = Evaluation(self.y_test, predicted)
            metric.output()
            print("\n\n\n")


if __name__ == '__main__':
    preprocessor = PreProcess(root='./corpus/corpus.json', save='./corpus/unordered_corpus.json')
    model = gbt(preprocessor.dataset)
    model.train(lda=False)
