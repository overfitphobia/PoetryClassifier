#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from preprocess import PreProcess
from eval import Evaluation
from feature import Feature

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

import os
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class logreg:

    def __init__(self, pre, istfidf, isnorm, islda, modelname):
        self.istfidf = istfidf
        self.isnorm = isnorm
        self.islda = islda
        self.modelname = modelname
        if istfidf:
            self.modelname += '_tfidf'
        if isnorm:
            self.modelname += '_norm'
        self.modelname += islda

        self.RAND_SEED = 17
        self.pre = pre
        self.dataset = pre.dataset
        self.corpus = pre.corpus
        self.labels = pre.labels
        self.DICT_LABEL2INT = pre.DICT_LABEL2INT
        self.subjects = pre.subjects

    def dataset_gen(self, subject, valid=False):
        self.X_train, self.X_test, self.X_dev, \
        self.y_train, self.y_test, self.y_dev = self.pre.dataset_gen(subject, valid)

    def train(self, lda=False):
        """
        trains a xgboost GradientBoostingClassifier on each subject.
        """
        feature = Feature(trained=False)
        classifier = LogisticRegression(
            penalty='l2',
            max_iter=100,
            solver='liblinear',
            random_state=self.RAND_SEED)

        true_labels = []
        predicted_labels = []

        for subj in self.subjects:
            # preprocess training and testing set
            self.dataset_gen(subject=subj, valid=False)

            # train and predict
            pipeline_steps = [('vectorized', feature.vector)]
            if self.istfidf:
                pipeline_steps.append(('tf-idf', feature.tfidftransform))
            if self.islda == 'small':
                pipeline_steps.append(('lda', feature.ldatransform_small))
            elif self.islda == 'large':
                pipeline_steps.append(('lda', feature.ldatransform_large))
            else:
                pass
            if self.isnorm:
                pipeline_steps.append(('scalar', StandardScaler(with_mean=False)))
            pipeline_steps.append(('clf', classifier))
            model = Pipeline(steps=pipeline_steps)

            model.fit(self.X_train, self.y_train)

            predicted = model.predict(self.X_test)
            # hamming
            predicted_labels.append(predicted)
            true_labels.append(self.y_test)

            # Evaluate
            print("Evaluation report on the subject of " + str(subj))
            print("model score = " + str(model.score(self.X_test, self.y_test)))
            print(metrics.classification_report(self.y_test, predicted))
            print("\n\n\n")
        true_matrix, pred_matrix = np.array(true_labels, int).T, np.array(predicted_labels, int).T
        true_matrix[true_matrix == -1] = 0
        pred_matrix[pred_matrix == -1] = 0

        evaluation = Evaluation(self.subjects)
        evaluation.model_evaluate(true_matrix=true_matrix, pred_matrix=pred_matrix, model_name=self.modelname)


if __name__ == '__main__':
    preprocessor = PreProcess(root='./corpus/corpus.json', save='./corpus/corpus_nostopwords.json')
    g = logreg(preprocessor, istfidf=True, isnorm=True, islda='None', modelname='LogReg')
    g.train(lda=True)
