#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from preprocess import PreProcess
from eval import Evaluation
from feature import Feature

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler


import os
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class logreg:

    def __init__(self, pre):
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
        feature = Feature(trained=True)
        
        
        classifier = LogisticRegression(
                penalty = 'l2',
                max_iter = 100,
                solver = 'liblinear',
                random_state = self.RAND_SEED)

        
        true_labels = []
        predicted_labels = []


        for subj in self.subjects:
            # preprocess training and testing set
            self.dataset_gen(subject=subj, valid=False)

            # train and predict
            
            if lda:
                model = Pipeline([('vectorized', feature.vector),
                                  ('tf-idf', feature.tfidftransform),
                                  ('lda', feature.ldatransform),
                                  ('scalar', StandardScaler(with_mean = False)),
                                  ('clf', classifier)])
            else:
                model = Pipeline([('vectorized', feature.vector),
                                  ('tf-idf', feature.tfidftransform),
                                  #('scalar', StandardScaler(with_mean = False)),
                                  ('clf', classifier)])
            
            
            model.fit(self.X_train, self.y_train)

            predicted = model.predict(self.X_test)
            # hamming
            predicted_labels.append(predicted)
            true_labels.append(self.y_test)

            # Evaluate
            print("Evaluation report on the subject of " + str(subj))
            print("model score = " + str(model.score(self.X_test, self.y_test)))
            metric = Evaluation([subj,"not_"+subj])
            metric.model_evaluate(np.array(self.y_test), np.array(predicted))
            print("\n\n\n")
        true_matrix, pred_matrix = np.array(true_labels, int).T, np.array(predicted_labels, int).T
        true_matrix[true_matrix == -1] = 0
        pred_matrix[pred_matrix == -1] = 0
        
        fp = "lda_tfidf"
        
        np.savetxt("./logreg_data_eval/logreg_"+fp+"_true.csv", true_matrix, delimiter=",")
        np.savetxt("./logreg_data_eval/logreg_"+fp+"_pred.csv", pred_matrix, delimiter=",")
        
        
        evaluation = Evaluation(self.subjects)
        evaluation.model_evaluate(true_matrix=true_matrix, pred_matrix=pred_matrix)



        

if __name__ == '__main__':
    preprocessor = PreProcess(root='./corpus/corpus.json', save='./corpus/corpus_nostopwords.json')
    g = logreg(preprocessor)
    g.train(lda=True)
