#!/usr/bin/env python3
"""
Created on Sat Nov 17 02:02:18 2018

@author: katieliu
"""

from preprocess import PreProcess
from eval import Evaluation
from feature import Feature

from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import StandardScaler

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np

class gbt:

    def __init__(self, pre, paramfile):
        self.RAND_SEED = 17
        self.pre = pre
        self.dataset = pre.dataset
        self.corpus = pre.corpus
        self.labels = pre.labels
        self.DICT_LABEL2INT = pre.DICT_LABEL2INT
        self.subjects = pre.subjects
        if paramfile:
            with open(paramfile) as f:
                s = f.readlines()[0]
            import json
            self.params = json.loads(s)
        else:
            self.params = {}

    def dataset_gen(self, subject, valid=False):
        self.X_train, self.X_test, self.X_dev, \
        self.y_train, self.y_test, self.y_dev = self.pre.dataset_gen(subject, valid)

    def train(self, lda=False):
        """
        trains a xgboost GradientBoostingClassifier on each subject.
        """
        feature = Feature(trained=True)

        param_fixed = {
            'objective': 'binary:logistic',
            'silent': 1,
            'seed': self.RAND_SEED
        }
        
        base = {
                'learning_rate':0.1,
                'n_estimators':500,
                 "max_depth":5,
                 "min_child_weight":1,
                 "gamma":0,
                 "subsample":0.8,
                 "colsample_bytree":0.8,
                 "scale_pos_weight":1}
        
        true_labels = []
        predicted_labels = []


        for subj in self.subjects:
            # preprocess training and testing set
            self.dataset_gen(subject=subj, valid=False)

            # train and predict
            
            if subj in self.params.keys():
                param_fixed.update(self.params[subj])
            else:
                param_fixed.update(base)
            classifier = XGBClassifier(**param_fixed)
            
            if lda:
                model = Pipeline([('vectorized', feature.vector),
                                  ('tf-idf', feature.tfidftransform),
                                  ('lda', feature.ldatransform),
                                  #('scalar', StandardScaler(with_mean = False)),
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
        
        tune = "base_countv"
        
        np.savetxt("./gbt_data/gbt_"+tune+"_true.csv", true_matrix, delimiter=",")
        np.savetxt("./gbt_data/gbt_"+tune+"_pred.csv", pred_matrix, delimiter=",")
        
        
        evaluation = Evaluation(self.subjects)
        evaluation.model_evaluate(true_matrix=true_matrix, pred_matrix=pred_matrix)



if __name__ == '__main__':
    preprocessor = PreProcess(root='./corpus/corpus.json', save='./corpus/corpus_nostopwords.json')
    #g = gbt(preprocessor, "gbt_old_param.json")
    g = gbt(preprocessor, "")
    g.train(lda=False)
