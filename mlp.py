from preprocess import PreProcess
from eval import Evaluation
from feature import Feature
from sklearn.pipeline import Pipeline
import numpy as np

from sklearn.neural_network import MLPClassifier
import random

from sklearn.preprocessing import StandardScaler


class MLP:
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

    def train(self):
        feature = Feature(trained=False)

        # classifier = SGDClassifier(loss='hinge', penalty='l2',
        #                            max_iter=1000, shuffle=True, validation_fraction=0.1)

        # do not warm start
        classifier = MLPClassifier(solver='lbfgs',
                                   alpha=1e-5,
                                   activation='logistic',
                                   learning_rate='adaptive',
                                   hidden_layer_sizes=(20,),
                                   random_state=self.RAND_SEED
                                   )

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
    model = MLP(preprocessor, istfidf=True, isnorm=True, islda='None', modelname='mlp_tfidf_norm_lda')
    model.train()
