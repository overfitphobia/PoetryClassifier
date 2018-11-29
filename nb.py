from preprocess import PreProcess
from eval import Evaluation
from feature import Feature
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler


class NB:
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

    """
    TODO: Pipeline is a classical class which could extract, select features and train the model with some classifier
    Instruction for train_SVM:
        (1) vectorize the corpus with word counts
        (2) employ tf-idf transformation
        (3) fit and predict with SVM model (which is optimized by SGDOptimizer)
            Remeber to encode the label with +1/-1
    """

    def train(self):
        feature = Feature(trained=False)
        classifier = MultinomialNB()

        pipeline_steps = [('vectorized', feature.vector)]
        if self.istfidf:
            pipeline_steps.append(('tf-idf', feature.tfidftransform))
        if self.islda == 'small':
            pipeline_steps.append(('lda', feature.ldatransform_small))
            print('hahahahahha')
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
        # matrix[i][j] = 1 indicates entry i has label j,
        true_matrix, pred_matrix = np.array(true, int).T, np.array(predicted, int).T
        true_matrix[true_matrix == -1] = 0
        pred_matrix[pred_matrix == -1] = 0

        evaluation = Evaluation(self.subjects)
        evaluation.model_evaluate(true_matrix=true_matrix, pred_matrix=pred_matrix, model_name=self.modelname)


if __name__ == '__main__':
    preprocessor = PreProcess(root='./corpus/corpus.json', save='./corpus/corpus_nostopwords.json')
    # istfidf=False  isnorm=False  islda='null'  : [modelname]_null
    # istfidf=True   isnorm=False  islda='null'  : [modelname]_tfidf_null
    # istfidf=True   isnorm=True   islda='null'  : [modelname]_tfidf_norm_null
    # istfidf=True   isnorm=True   islda='small' : [modelname]_tfidf_norm_small
    # istfidf=True   isnorm=True   islda='large' : [modelname]_tfidf_norm_large
    model = NB(preprocessor, istfidf=False, isnorm=False, islda='null', modelname='NB')
    model.train()
