from preprocess import PreProcess
from eval import Evaluation
from feature import Feature

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.neural_network import MLPClassifier
import random

from collections import Counter
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

    """
    TODO: Pipeline is a classical class which could extract, select features and train the model with some classifier
    Instruction for train_SVM:
        (1) vectorize the corpus with word counts
        (2) employ tf-idf transformation
        (3) fit and predict with SVM model (which is optimized by SGDOptimizer)
            Remeber to encode the label with +1/-1
    """
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
                              ('scalar', StandardScaler(with_mean=False)),
                              ('clf', classifier)])
        else:
            model = Pipeline([('vectorized', feature.vector),
                              ('tf-idf', feature.tfidftransform),
                              ('scalar', StandardScaler(with_mean=False)),
                              ('clf', classifier)])

        for subj in self.subjects:
            # preprocess training and testing set
            self.dataset_gen(subj, valid=False)

            # vx = feature.vector.fit_transform(self.X_train)
            # tfidf = feature.tfidftransform.fit_transform(vx)

            # print tfidf.shape

            # train and predict
            model.fit(self.X_train, self.y_train)
            predict = model.predict(self.X_test)
            # Evaluate
            print("Evaluation report on the subject of " + str(subj))
            print("model score = " + str(model.score(self.X_test, self.y_test)))
            metric = Evaluation(self.y_test, predict)
            metric.output()


if __name__ == '__main__':
    preprocessor = PreProcess(root='./corpus/corpus.json', save='./corpus/corpus_nostopwords.json')
    model = MLP(preprocessor)
    model.train(lda=False)
