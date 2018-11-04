from sklearn import metrics


class Evaluation:
    def __init__(self, gtruth, predicted):
        self.predicted = predicted
        self.gtruth = gtruth
        self.output()

    def output(self):
        self.report()

    def report(self):
        print("report on the whole classification")
        print(metrics.classification_report(self.gtruth, self.predicted))

        print("confusion matrix on the whole classification")
        print(metrics.confusion_matrix(self.gtruth, self.predicted))



