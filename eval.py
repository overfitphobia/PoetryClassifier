from sklearn import metrics


class Evaluation:
    def __init__(self, subjs):
        self.subjs = subjs

    def model_evaluate(self, true_matrix, pred_matrix):
        assert true_matrix.shape == pred_matrix.shape

        print("Model evaluation:")
        print("Hamming Loss: {}".format(metrics.hamming_loss(true_matrix, pred_matrix)))
        print("Classification Report:")
        import warnings
        warnings.filterwarnings('ignore')
        print(metrics.classification_report(true_matrix, pred_matrix, target_names=self.subjs))


if __name__ == '__main__':
    import numpy as np
    a = np.array([[0, 1, 1, 1], [0, 0, 0, 1], [1, 0, 1, 0]])
    b = np.array([[0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1]])
    print(metrics.classification_report(a, b))
