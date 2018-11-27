from sklearn import metrics
import h5py
import glob


class Evaluation:
    def __init__(self, subjs):
        self.subjs = subjs

    def model_evaluate(self, true_matrix, pred_matrix, model_name, save=True):
        assert true_matrix.shape == pred_matrix.shape

        eval_result = list()
        eval_result.append("Model evaluation:")
        eval_result.append("Hamming Loss: {}".format(metrics.hamming_loss(true_matrix, pred_matrix)))
        eval_result.append("Classification Report:")
        import warnings
        warnings.filterwarnings('ignore')
        eval_result.append(metrics.classification_report(true_matrix, pred_matrix, target_names=self.subjs))

        for line in eval_result:
            print(line)

        if save:
            with h5py.File('results/{}_matrix.h5'.format(model_name), 'w') as hf:
                hf.create_dataset('true_matrix', data=true_matrix)
                hf.create_dataset('pred_matrix', data=pred_matrix)
            with open('results/{}_eval.txt'.format(model_name), 'w') as fp:
                for line in eval_result:
                    fp.write(line)
                    fp.write('\n')

    @staticmethod
    def overall_evaluate():
        data = {}
        for file in glob.glob('results/*.h5'):
            model_name = file.replace('results/', '').split('_')[0]
            with h5py.File(file, 'r') as hf:
                true_matrix = hf['true_matrix'][:]
                pred_matrix = hf['pred_matrix'][:]
                data[model_name] = {'true_matrix': true_matrix, 'pred_matrix': pred_matrix}

    @staticmethod
    def make_diagrams(self):
        pass


if __name__ == '__main__':
    Evaluation.overall_evaluate()
    Evaluation.make_diagrams()
