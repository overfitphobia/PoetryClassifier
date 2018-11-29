from sklearn import metrics
import h5py
import glob
import matplotlib.pyplot as plt
import numpy as np
import os


class Evaluation:
    subjects = ["LOVE", "NATURE", "S. CO.", "RELI.", "LIVI.", "RELA.", "ACT.", "A & S", "M & F"]

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
            with h5py.File('temp_data/{}_data.h5'.format(model_name), 'w') as hf:
                hf.create_dataset('true_matrix', data=true_matrix)
                hf.create_dataset('pred_matrix', data=pred_matrix)
            with open('reports/{}_eval.txt'.format(model_name), 'w') as fp:
                for line in eval_result:
                    fp.write(line)
                    fp.write('\n')

    @staticmethod
    def overall_evaluate():
        data = {}
        # load matrix data from files
        for file in glob.glob('temp_data/*.h5'):
            model_name = file.replace('temp_data/', '').replace('_data.h5', '')
            with h5py.File(file, 'r') as hf:
                true_matrix = hf['true_matrix'][:]
                pred_matrix = hf['pred_matrix'][:]
                import warnings
                warnings.filterwarnings('ignore')
                report_dict = metrics.classification_report(true_matrix, pred_matrix, target_names=Evaluation.subjects,
                                                            output_dict=True)
                precisions = []
                for subj in Evaluation.subjects:
                    precisions.append(report_dict[subj]['precision'])
                data[model_name] = precisions
        return data

    @staticmethod
    def make_diagrams(d, x_label='x', y_label='y', title='diagram'):
        plt.rcParams["figure.figsize"] = (8, 6)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        x = np.arange(len(Evaluation.subjects))
        plt.xticks(x, Evaluation.subjects)
        for model in d.keys():
            y = d[model]
            assert len(y) == len(Evaluation.subjects)
            plt.plot(x, y, label=model)
        plt.legend()
        if not os.path.isdir('diagrams'):
            os.mkdir('diagrams')
        plt.savefig('diagrams/{}.png'.format('_'.join(title.lower().split(' '))))
        plt.show()


if __name__ == '__main__':
    data = Evaluation.overall_evaluate()
    Evaluation.make_diagrams(data, x_label='Subjects', y_label='Precision', title='Model Precision Diagram')
