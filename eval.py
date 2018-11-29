from sklearn import metrics
import h5py
import glob
import matplotlib.pyplot as plt
import numpy as np
import os


class Evaluation:
    subjects = ["LOVE", "NATU.", "S. CO.", "RELI.", "LIVI.", "RELA.", "ACT.", "A & S", "M & F"]

    def __init__(self, subjs):
        self.subjs = subjs

    def model_evaluate(self, true_matrix, pred_matrix, model_name, save=True):
        assert true_matrix.shape == pred_matrix.shape

        eval_result = list()
        eval_result.append("Model evaluation:")
        eval_result.append("Hamming Loss: {}".format(metrics.hamming_loss(true_matrix, pred_matrix)))
        eval_result.append("Accuracy evaluation: {}".format(metrics.accuracy_score(true_matrix, pred_matrix)))
        eval_result.append("Accuracy for each category:")
        eval_result.append("{:25s} {}".format("SUBJECT", "ACCURACY"))
        for index in range(len(self.subjs)):
            eval_result.append("{:25s}{:7.4f}".format(self.subjs[index],
                                                      float(sum(true_matrix[:, index] == pred_matrix[:, index]))
                                                      / float(len(true_matrix[:, index]))))

        eval_result.append("Classification Report:")
        import warnings
        warnings.filterwarnings('ignore')
        eval_result.append(metrics.classification_report(true_matrix, pred_matrix, target_names=self.subjs))

        for line in eval_result:
            print(line)

        if save:
            if not os.path.isdir('eval_data'):
                os.mkdir('eval_data')
            if not os.path.isdir('reports'):
                os.mkdir('reports')
            with h5py.File('eval_data/{}_data.h5'.format(model_name), 'w') as hf:
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
        for file in glob.glob('eval_data/*.h5'):
            model_name = file.replace('eval_data/', '').replace('_data.h5', '')
            with h5py.File(file, 'r') as hf:
                true_matrix = hf['true_matrix'][:]
                pred_matrix = hf['pred_matrix'][:]
                import warnings
                warnings.filterwarnings('ignore')
                report_dict = metrics.classification_report(true_matrix, pred_matrix, target_names=Evaluation.subjects,
                                                            output_dict=True)
                index_subj = 0
                for entity in report_dict.keys():
                    report_dict[entity]['accuracy'] = \
                        float(sum(true_matrix[:, index_subj] == pred_matrix[:, index_subj])) \
                        / float(len(true_matrix[:, index_subj]))
                    index_subj += 1
                    if index_subj == 9:
                        break

                precisions = []
                for subj in Evaluation.subjects:
                    precisions.append(report_dict[subj]['precision'])
                data[model_name] = precisions
        return data

    @staticmethod
    def make_diagrams(d, x_label='x', y_label='y', title='diagram'):
        f, axarr = plt.subplots(3, sharex='all')
        axarr[0].set_title(title)
        axarr[2].set_xlabel(x_label)
        axarr[0].ylabel(y_label)
        x = np.arange(len(Evaluation.subjects))
        plt.xticks(x, Evaluation.subjects)
        for model in d.keys():
            axarr[0].plot(x, d[model]['precision'], label=model)
            axarr[1].plot(x, d[model]['recall'], label=model)
            axarr[2].plot(x, d[model]['f1'], label=model)
        plt.legend()
        if not os.path.isdir('diagrams'):
            os.mkdir('diagrams')
        plt.savefig('diagrams/{}.png'.format('_'.join(title.lower().split(' '))))
        plt.show()


if __name__ == '__main__':
    data = Evaluation.overall_evaluate()
    Evaluation.make_diagrams(data, x_label='Subjects', y_label='Precision', title='Model Precision Diagram')
    Evaluation.make_diagrams(data, x_label='Subjects', y_label='Accuracy', title='Model Precision Diagram')
