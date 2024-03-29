from sklearn import metrics
import h5py
import glob
import matplotlib.pyplot as plt
import numpy as np
import os


class Evaluation:
    subjects = ["LOV.", "NAT.", "S.C.", "REL.", "LIV.", "REL.", "ACT.", "A&S", "M&F"]
    features = ["cv", "tfidf", "tfidf_norm", "tfidf_norm_lda-small", "tfidf_norm_lds-large"]

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
                                                      metrics.accuracy_score(true_matrix[:, index],
                                                                             pred_matrix[:, index])))

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
    def overall_evaluate(prefix=''):
        data = {}
        # load matrix data from files
        for file in glob.glob('eval_data/{}*.h5'.format(prefix)):
            tokens = file.replace('eval_data/', '').replace('_data.h5', '').split('_')
            model_name = tokens[0]
            feature_name = '_'.join(tokens[1:])
            with h5py.File(file, 'r') as hf:
                true_matrix = hf['true_matrix'][:]
                pred_matrix = hf['pred_matrix'][:]
                accuracy = []
                import warnings
                warnings.filterwarnings('ignore')
                for i in range(len(Evaluation.subjects)):
                    accuracy.append(metrics.accuracy_score(true_matrix[:, i], pred_matrix[:, i]))
                precision = metrics.precision_score(true_matrix, pred_matrix, average=None)
                recall = metrics.recall_score(true_matrix, pred_matrix, average=None)
                f1_score = metrics.f1_score(true_matrix, pred_matrix, average=None)
                if model_name not in data:
                    data[model_name] = dict()
                data[model_name][feature_name] = dict()
                data[model_name][feature_name]['accuracy'] = accuracy
                data[model_name][feature_name]['precision'] = precision
                data[model_name][feature_name]['recall'] = recall
                data[model_name][feature_name]['f1-score'] = f1_score
        return data

    @staticmethod
    def make_model_performance_diagrams(model_data, title='diagram'):
        assert len(model_data.keys()) == 5
        x = np.arange(len(Evaluation.subjects))
        markers = ['.', '*', '+', 'x', 'd']

        plt.rcParams["figure.figsize"] = (12, 7)
        fig = plt.figure(1)

        # accuracy
        plt.subplot(221)
        i = 0
        for feature in sorted(model_data.keys()):
            plt.plot(x, model_data[feature]['accuracy'], label=feature, marker=markers[i])
            i += 1
        plt.ylabel('Accuracy')
        plt.xticks(x, Evaluation.subjects)

        # precision
        plt.subplot(222)
        i = 0
        for feature in sorted(model_data.keys()):
            plt.plot(x, model_data[feature]['precision'], label=feature, marker=markers[i])
            i += 1
        plt.ylabel('Precision')
        plt.xticks(x, Evaluation.subjects)

        # recall
        plt.subplot(223)
        i = 0
        for feature in sorted(model_data.keys()):
            plt.plot(x, model_data[feature]['recall'], label=feature, marker=markers[i])
            i += 1
        plt.ylabel('Recall')
        plt.xticks(x, Evaluation.subjects)

        # f1
        plt.subplot(224)
        i = 0
        for feature in sorted(model_data.keys()):
            plt.plot(x, model_data[feature]['f1-score'], label=feature, marker=markers[i])
            i += 1
        plt.ylabel('F1-Score')
        plt.xticks(x, Evaluation.subjects)

        lgd = plt.legend(loc='lower left', bbox_to_anchor=(1.04, 0))
        suptitle = fig.suptitle(title, fontsize=20, y=0.95)
        if not os.path.isdir('diagrams'):
            os.mkdir('diagrams')
        plt.savefig('diagrams/{}.png'.format('_'.join(title.lower().split(' '))),
                    bbox_extra_artists=(lgd, suptitle,), bbox_inches='tight')

        plt.close(fig)

    @staticmethod
    def make_model_performance_eval_diagrams(model_data, features=None, title='eval diagram'):
        if features is None or len(features) == 0:
            print('Wrong parameter: features')
            return

        features = sorted(features)
        x = np.arange(len(Evaluation.subjects))
        markers = ['.', '*']

        plt.rcParams["figure.figsize"] = (12, 7)
        fig = plt.figure(1)

        # accuracy
        plt.subplot(121)
        i = 0
        for feature in features:
            plt.plot(x, model_data[feature]['accuracy'], label=feature, marker=markers[i])
            i += 1
        plt.ylabel('Accuracy')
        plt.xticks(x, Evaluation.subjects)

        # f1
        plt.subplot(122)
        i = 0
        for feature in features:
            plt.plot(x, model_data[feature]['f1-score'], label=feature, marker=markers[i])
            i += 1
        plt.ylabel('F1-Score')
        plt.xticks(x, Evaluation.subjects)

        lgd = plt.legend(loc='lower left', bbox_to_anchor=(1.04, 0))
        suptitle = fig.suptitle(title, fontsize=20, y=0.95)
        if not os.path.isdir('diagrams'):
            os.mkdir('diagrams')
        plt.savefig('diagrams/{}.png'.format('_'.join(title.lower().split(' '))),
                    bbox_extra_artists=(lgd, suptitle,), bbox_inches='tight')

        plt.close(fig)


if __name__ == '__main__':
    data = Evaluation.overall_evaluate(prefix='LogReg')
    for model_name in data.keys():
        # Evaluation.make_model_performance_diagrams(data[model_name], title='{} Performance Diagram'.format(model_name))
        Evaluation.make_model_performance_eval_diagrams(data[model_name], features=['cv', 'tfidf'],
                                                        title='{} Performance Eval Diagram'.format(model_name))
