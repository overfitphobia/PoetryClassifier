import numpy as np
from eval import Evaluation

subjects = ["LOVE", "NATURE", "SOCIAL COMMENTARIES", "RELIGION",
            "LIVING", "RELATIONSHIPS", "ACTIVITIES", "ARTS & SCIENCES", "MYTHOLOGY & FOLKLORE"]
def eval_gen():
    true, predicted = [], []
    for subj in subjects:
        inpath = './result/' + subj[0:4]
        true.append(np.load(inpath + '_true.npy').tolist())
        predicted.append(np.load(inpath + '_pred.npy').tolist())

    true_matrix, pred_matrix = np.array(true, int).T, np.array(predicted, int).T
    true_matrix[true_matrix == -1] = 0
    pred_matrix[pred_matrix == -1] = 0

    evaluation = Evaluation(subjects)
    evaluation.model_evaluate(true_matrix=true_matrix, pred_matrix=pred_matrix, model_name='textcnn')


if __name__ == '__main__':
    eval_gen()

