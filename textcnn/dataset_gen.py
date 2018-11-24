import json
import numpy as np
import os
from sklearn.model_selection import train_test_split

DICT_LABEL2INT = {
    0: "LOVE",
    1: "NATURE",
    2: "SOCIAL COMMENTARIES",
    3: "RELIGION",
    4: "LIVING",
    5: "RELATIONSHIPS",
    6: "ACTIVITIES",
    7: "ARTS & SCIENCES",
    8: "MYTHOLOGY & FOLKLORE"
}


def dataset_gen(jsonFile):
    def preprocessor(file):
        with open(file) as inputFile:
            dataset = json.load(inputFile)
        corpus = [entity['content'] for entity in dataset]
        labels = [entity['label'] for entity in dataset]

        median = np.median([len(entity.split(" ")) for entity in corpus])
        return median, corpus, labels

    def format_write(pos_path, neg_path, x, y, int_type):
        with open(pos_path, 'w', encoding='utf8') as pos, open(neg_path, 'w', encoding='utf8') as neg:
            for i in range(len(y)):
                if int_type in y[i]:
                    pos.write(x[i])
                    pos.write('\n')
                else:
                    neg.write(x[i])
                    neg.write('\n')

    def data_gen(intType, corpus, labels):
        if not os.path.exists("data/" + DICT_LABEL2INT[intType][0:4]):
            os.mkdir("data/" + DICT_LABEL2INT[intType][0:4])
        posPath = "data/" + DICT_LABEL2INT[intType][0:4] + "/" + DICT_LABEL2INT[intType][0:4].lower() + ".positive"
        negPath = "data/" + DICT_LABEL2INT[intType][0:4] + "/" + DICT_LABEL2INT[intType][0:4].lower() + ".negative"
        x_train, x_dev, y_train, y_dev = \
            train_test_split(corpus, labels, test_size=0.1, shuffle=True, random_state=17)
        format_write(posPath + ".train", negPath + ".train", x_train, y_train, intType)
        format_write(posPath + ".test", negPath + ".test", x_dev, y_dev, intType)

    alignData, corpusData, labelsData = preprocessor(jsonFile)
    for typeIntData in range(9):
        data_gen(typeIntData, corpusData, labelsData)
    print('align is %d' % alignData)


if __name__ == '__main__':
    dataset_gen('../corpus/corpus_with_stopwords.json')





