import json
import numpy as np
import os

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

        median = np.median([len(entity.split(" ")) for entity in _corpus])
        return median, corpus, labels

    def data_gen(intType, corpus, labels):
        if not os.path.exists("data/" + DICT_LABEL2INT[intType][0:4]):
            os.mkdir("data/" + DICT_LABEL2INT[intType][0:4])
        posPath = "data/" + DICT_LABEL2INT[intType][0:4] + "/" + DICT_LABEL2INT[intType][0:4].lower() + ".positive"
        negPath = "data/" + DICT_LABEL2INT[intType][0:4] + "/" + DICT_LABEL2INT[intType][0:4].lower() + ".negative"
        with open(posPath, 'w', encoding='utf8') as pos, open(negPath, 'w', encoding='utf8') as neg:
            for index in range(len(labels)):
                if intType in labels[index]:
                    pos.write(corpus[index])
                    pos.write('\n')
                else:
                    neg.write(corpus[index])
                    neg.write('\n')

    alignData, corpusData, labelsData = preprocessor(jsonFile)
    for typeIntData in range(9):
        data_gen(typeIntData, corpusData, labelsData)
    print('align is %d' % alignData)


if __name__ == '__main__':
    dataset_gen('../corpus/ordered_corpus.json')





