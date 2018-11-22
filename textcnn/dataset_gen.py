import json
import numpy as np

DICT_LABEL2INT = {
    "LOVE": 0,
    "NATURE": 1,
    "SOCIAL COMMENTARIES": 2,
    "RELIGION": 3,
    "LIVING": 4,
    "RELATIONSHIPS": 5,
    "ACTIVITIES": 6,
    "ARTS & SCIENCES": 7,
    "MYTHOLOGY & FOLKLORE": 8
}

def dataset_gen():
    def preprocess(file):
        with open(file) as inputFile:
            dataset = json.load(inputFile)
        _corpus = [entity['content'] for entity in dataset]
        _labels = [entity['label'] for entity in dataset]

        _median = np.median([len(entity.split(" ")) for entity in _corpus])
        return _median, _corpus, _labels

    def data_gen(_intType, _corpus, _labels):
        _pospath = "data/" + str(_intType) + "/postive." + str(_intType)
        _negpath = "data/" + str(_intType) + "/negative." + str(_intType)
        with open(_pospath, 'w', encoding='utf8') as _pos, open(_negpath, 'w', encoding='utf8') as _neg:
            for index in range(len(_labels)):
                if _intType in _labels[index]:
                    _pos.write(_corpus[index])
                    _pos.write('\n')
                else:
                    _neg.write(_corpus[index])
                    _neg.write('\n')

    align, corpus, labels = preprocess('../corpus/ordered_corpus.json')
    for typeInt in range(9):
        data_gen(typeInt, corpus, labels)
    print('align is %d' % align)


if __name__ == '__main__':
    dataset_gen()





