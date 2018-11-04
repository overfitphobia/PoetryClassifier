import json
import string
from nltk import word_tokenize
from nltk.corpus import stopwords

class PreProcess:
    """
    [1] [Y] tokenization : decapticalize to lower case,
                       remove stopwords and punctuation, treating them as noise
    [2] optimal solution:
        [N] (a) pos_tag with nltk
        [N] (b) enchant of nltk to have words' grammar checks
    """
    def __init__(self, root, save):
        self.rootPath = root
        self.savePath = save
        self.DICT_LABEL2INT = {
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

        # TODO: Remember to fine-tune the boolean parameter below to accelerate the proprocess
        # TODO: Each time you have employed modification, please set these parameters and remain the save in a json file
        self.reload(reload=True)
        self.tokenizer(save=True, act=False)

    def reload(self, reload=False):
        if reload:
            with open(self.savePath) as file:
                self.dataset = json.load(file)
        else:
            with open(self.rootPath) as file:
                self.dataset = json.load(file)

    def tokenizer(self, save=False, act=True):
        if act:
            stopwords_set = set(stopwords.words('english'))
            for index in range(len(self.dataset)):
                tokens = [word for word in word_tokenize(self.dataset[index]['content'].lower())
                          if (word not in stopwords_set and word not in string.punctuation)]
                labels = [self.DICT_LABEL2INT[label] for label in self.dataset[index]['label']]
                self.dataset[index]['content'] = " ".join(tokens)
                self.dataset[index]['label'] = labels
            if save:
                with open(self.savePath, 'w') as file:
                    json.dump(self.dataset, file, indent=4)