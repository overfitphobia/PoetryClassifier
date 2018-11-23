import json
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


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

        self.test_rate = 0.2
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
        self.subjects = ["LOVE", "NATURE", "SOCIAL COMMENTARIES", "RELIGION",
                         "LIVING", "RELATIONSHIPS", "ACTIVITIES", "ARTS & SCIENCES", "MYTHOLOGY & FOLKLORE"]

        # TODO: Remember to fine-tune the boolean parameter below to accelerate the proprocess
        # TODO: Each time you have employed modification, please set these parameters and remain the save in a json file
        self.reload(reload=True)
        self.tokenizer(save=False, act=False, rmstopwords=True)
        self.corpus = [entity['content'] for entity in self.dataset]
        self.labels = [entity['label'] for entity in self.dataset]

    def reload(self, reload=False):
        if reload:
            with open(self.savePath) as file:
                self.dataset = json.load(file)
        else:
            with open(self.rootPath) as file:
                self.dataset = json.load(file)

    def tokenizer(self, save=False, act=True, rmstopwords=False):
        if act:
            stopwords_set = set(stopwords.words('english'))
            for index in range(len(self.dataset)):
                if rmstopwords:
                    tokens = [word for word in word_tokenize(self.dataset[index]['content'].lower())
                              if (word not in stopwords_set and word not in string.punctuation)]
                else:
                    tokens = [word for word in word_tokenize(self.dataset[index]['content'].lower())
                              if word not in string.punctuation]
                labels = [self.DICT_LABEL2INT[label] for label in self.dataset[index]['label']]
                self.dataset[index]['content'] = " ".join(tokens)
                # self.dataset[index]['content'] = tokens
                self.dataset[index]['label'] = labels
            if save:
                with open(self.savePath, 'w') as file:
                    json.dump(self.dataset, file, indent=4)

    def dataset_gen(self, subj, valid=False):
        target = self.DICT_LABEL2INT[subj]
        label_encoded = [1 if target in entity['label'] else -1 for entity in self.dataset]
        x_train, x_test, y_train, y_test = \
            train_test_split(self.corpus, label_encoded, test_size=self.test_rate, shuffle=False, random_state=17)
        if valid:
            x_train, x_dev, y_train, y_dev = \
                train_test_split(x_train, y_train, test_size=0.1, shuffle=False, random_state=17)
        else:
            x_dev, y_dev = None, None
        return x_train, x_test, x_dev, y_train, y_test, y_dev
