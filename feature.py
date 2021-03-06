from preprocess import PreProcess
import sklearn.feature_extraction.text as feature_ext
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import Word2Vec


class Feature:
    def __init__(self, trained=False):
        self.vector = feature_ext.CountVectorizer(ngram_range=(1, 2))
        self.tfidftransform = feature_ext.TfidfTransformer(norm='l2')
        self.ldatransform_small = LatentDirichletAllocation(n_components=50)
        self.ldatransform_large = LatentDirichletAllocation(n_components=100)
        if trained:
            self.w2vmodel = Word2Vec.load('./model/word2vector.model')

    @staticmethod
    def word2vector(dataset, save=True):
        print("train the whole corpus with word2vec")
        corpus = [entity['content'] for entity in dataset]
        model = Word2Vec(corpus, size=100, window=5, min_count=1)
        if save:
            model.save('./model/word2vector.model')
            print("model has been successfully saved in word2vector.model")


if __name__ == '__main__':
    preprocessor = PreProcess(root='./corpus/corpus.json', save='./corpus/unordered_corpus_list.json')
    train_w2v = Feature(trained=True)
    train_w2v.word2vector(preprocessor.dataset, save=True)
    print(train_w2v.w2vmodel.most_similar(['love']))

