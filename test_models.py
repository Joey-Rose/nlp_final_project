from gensim.models import Word2Vec

#load model
new_model = Word2Vec.load('regular_model.bin')
print(new_model.wv.similarity('woman', 'caring'))