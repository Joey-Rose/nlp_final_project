from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import debias
import json
import numpy as np
import we
from scipy import spatial

#load model
new_model = Word2Vec.load('balanced_model.bin')
# print(new_model.wv.similarity('woman', 'caring'))

# filename = r'C:\Users\Joey\Documents\nlp\dc4\GoogleNews-vectors-negative300.bin.gz'
# typical_model = KeyedVectors.load_word2vec_format(filename, binary=True)
# print(typical_model.similarity('man', 'caring'))

#their data was 300 dimensions
# with open('text.txt', 'r', encoding = 'utf-8') as file:
#         #print(filename)
#         data = file.read()
# data = data.split()
#print(len(data))

# vocabDict = new_model.wv.vocab
# #create .txt file to pass into we.py to create object
# wordEmbeddingsArray = [None] * len(vocabDict)
# counter = 0
# for word in vocabDict:
#     tmpList = new_model['{}'.format(word)].astype('str').tolist()
#     #print(tmpList)
#     tmpList.append(word)
    
#     wordEmbeddingsArray[counter] = ' '.join(tmpList)
#     counter += 1

# #our embeddings are 100D instead of 300D
# with open('embeddings.txt', 'w', encoding = 'utf-8') as f:
#     # for embedding in wordEmbeddingsArray:
#     #     f.writelines(embedding)
#     f.write('\n'.join(wordEmbeddingsArray))

# load google news word2vec
E = we.WordEmbedding('embeddings.txt')
#Lets load some gender related word lists to help us with debiasing
with open('./data/definitional_pairs.json', "r") as f:
    defs = json.load(f)
print("definitional", defs)

with open('./data/equalize_pairs.json', "r") as f:
    equalize_pairs = json.load(f)

with open('./data/gender_specific_seed.json', "r") as f:
    gender_specific_words = json.load(f)
print("gender specific", len(gender_specific_words), gender_specific_words[:10])

debias.debias(E, gender_specific_words, defs, equalize_pairs)

def cosineSimilarity(firstVector, secondVector):
    return 1 - spatial.distance.cosine(firstVector, secondVector)

firstVector = E.v('man').tolist()
secondVector = E.v('fragile').tolist()

print(cosineSimilarity(firstVector, secondVector))
