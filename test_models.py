from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import debias
import json
import numpy as np
import pandas as pd
import we
from scipy import spatial
import numpy as np

#load 1st and 2nd models
vanilla_model = Word2Vec.load('vanilla_model.bin')
wino_model = Word2Vec.load('wino_model.bin')

#here's where you test the first two models
# print(new_model.wv.similarity('woman', 'caring'))

# filename = r'C:\Users\Joey\Documents\nlp\dc4\GoogleNews-vectors-negative300.bin.gz'
# typical_model = KeyedVectors.load_word2vec_format(filename, binary=True)
# print(typical_model.similarity('Susan', 'caring'))

#their data was 300 dimensions
# with open('text.txt', 'r', encoding = 'utf-8') as file:
#         #print(filename)
#         data = file.read()
# data = data.split()
#print(len(data))

#vocabDict = vanilla_model.wv.vocab
#create .txt file to pass into we.py to create object
# wordEmbeddingsArray = [None] * len(vocabDict)
# counter = 0
# for word in vocabDict:
#     tmpList = vanilla_model['{}'.format(word)].astype('str').tolist()
#     #print(tmpList)
#     tmpList.append(word)

#     wordEmbeddingsArray[counter] = ' '.join(tmpList)
#     counter += 1

# #our embeddings are 100D instead of 300D
# with open('vanilla_embeddings.txt', 'w', encoding = 'utf-8') as f:
#     # for embedding in wordEmbeddingsArray:
#     #     f.writelines(embedding)
#     f.write('\n'.join(wordEmbeddingsArray))

#load 3rd and 4th models by getting bolu paper's version of a word2vec object
bolu_model = we.WordEmbedding('vanilla_embeddings.txt')
wino_and_bolu_model = we.WordEmbedding('wino_embeddings.txt')

#Lets load some gender related word lists to help us with debiasing
with open('./data/definitional_pairs.json', "r") as f:
    defs = json.load(f)
print("definitional", defs)

with open('./data/equalize_pairs.json', "r") as f:
    equalize_pairs = json.load(f)

with open('./data/gender_specific_seed.json', "r") as f:
    gender_specific_words = json.load(f)
print("gender specific", len(gender_specific_words), gender_specific_words[:10])

debias.debias(wino_and_bolu_model, gender_specific_words, defs, equalize_pairs)

def cosineSimilarity(firstVector, secondVector):
    return np.dot(firstVector, secondVector) / (np.linalg.norm(firstVector) * np.linalg.norm(secondVector))

firstVector = wino_and_bolu_model.v('he').tolist()
secondVector = wino_and_bolu_model.v('smart').tolist()

print(cosineSimilarity(firstVector, secondVector))

firstVector = wino_and_bolu_model.v('she').tolist()
secondVector = wino_and_bolu_model.v('smart').tolist()

print(cosineSimilarity(firstVector, secondVector))


# Loading list of adjectives
data = pd.read_csv('./data/adjectives.csv')
print(data.head)
print(list(data))

posMascWords = data['masc_pos_terms'].tolist()
negMascWords = data['masc_neg_terms'].tolist()
neutMascWords = data['masc_neu_terms'].tolist()
posFemWords = data['fem_pos_terms'].tolist()
negFemWords = data['fem_neg_terms'].tolist()
neutFemWords = data['fem_neu_terms'].tolist()

wordsLists = [posMascWords, negMascWords, neutMascWords, posFemWords, negFemWords, neutFemWords]
print(wordsLists)

# create pronoun vectors for testing
# Model 1
vanillaHe = vanilla_model['he']
print(vanillaHe)
vanillaShe = vanilla_model['she']

# Model 2
boluHe = bolu_model.v('he').tolist()
print(boluHe)
boluShe = bolu_model.v('she').tolist()

# Model 3
winoHe = wino_model['he']
winoShe = wino_model['she']

# Model 4
winoAndBoluHe = wino_and_bolu_model.v('he').tolist()
winoAndBoluShe = wino_and_bolu_model.v('she').tolist()


# Naming scheme: Model the embeddings are based on, sentiment+gender adjective list
vanillaPosMasc = []
vanillaNegMasc = []
vanillaNeutMasc = []
vanillaPosFem = []
vanillaNegFem = []
vanillaNeutFem = []

boluPosMasc = []
boluNegMasc = []
boluNeutMasc = []
boluPosFem = []
boluNegFem = []
boluNeutFem = []

winoPosMasc = []
winoNegMasc = []
winoNeutMasc = []
winoPosFem = []
winoNegFem = []
winoNeutFem = []

winoAndBoluPosMasc = []
winoAndBoluNegMasc = []
winoAndBoluNeutMasc = []
winoAndBoluPosFem = []
winoAndBoluNegFem = []
winoAndBoluNeutFem = []


vanillaDiffsList = [vanillaPosMasc, vanillaNegMasc, vanillaNeutMasc, vanillaPosFem, vanillaNegFem, vanillaNeutFem]
boluDiffsList = [boluPosMasc, boluNegMasc, boluNeutMasc, boluPosFem, boluNegFem, boluNeutFem]
winoDiffsList = [winoPosMasc, winoNegMasc, winoNeutMasc, winoPosFem, winoNegFem, winoNeutFem]
winoAndBoluDiffsList = [winoAndBoluPosMasc, winoAndBoluNegMasc, winoAndBoluNeutMasc, winoAndBoluPosFem, winoAndBoluNegFem, winoAndBoluNeutFem]
index = 0
for wordList in wordsLists:
    for word in wordList:
        print(word)
        #for each word in the list, compute diffs of cosine sim from balanced embeddings and regular embeddings
        try:
            vanillaVector = vanilla_model[word]
            boluVector = bolu_model.v(word).tolist()
            winoVector = wino_model[word]
            winoAndBoluVector = wino_and_bolu_model.v(word).tolist()

            # vanilla
            mascSim = cosineSimilarity(vanillaHe, vanillaVector)
            femSim = cosineSimilarity(vanillaShe, vanillaVector)
            diff = abs(mascSim - femSim)
            print(diff)
            vanillaDiffsList[index].append(diff)

            # bolu
            mascSim = cosineSimilarity(boluHe, boluVector)
            femSim = cosineSimilarity(boluShe, boluVector)
            diff = abs(mascSim - femSim)
            print(diff)
            boluDiffsList[index].append(diff)

            # wino
            mascSim = cosineSimilarity(winoHe, winoVector)
            femSim = cosineSimilarity(winoShe, winoVector)
            diff = abs(mascSim - femSim)
            print(diff)
            winoDiffsList[index].append(diff)

            # wino and bolu
            mascSim = cosineSimilarity(winoAndBoluHe, winoAndBoluVector)
            femSim = cosineSimilarity(winoAndBoluShe, winoAndBoluVector)
            diff = abs(mascSim - femSim)
            print(diff)
            winoAndBoluDiffsList[index].append(diff)


        except:
            vanillaDiffsList[index].append('NaN')
            boluDiffsList[index].append('NaN')
            winoDiffsList[index].append('NaN')
            winoAndBoluDiffsList[index].append('NaN')

    index+=1

info = {'Vanilla model + Positive Masculine words': vanillaDiffsList[0],
        'Vanilla model + Negative Masculine words': vanillaDiffsList[1],
        'Vanilla model + Neutral Masculine words': vanillaDiffsList[2],
        'Vanilla model + Positive Feminine words': vanillaDiffsList[3],
        'Vanilla model + Negative Feminine words': vanillaDiffsList[4],
        'Vanilla model + Neutral Feminine words': vanillaDiffsList[5],
        'Bolu model + Positive Masculine words': boluDiffsList[0],
        'Bolu model + Negative Masculine words': boluDiffsList[1],
        'Bolu model + Neutral Masculine words': boluDiffsList[2],
        'Bolu model + Positive Feminine words': boluDiffsList[3],
        'Bolu model + Negative Feminine words': boluDiffsList[4],
        'Bolu model + Neutral Feminine words': boluDiffsList[5],
        'Wino model + Positive Masculine words': winoDiffsList[0],
        'Wino model + Negative Masculine words': winoDiffsList[1],
        'Wino model + Neutral Masculine words': winoDiffsList[2],
        'Wino model + Positive Feminine words': winoDiffsList[3],
        'Wino model + Negative Feminine words': winoDiffsList[4],
        'Wino model + Neutral Feminine words': winoDiffsList[5],
        'Wino and Bolu model + Positive Masculine words': winoAndBoluDiffsList[0],
        'Wino and Bolu model + Negative Masculine words': winoAndBoluDiffsList[1],
        'Wino and Bolu model + Neutral Masculine words': winoAndBoluDiffsList[2],
        'Wino and Bolu model + Positive Feminine words': winoAndBoluDiffsList[3],
        'Wino and Bolu model + Negative Feminine words': winoAndBoluDiffsList[4],
        'Wino and Bolu model + Neutral Feminine words': winoAndBoluDiffsList[5],}

df = pd.DataFrame(info)
df.to_csv(r'results.csv', index=False, header=True)
print(df.head())
