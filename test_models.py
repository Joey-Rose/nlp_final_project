from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import debias
import json
import numpy as np
import pandas as pd
import we
from scipy import spatial

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
    return 1 - spatial.distance.cosine(firstVector, secondVector)

firstVector = wino_and_bolu_model.v('he').tolist()
secondVector = wino_and_bolu_model.v('smart').tolist()

print(cosineSimilarity(firstVector, secondVector))

firstVector = wino_and_bolu_model.v('she').tolist()
secondVector = wino_and_bolu_model.v('smart').tolist()

print(cosineSimilarity(firstVector, secondVector))

# heVector = E.v('he').tolist()
# sheVector = E.v('she').tolist()
# posMascWords = ['faithful', 'responsible', 'adventurous', 'grand', 'worthy', 'brave', 'good', 'normal', 'ambitious',
#                 'gallant', 'mighty', 'loyal', 'valiant', 'courteous', 'powerful', 'rational', 'supreme', 'meritorious',
#                 'serene', 'godlike', 'noble', 'rightful', 'eager', 'financial', 'chivalrous']
# negMascWords = ['unjust', 'dumb', 'violent', 'weak', 'evil', 'stupid', 'petty', 'brutal', 'wicked', 'rebellious', 'bad',
#             'worthless', 'hostile', 'careless', 'unsung', 'abusive', 'financial', 'feudal', 'false', 'feeble',
#             'impotent', 'dishonest', 'ungrateful','unfaithful', 'incompetent']
# neutMascWords = ['german', 'teutonic', 'financial', 'feudal', 'later', 'austrian', 'feudatory', 'maternal', 'bavarian',
#                  'negro', 'paternal', 'frankish', 'welsh', 'eccliastical', 'rural', 'persian', 'belted', 'swiss',
#                  'finnish', 'national', 'priestly', 'merovingian', 'capetian', 'prussian', 'racial']
# posFemWords = ['pretty', 'fair', 'beautiful', 'lovely', 'charming', 'sweet', 'grand', 'stately', 'attractive', 'chaste',
#                'virtuous', 'fertile', 'delightful', 'gentle', 'privileged', 'romantic', 'enchanted', 'kindly',
#                'elegant', 'dear', 'devoted', 'beauteous', 'sprightly', 'beloved', 'pleasant']
# negFemWords = ['horrible', 'destructive', 'notorious', 'dreary', 'ugly', 'weird', 'harried', 'diabetic', 'discontented',
#                'infected', 'unmarried', 'unequal', 'widowed', 'unhappy', 'horrid', 'pitiful', 'frightful', 'artificial',
#                'sullen', 'hysterical', 'awful', 'haughty', 'terrible', 'damned', 'topless']
# neutFemWords = ['virgin', 'alleged', 'maiden', 'russian', 'fair', 'widowed', 'grand', 'byzantine', 'fashionable',
#                 'aged', 'topless', 'withered', 'colonial', 'diabetic', 'burlesque', 'blonde', 'parisian', 'clad',
#                 'female', 'oriental', 'ancient', 'feminist', 'matronly', 'pretty', 'asiatic']

# wordsLists = [posMascWords, negMascWords, neutMascWords, posFemWords, negFemWords, neutFemWords]

# # Naming scheme: Model the embeddings are based on, sentiment+gender adjective list
# balPosMasc = []
# balNegMasc = []
# balNeutMasc = []
# balPosFem = []
# balNegFem = []
# balNeutFem = []

# balDiffsList = [balPosMasc, balNegMasc, balNeutMasc, balPosFem, balNegFem, balNeutFem]

# regPosMasc = []
# regNegMasc = []
# regNeutMasc = []
# regPosFem = []
# regNegFem = []
# regNeutFem = []

# regDiffsList = [regPosMasc, regNegMasc, regNeutMasc, regPosFem, regNegFem, regNeutFem]

# # for word in negMascWords:
# #     negVector = E.v(word).tolist()
# #     mascSim = cosineSimilarity(heVector, negVector)
# #     femSim = cosineSimilarity(sheVector, negVector)
# #     diff = mascSim - femSim
# #     print(word)
# #     print('He + MascAdj Cosine Sim: ' + str(mascSim))
# #     print('She + MascAdj Cosine Sim: ' + str(femSim))
# #     print('Difference: ' + str(diff) + '\n')

# index = 0
# for wordList in wordsLists:
#     for word in wordList:
#         print(word)
#         #for each word in the list, compute diffs of cosine sim from balanced embeddings and regular embeddings
#         try:
#             balVector = E.v(word).tolist()
#             mascSim = cosineSimilarity(heVector, balVector)
#             femSim = cosineSimilarity(sheVector, balVector)
#             diff = mascSim - femSim
#             print(diff)
#             balDiffsList[index].append(diff)

#             regVector = R.v(word).tolist()
#             mascSim = cosineSimilarity(heVector, regVector)
#             femSim = cosineSimilarity(sheVector, regVector)
#             diff = mascSim - femSim
#             print(diff)
#             regDiffsList[index].append(diff)
#         except:
#             balDiffsList[index].append('NaN')
#             regDiffsList[index].append('NaN')

#     index+=1

# print(balDiffsList)
# print(regDiffsList)

# info = {'Balanced model + Positive Masculine words': balDiffsList[0],
#         'Balanced model + Negative Masculine words': balDiffsList[1],
#         'Balanced model + Neutral Masculine words': balDiffsList[2],
#         'Balanced model + Positive Feminine words': balDiffsList[3],
#         'Balanced model + Negative Feminine words': balDiffsList[4],
#         'Balanced model + Neutral Feminine words': balDiffsList[5],
#         'Regular model + Positive Masculine words': regDiffsList[0],
#         'Regular model + Negative Masculine words': regDiffsList[1],
#         'Regular model + Neutral Masculine words': regDiffsList[2],
#         'Regular model + Positive Feminine words': regDiffsList[3],
#         'Regular model + Negative Feminine words': regDiffsList[4],
#         'Regular model + Neutral Feminine words': regDiffsList[5],}

# df = pd.DataFrame(info)
# df.to_csv(r'results.csv', index=False, header=True)
# print(df.head())
