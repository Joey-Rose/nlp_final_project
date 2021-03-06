from gensim.models import Word2Vec
import string
from nltk.tokenize import sent_tokenize
import os
import spacy

nlp = spacy.load('en_core_web_sm')

#dictionary used to quickly swap gendered words when doubling corpus size
genderedWords = {
'actor': 'actress',
'actors': 'actresses',
'actress': 'actor',
'actresses': 'actors',
'airman': 'airwoman',
'airmen': 'airwomen',
'airwoman': 'airman',
'airwomen': 'airmen',
'aunt': 'uncle',
'aunts': 'uncles',
'boy': 'girl',
'boys': 'girls',
'bride': 'groom',
'brides': 'grooms',
'brother': 'sister',
'brothers': 'sisters',
'businessman': 'businesswoman',
'businessmen': 'businesswomen',
'businesswoman': 'businessman',
'businesswomen': 'businessmen',
'chairman': 'chairwoman',
'chairmen': 'chairwomen',
'chairwoman': 'chairman',
'chairwomen': 'chairman',
'chick': 'dude',
'chicks': 'dudes',
'dad': 'mom',
'dads': 'moms',
'daddy': 'mommy',
'daddies': 'mommies',
'daughter': 'son',
'daughters': 'sons',
'dude': 'chick',
'dudes': 'chicks',
'father': 'mother',
'fathers': 'mothers',
'female': 'male',
'females': 'males',
'gal': 'guy',
'gals': 'guys',
'mentleman': 'lady',
'gentlemen': 'ladies',
'girl': 'boy',
'girls': 'boys',
'granddaughter': 'grandson',
'granddaughters': 'grandsons',
'grandson': 'granddaughter',
'grandsons': 'granddaughters',
'groom': 'bride',
'grooms': 'brides',
'guy': 'girl',
'guys': 'girls',
'he': 'she',
'herself': 'himself',
'him': 'her',
'himself': 'herself',
'his': 'her',
'husband': 'wife',
'husbands': 'wives',
'king': 'queen',
'kings': 'queens',
'ladies': 'gentlemen',
'lady': 'gentleman',
'lord': 'lady',
'lords': 'ladies',
"ma'am": 'sir',
'male': 'female',
'males': 'females',
'man': 'woman',
'men': 'women',
'miss': 'sir',
'mom': 'dad',
'moms': 'dads',
'mommy': 'daddy',
'mommies': 'daddies',
'mother': 'father',
'mothers': 'fathers',
'mr.': 'mrs.',
'mrs.': 'mr.',
'ms.': 'mr.',
'policeman': 'policewoman',
'policewoman': 'policeman',
'prince': 'princess',
'princes': 'princesses',
'princess': 'prince',
'princesses': 'princes',
'queen': 'king',
'queens': 'kings',
'she': 'he',
'sir': "ma'am",
'sister': 'brother',
'sisters': 'brothers',
'son': 'daughter',
'sons': 'daughters',
'spokesman': 'spokeswoman',
'spokesmen': 'spokeswomen',
'spokeswoman': 'spokesman',
'spokeswomen': 'spokesmen',
'uncle': 'aunt',
'uncles': 'aunts',
'wife': 'husband',
'wives': 'husbands',
'woman': 'man',
'women': 'men',
'cowboy': 'cowgirl',
'cowboys': 'cowgirls',
'camerawomen': 'cameramen',
'cameraman': 'camerawoman',
'busboy': 'busgirl',
'busboys': 'busgirls',
'bellboy': 'bellgirl',
'bellboys': 'bellgirls',
'barman': 'barwoman',
'barmen': 'barwomen',
'tailor': 'seamstress',
'tailors': 'seamstress',
'governor': 'governess',
'governors': 'governesses',
'adultor': 'adultress',
'adultors': 'adultresses',
'god': 'godess',
'gods': 'godesses',
'host': 'hostess',
'hosts': 'hostesses',
'abbot': 'abbess',
'abbots': 'abbesses',
'bachelor': 'spinster',
'bachelors': 'spinsters',
'baron': 'baroness',
'barons': 'barnoesses',
'beau': 'belle',
'beaus': 'belles',
'bridegroom': 'bride',
'bridegrooms': 'brides',
'duke': 'duchess',
'dukes': 'duchesses',
'emperor': 'empress',
'emperors': 'empresses',
'enchanter': 'enchantress',
'fiance': 'fiancee',
'fiances': 'fiancees',
'gentleman': 'lady',
'grandfather': 'grandmother',
'grandfathers': 'grandmothers',
'headmaster': 'headmistress',
'headmasters': 'headmistresses',
'hero': 'heroine',
'heros': 'heroines',
'lad': 'lass',
'lads': 'lasses',
'landlord': 'landlady',
'landlords': 'landladies',
'manservant': 'maidservant',
'manservants': 'maidservants',
'marquis': 'marchioness',
'masseur': 'masseuse',
'masseurs': 'masseuses',
'master': 'mistress',
'masters': 'mistresses',
'monk': 'nun',
'monks': 'nuns',
'nephew': 'niece',
'nephews': 'nieces',
'priest': 'priestess',
'priests': 'priestesses',
'sorcerer': 'sorceress',
'sorcerers': 'sorceresses',
'stepfather': 'stepmother',
'stepfathers': 'stepmothers',
'stepson': 'stepdaughter',
'stepsons': 'stepdaughters',
'steward': 'stewardess',
'stewards': 'stewardesses',
'waiter': 'waitress',
'waiters': 'waitresses',
'widower': 'widow',
'widowers': 'widows',
'wizard': 'witch',
'wizards': 'witches',
'cowgirl': 'cowboy',
'cowgirls': 'cowboys',
'cameramen': 'camerawomen',
'camerawoman': 'cameraman',
'busgirl': 'busboy',
'busgirls': 'busboys',
'bellgirl': 'bellboy',
'bellgirls': 'bellboys',
'barwoman': 'barman',
'barwomen': 'barmen',
'seamstress': 'tailor',
'seamstresses': 'tailors',
'governess': 'governor',
'governesses': 'governors',
'adultress': 'adultor',
'adultresses': 'adultors',
'godess': 'god',
'godesses': 'gods',
'hostess': 'host',
'hostesses': 'hosts',
'abbess': 'abbot',
'abbesses': 'abbots',
'spinster': 'bachelor',
'spinsters': 'bachelors',
'baroness': 'baron',
'barnoesses': 'barons',
'belle': 'beau',
'belles': 'beaus',
'duchess': 'duke',
'duchesses': 'dukes',
'empress': 'emperor',
'empresses': 'emperors',
'enchantress': 'enchanter',
'fiancee': 'fiance',
'grandmother': 'grandfather',
'grandmothers': 'grandfathers',
'headmistress': 'headmaster',
'headmistresses': 'headmasters',
'heroine': 'hero',
'heroines': 'heros',
'lass': 'lad',
'lasses': 'lads',
'landlady': 'landlord',
'landladies': 'landlords',
'maidservant': 'manservant',
'maidservants': 'manservants',
'marchioness': 'marquis',
'masseuse': 'masseur',
'masseuses': 'masseurs',
'mistress': 'master',
'mistresses': 'masters',
'nun': 'monk',
'nuns': 'monks',
'niece': 'nephew',
'nieces': 'nephews',
'priestess': 'priest',
'priestesses': 'priests',
'sorceress': 'sorcerer',
'sorceresses': 'sorcerers',
'stepmother': 'stepfather',
'stepmothers': 'stepfathers',
'stepdaughter': 'stepson',
'stepdaughters': 'stepsons',
'stewardess': 'steward',
'stewardesses': 'stewards',
'waitress': 'waiter',
'waitresses': 'waiters',
'widow': 'widower',
'widows': 'widowers',
'witch': 'wizard',
'witches': 'wizards',
}

#big list that sentences get appended to
sentences = []

directory = r'./novels'
for filename in os.listdir(directory):
    #define training data
    with open(r'./novels/{}'.format(filename), 'r', encoding = 'utf-8') as file:
        #print(filename)
        data = file.read().replace('\n', ' ')
    
    
    #break up document into a list of sentences
    sentencesList = sent_tokenize(data)

    #append normal version of every sentence and its gender-swapped version
    for sentence in sentencesList:
        #remove punctuation
        for c in string.punctuation:
            sentence = sentence.replace(c," ")
        #turn sentence into a list of words
        words = sentence.split()
        tmpList1 = []
        #tmpList2 = []
        for i in range(len(words)):
            word = words[i]
            
            #if word is an empty string, skip over it!
            if word[0] == ' ':
                continue
            
            # #if the word is a gendered noun/pronoun, swap it!
            # #if the gendered word is lowercase, the swap is super easy
            # if word in genderedWords:
            #     tmpList2.append(genderedWords['{}'.format(word)])
            
            # #if the gendered word is uppercase, cast it as lowercase and swap it with its uppercased counterpart
            # elif word.lower() in genderedWords:
            #     tmpList2.append(genderedWords['{}'.format(word.lower())].capitalize())

            # #else carry the word over to both lists
            # else:
            #     tmpList2.append(word)
            tmpList1.append(word)

        #add these sentence pairs to the big list!
        sentences.append(tmpList1)
        #sentences.append(tmpList2)

#print(sentences)
#train model
model = Word2Vec(sentences, min_count = 1)
#save model
model.save('vanilla_model.bin')
print('success')