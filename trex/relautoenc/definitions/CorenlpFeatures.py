import nltk
import re, string
# import settings
import pickle

"""
parsing = 0
entities = 1
trig = 2
sentence = 3
pos = 4
docPath = 5
"""

BOW = 0
E1, E2 = 1, 3
E1TYPE, E2TYPE = 2, 4
PAIRTYPE = 5
PATH = 6
POS = 7
TRIG = 8

    
#  ======= Relation features =======
stopwords_list = nltk.corpus.stopwords.words('english')
_digits = re.compile('\d')

def bow(info, arg1, arg2):
    return info[BOW].split()

def bow_clean(info, arg1, arg2):
    return info[BOW].split()

def bigrams(info, arg1, arg2):
    between = bow_clean(info, arg1, arg2)
    return [x[0]+'_'+x[1] for x in zip(between, between[1:])]

def trigrams(info, arg1, arg2):
    between = bow_clean(info, arg1, arg2)
    return [x[0]+'_'+x[1]+'_'+x[2] for x in zip(between, between[1:], between[2:])]

def skiptrigrams(info, arg1, arg2):
    between = bow_clean(info, arg1, arg2)
    return [x[0]+'_X_'+x[2] for x in zip(between, between[1:], between[2:])]

"""
def skipfourgrams(info, arg1, arg2):
    between = info[sentence][info[sentence].find(arg1):info[sentence].rfind(arg2)+len(arg2)].split()
    tmp = []
    for word in between:
        for pun in string.punctuation:
            word = word.strip(pun)
        if word != '':
            tmp.append(word.lower())
    return [x[0]+'_X_'+x[2] + '_' + x[3] for x in zip(tmp, tmp[1:], tmp[2:], tmp[3:])] +\
           [x[0]+'_'+x[1]+'_X_' + x[3] for x in zip(tmp, tmp[1:], tmp[2:], tmp[3:])]
"""
def trigger(info, arg1, arg2):
    return info[TRIG].replace('TRIGGER:', '')

def entityTypes(info, arg1, arg2):
    return info[PAIRTYPE]

def entity1Type(info, arg1, arg2):
    return info[E1TYPE]

def entity2Type(info, arg1, arg2):
    return info[E2TYPE]

def arg1(info, arg1, arg2):
    return info[E1]

def arg1_lower(info, arg1, arg2):
    return info[E1].lower()

def arg1unigrams(info, arg1, arg2):
    return info[E1].lower().split()

def arg2(info, arg1, arg2):
    return info[E2]

def arg2_lower(info, arg1, arg2):
    return info[E2].lower()

def arg2unigrams(info, arg1, arg2):
    return info[E2].lower().split()

def lexicalPattern(info, arg1, arg2):
    return info[PATH]

def dependencyParsing(info, arg1, arg2):
    return info[PATH]

def posPatternPath(info, arg1, arg2):
    return info[POS]


def getBasicCleanFeatures():
    features = [trigger, entityTypes, arg1_lower, arg2_lower, bow_clean, entity1Type, entity2Type, lexicalPattern,
                posPatternPath]
    return features

