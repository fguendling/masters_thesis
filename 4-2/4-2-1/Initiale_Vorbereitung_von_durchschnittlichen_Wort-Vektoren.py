# Original script: main_testwordvec1.py
# Augrund der Datenbankverbindung ist dieses Skript nicht ohne Weiteres ausführbar.

# -*- coding: utf-8 -*-
import sys
sys.path.append('../')  
#^ path to "util" folder should actually go to path of os...
from util.globals import GetBaseDirectory, _log
from prep.connectdb import *
import spacy
from spacy.tokens import Span
from spacy.matcher import Matcher
from datamodel.convertedparagraph import ConvertedParagraph
from datamodel.converteddocument import ConvertedDocument
from datamodel.ontologywordvector import OntologyWordVector
from datamodel.ontology import OntologyDefinition
from datamodel.uploadfile import UploadFile
from mongoengine import *
from pprint import pprint
import os
import glob
import fitz
import ftfy
from spacy.lang.en.stop_words import STOP_WORDS
import pickle 
import numpy as np

db = None    # set by connectdb()
nlp = None   

def get_mean_vector(word2vec_model, words):
    # remove out-of-vocabulary words
    words = [word for word in words if word in word2vec_model.vocab]
    if len(words) >= 1:
        return np.mean(word2vec_model[words], axis=0)
    else:
        return []

lTops = [
     ["REQ", r"/Users/felix/FRA_UAS/master/external_pdf/reqs"],
     ["ARCH", r"/Users/felix/FRA_UAS/master/external_pdf/sysarch"],
     ["DESIGN", r"/Users/felix/FRA_UAS/master/external_pdf/sysdesign"],
     ["LOG", r"/Users/felix/FRA_UAS/master/external_pdf/logistics"],
]

lBlockedWords= [
    "system",
    "datum",
    "user",
    "service",
    "model",
    "use",
    "process",
    "base",
    "time",
    "level",
    "figure",
    "support",
    "requirement",
    "management",
    "software",
    "access",
    "business",
    "include",
    "driver",
    "content",
    "number",
    "data"]

def PrepareCorpus(pSelTopics = ["REQ", "ARCH", "DESIGN", "LOG"]):

    for (sTop, sDir) in lTops:

        if not sTop in pSelTopics:
            continue
        
        TextCorpus.objects(Topic=sTop).delete()

        for sF in glob.glob(os.path.join(sDir,'*.pdf')):
            
            print("reading %s" % sF)
            _pdf = None
            try:
                _pdf = fitz.open(sF)
            except:
                print("reading failed")
            
            if _pdf != None:

                _Text = ""
                _C = 0
                print("reading text")
                for page in _pdf:
                    _Text += page.getText().replace("\n", "").replace("\t", "") + " "
                    _C += 1

                print(".. %s pages" % _C)

                print("converting")
                _Text = ftfy.fix_encoding(_Text)
                _Text = ftfy.fixes.fix_surrogates(_Text)
                _Text = _Text.replace("\uFFFD", "");
                _Text = _Text.replace("\uf0b7", "");
                _nlpdoc = nlp(_Text)
                print("%s tokens" % len(_nlpdoc))
                
                _lemma = ""
                
                print("lemma and stopwords")
                _C = 0
                token_list = []
                token_dict = {}
                for token in _nlpdoc:
                    lexeme = nlp.vocab[token.text]
                    _s = token.lemma_.lower()
                    
                    if (lexeme.is_stop == False and
                        token.is_space == False and
                        token.is_digit == False  and
                        len(token.text) > 1 and
                        not _s in lBlockedWords and
                        token.is_punct == False and
                        token.lemma_ != '-PRON-'):
                        c = token_dict.setdefault(_s, 0) 
                        token_dict[_s] +=1
                        _lemma += _s +" "
                        _C += 1
                print("%s final words " % _C)

                allw = [ [t, token_dict[t]] for t in token_dict]
                allw.sort(key=lambda x: x[1], reverse=True)
                print(allw)
    
                finDoc = TextCorpus()
                finDoc.OriginalFileName = sF
                finDoc.Topic = sTop
                finDoc.OriginalText = _Text
                finDoc.LemmaNoStopText = _lemma
                if len(allw) > 50:
                    finDoc.Top50Words = allw[:50]
                else:
                    finDoc.Top50Words = allw
                    
                finDoc.save()
                print("document saved")

allvecs = {}

def SaveVecs():
    
    for tc in TextCorpus.objects():
        
        print(tc.OriginalFileName)
        _nlp = nlp(tc.LemmaNoStopText)
        if not tc.Topic in allvecs:
            allvecs[tc.Topic] = []
        allvecs[tc.Topic].append(_nlp.vector)
        tc.SpacyVector = pickle.dumps(_nlp.vector)
        tc.save()
        
    for k in allvecs:
        print("topic %s sums:" % k)
        for vv in allvecs[k]:
            print(vv.sum())

class TextCorpus(Document):
    meta = {'collection': 'traincorpus'}
    OriginalFileName = StringField()
    Topic = StringField()
    OriginalText = StringField()
    LemmaNoStopText = StringField()
    SpacyVector = BinaryField() 
    Top50Words = ListField()   #: top 50 keywords
    
    def GetVector(self):
        return pickle.loads(self.SpacyVector)

    def Similarity(self, pCompareVector):
        v = self.GetVector()
        return (np.dot(v, pCompareVector) / (np.linalg.norm(v) * np.linalg.norm(pCompareVector)))

def RawCompare():

    print("Similarity score inside REQ")
    t1 = nlp(TextCorpus.objects(Topic="REQ")[0].LemmaNoStopText)
    for tc in TextCorpus.objects(Topic="REQ")[1:]:
        tx = nlp(tc.LemmaNoStopText)
        print(t1.similarity(tx), tc.OriginalFileName)

    print("Similarity score inside ARCH")
    t1 = nlp(TextCorpus.objects(Topic="ARCH")[0].LemmaNoStopText)
    for tc in TextCorpus.objects(Topic="ARCH")[1:]:
        tx = nlp(tc.LemmaNoStopText)
        print(t1.similarity(tx), tc.OriginalFileName)

    print("Similarity score inside DESIGN")
    t1 = nlp(TextCorpus.objects(Topic="DESIGN")[0].LemmaNoStopText)
    for tc in TextCorpus.objects(Topic="DESIGN")[1:]:
        tx = nlp(tc.LemmaNoStopText)
        print(t1.similarity(tx), tc.OriginalFileName)
    
    print("Similarity score from REQ in ARCH")
    t1 = nlp(TextCorpus.objects(Topic="REQ")[0].LemmaNoStopText)
    for tc in TextCorpus.objects(Topic="ARCH"):
        tx = nlp(tc.LemmaNoStopText)
        print(t1.similarity(tx), tc.OriginalFileName)

    print("Similarity score from REQ in ARCH")
    t1 = nlp(TextCorpus.objects(Topic="REQ")[0].LemmaNoStopText)
    for tc in TextCorpus.objects(Topic="DESIGN"):
        tx = nlp(tc.LemmaNoStopText)
        print(t1.similarity(tx), tc.OriginalFileName)

nlp = None

def remove_pronoun(text):
    doc = nlp(text.lower())
    result = [token for token in doc if token.lemma_ != '-PRON-']
    return " ".join(result)

def ConnectLoad():

    print("connect db")
    connectdb()

    print("load spacy model")
    global nlp
    
    if nlp == None:
        # changed from lg to md 
        nlp = spacy.load("en_core_web_md")

def GetTotalVec(pTopic):
    print("Getting all vecs of %s" % pTopic)
    varr = [tc.GetVector() for tc in TextCorpus.objects(Topic=pTopic)]
    print("Averaging")
    avgvec = np.array(varr).mean(axis=0)
    return  avgvec

def GetGlobVocs(pTopic=None):
    d = {}
    if pTopic == None:
        alltcs = TextCorpus.objects()
    else:
        alltcs = TextCorpus.objects(Topic=pTopic)
            
    for tc in alltcs:
        for tok, totcnt in tc.Top50Words:
            d.setdefault(tok, [0, 0])
            # add the total count over all words
            d[tok][0] += totcnt
            # count the document occurence
            d[tok][1] += 1

    if pTopic == None:
        fname = os.path.join( r"/Users/felix/FRA_UAS/master/external_pdf", "allwords.csv")
    else:
        fname = os.path.join( r"/Users/felix/FRA_UAS/master/external_pdf", "allwords_%s.csv" % pTopic)
            
    f = open( fname, "w+", encoding='utf8')
    for tok in d:
        f.write("%s;%s;%s\n" % (tok, d[tok][0], d[tok][1]))
    f.close()
    return d

def SaveVecsToOnts(pProj):

    OntologyWordVector.objects(ProjectKey=pProj).delete()

    for (_key, _path) in lTops:

        print("getting glob vec 4 %s" % _key)
        _ontwv = OntologyWordVector()
        _ontwv.OntologyKey = _key
        _ontwv.ProjectKey = pProj
        _ontwv.SaveVector(GetTotalVec(_key))
        print("saving glob vec %s" % _key)
        _ontwv.save()
        print("saved %s" % _key)

if __name__ == "__main__":

    #ConnectLoad()    
    #SaveVecsToOnts("DGDTST")
    #PrepareCorpus()
    #SaveVecs()    
    RawCompare()  
    #print("Similarity score from REQ in ARCH")
    #v1 = TextCorpus.objects(Topic="REQ")[0].GetVector()
    #for tc in TextCorpus.objects(Topic="DESIGN"):
    #    print(tc.Similarity(v1), tc.OriginalFileName)

    if False:
        GetGlobVocs()
        GetGlobVocs("DESIGN")
        GetGlobVocs("REQ")
        GetGlobVocs("ARCH")
        GetGlobVocs("LOG")

    if False:
        vdesign = GetTotalVec("DESIGN")
        vreq = GetTotalVec("REQ")
        varch = GetTotalVec("ARCH")
        vlog = GetTotalVec("LOG")
        
        print("Checking doc")
        fname = os.path.join( r"/Users/felix/FRA_UAS/master/external_pdf/reqs", "stats.csv")
        f = open( fname, "w+", encoding='utf8')
        for tc in TextCorpus.objects():
            f.write("%s;%s;%s;%s;%s;%s\n" % (tc.id, tc.Topic, 
                                         tc.Similarity(vdesign), 
                                         tc.Similarity(vreq), 
                                         tc.Similarity(varch),
                                         tc.Similarity(vlog))) #', tc.OriginalFileName'
        f.close()