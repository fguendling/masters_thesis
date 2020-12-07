# die Erzeugung der Heatmaps erfolgt über den Aufruf eines webbasierten Frontends, unter anderem auch wegen der Datenbankverbindung, die hier genutzt wird. Dieses Skript ist daher nicht ohne Weiteres ausführbar, es verdeutlicht aber wie die Heatmaps erstellt wurden.

# -*- coding: utf-8 -*-

import os, datetime, sys
from flask import Blueprint, Flask, jsonify, flash, request, redirect, url_for, Response, json
from flask import render_template, session
from flask import current_app as app
from util.globals import _log, _getUploadFileName
from util.globaldecorators import *
from datamodel.converteddocument import ConvertedDocument
from datamodel.convertedparagraph import ConvertedParagraph
import pandas as pd
import spacy
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib import rcParams
from numpy import random
rcParams.update({'figure.autolayout': True})

text_similarity = Blueprint('similarity', __name__,
                        template_folder='templates')

# es wird ein vortrainiertes spaCy-Modell verwendet. 
# Die similarity() Funktion die unten verwendet wird,
# basiert ebenfalls auf diesem Modell.
nlp = spacy.load("en_core_web_lg")
all_stopwords = nlp.Defaults.stop_words

@text_similarity.route('/similarity_test_drive', methods=['GET', 'POST'])
@project_required
def similarity_test_drive(sProj):
    text_bucket = {}
    text_bucket_cleaned = {}
    
    alldocs = ConvertedDocument.GetAllObjects(sProj)
    
    # get text content of documents and assign all document texts to variables here
    for convdoc in alldocs:
        text_bucket[convdoc.UploadFileId] = ""               
        _rootChapters = convdoc.GetRootChapters()
        print('working on ', convdoc.FileName)
        
        for i in range(len(_rootChapters)):
            _tocid = _rootChapters[i].InternalId    
            _AllSubs = convdoc.GetSubChapterIdsOfChapter(_tocid)
            startchapt = convdoc.GetChapterById(_tocid)
            lAllSubs = [[int(_tocid), 0, startchapt.HeadNumber],]
        
            if _AllSubs:
                lAllSubs.extend( [[_tcobj.InternalId, _lvl, head] for (_tcobj, _lvl, head) in _AllSubs])
        
            for (_tcid, lvl, head) in lAllSubs:
                for p in ConvertedParagraph.objects( ProjectKey = sProj, UploadFileId = str(convdoc.UploadFileId), GlobalChapterId = str(_tcid)):
                    #es gäbe die Möglichkeiten Listen gesondert zu behandeln, wird hier nicht gemacht
                    #if p.IsList:
                    #AnnotatedTextV2 gibt es nur bei den neueren Dokumenten
                    for ann in p.AnnotatedTextV2:
                        if ann.OrgText == '.':
                            text_bucket[convdoc.UploadFileId] = text_bucket[convdoc.UploadFileId].rstrip()
                            text_bucket[convdoc.UploadFileId] += ann.OrgText + ' ' 
                        else:                                
                            text_bucket[convdoc.UploadFileId] += ann.OrgText + ' ' 
        
        # remove stop words, store cleaned texts in text_bucket_cleaned
        doc = nlp(text_bucket[convdoc.UploadFileId])
        text_bucket_cleaned[convdoc.UploadFileId] = ""
        for token in doc:
            if (token.is_stop == False):    
                text_bucket_cleaned[convdoc.UploadFileId] += ' ' + token.text
    
    heatmap_data = {}
    heatmap_index = []
    
    # either text_bucket.items() or text_bucket_cleaned.items() can be used
    for key, val in text_bucket_cleaned.items():
        if key == '5ee0eb293443bfa0c81acaf0':
            # skip nonsense document
            continue
        else:
            heatmap_data[key[5:]] = []
            heatmap_index.append(key[5:])
            
            for other_key, other_val in text_bucket_cleaned.items():
                # skip nonsense document again
                if other_key != '5ee0eb293443bfa0c81acaf0':
                    # append the score to that list
                    doc = nlp(val)                    
                    other_doc = nlp(other_val)
        
                    heatmap_data[key[5:]].append(
                        doc.similarity(other_doc)
                        ) 

    # create visuals
    df = pd.DataFrame( heatmap_data
            , index = heatmap_index
        )    
    
    sn.heatmap(df, xticklabels=1, yticklabels=1)   # annot=true führt dazu, dass auch die Zahlen angezeigt werden
    plt.savefig('heatmap.png', dpi=2000)   
    
    return render_template("similarity.html")
                    
"""
df = pd.DataFrame(columns = ['text', 'has_vector', 'vector_norm', 'is_oov'])

for token in doc1:
    
    df = df.append({'text': token.text,
                    'has_vector': token.has_vector,
                    'vector_norm': token.vector_norm,
                    'is_oov': token.is_oov}, ignore_index=True)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  
    print(df.sort_values('vector_norm'))   
"""  
