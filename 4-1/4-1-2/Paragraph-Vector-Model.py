# die Erzeugung des Streudiagramms erfolgt über den Aufruf eines webbasierten Frontends, unter anderem auch wegen der Datenbankverbindung, die hier genutzt wird. Dieses Skript ist daher nicht ohne Weiteres ausführbar, es verdeutlicht aber wie das Streudiagramm erstellt wurde.

# from https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html
# and https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b 
import gensim
import os, datetime, sys
from flask import Blueprint, Flask, jsonify, flash, request, redirect, url_for, Response, json
from flask import render_template, session
from flask import current_app as app
from util.globals import _log, _getUploadFileName
from util.globaldecorators import *
from datamodel.converteddocument import ConvertedDocument
from datamodel.convertedparagraph import ConvertedParagraph
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib
matplotlib.use('Agg')

document_embeddings = Blueprint('similarity1', __name__,
                        template_folder='templates')

# function that reads and preprocesses each document
@document_embeddings.route('/document_embeddings', methods=['GET', 'POST'])
@project_required
# reduce dimensions with t-sne and visualise all documents
def display_t_sne_plot(sProj):
    
    raw_vals, categories = train_model(sProj)
    categories = [i[-7:].strip() for i in categories]
    vals = np.array(raw_vals)
    # this is how the output of train_model look(ed, previously)s 
    """
    vals = np.array([[-6.0056877,  -3.3252418,  -3.2914245,  -4.4289308,  -9.126206,
                  3.9861972,  -5.5491056,   1.0337074,   2.6021235,   0.09615537,
                  2.0758114,  -1.3636483, -0.19596846,  5.751454,   -0.79410136,
                  -0.09732328,  0.06497823, -2.36947,  -5.1114163,   3.7325873,
                  1.0327809,  -9.841449,   -2.8236568,   0.5371798, -7.5671687,
                  3.2847292,  -1.5054959,  -4.314939,   -0.21686949,  3.728024,
                  -5.5056667,   3.5752287 ,  3.7780712 ,  1.3834304,  -1.1310588,
                  1.8249911,  1.1088371,  -5.2905045,  -3.1455712,   4.1983433 ,
                  -3.0523112,   0.8893321,  1.013258,    1.9022989,  -0.37644878 ,
                  1.4101107,  11.993529,   -0.57231873,  7.95054 ,   -4.4593906 ],
                  
                  [-5.9916797,  -3.386788,   -3.363081 ,  -4.250991 ,  -8.937502,
                   3.858905, -5.608106 ,   1.0081686,   2.7668,      0.1006052,
                   2.086436,   -1.5383748, -0.19604069,  5.8239813,  -0.8375895,
                   -0.02436497,  0.05596003, -2.5574841, -5.235597,    3.7774284,
                   1.0253023 , -9.774088,   -2.9357855,  0.6036386, -7.4122148 ,
                   3.0679445,  -1.4342229,  -4.290678,   -0.2736783,   3.7379482,
                    -5.5302377,   3.5458927,   3.7059715,   1.1903714,  -1.0795838 ,
                    1.732314,  1.1018312,  -5.3435397,  -3.2276917,   4.1050053,
                    -3.3019807 ,  0.9254986,  0.85684216 , 1.9492788,  -0.13762434,
                    1.3431845,  12.100708,   -0.4694627,  7.73416,    -4.4255986]])
    """ 

    # perplexity was 40 earlier
    tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=300)
    tsne_results = tsne.fit_transform(vals)
    
    print(tsne_results)

    df = pd.DataFrame()

    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]
    df['label'] = categories

    """
    the df looks like this at this positon:
    tsne-2d-one  tsne-2d-two
    0     4.041599    25.535725
    1   -54.693726    84.739906
    2    14.921980  -108.493668
    3  -135.677734  -181.068268
    4   -88.771080    54.175648
    """
    print(df)
    try:
        sns.scatterplot(
            data=df,
            x="tsne-2d-one", y="tsne-2d-two",
            # should work now with exactly 32 docs: palette=sns.color_palette("husl", 32),
            # if there are less than 10 categories: 
            #palette=sns.color_palette('husl', n_colors=7), # es sind 7  
            hue="label",
            legend="brief",
            )
        plt.savefig('scatterplot.png', dpi=3000)  
    except:
        print('err related to number of categories')

    return render_template("similarity1.html")

def train_model(sProj):
    # create train and test corpus 
    print('read_corpus:', read_corpus(sProj))
    raw_train_corpus, filenames = read_corpus(sProj)
    train_corpus = list(raw_train_corpus)

    # print('train_corpus:', train_corpus)
    #as of now the train and test is not different - for testing purpose
    test_corpus = list(read_corpus(sProj, tokens_only=True))
    print('test_corpus:', len(test_corpus))    
    
    #Training of the Model
    model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    # infer the vectors for every document 
    the_vectors = []
    for idx in range(len(train_corpus)):        
        the_vectors.append(model.infer_vector(test_corpus[idx]))

    return the_vectors, filenames
    
def read_corpus(sProj, tokens_only=False):
    text_bucket = {}
    text_bucket_cleaned = {}
    
    alldocs = ConvertedDocument.GetAllObjects(sProj)
    
    # get text content of documents and assign all document texts to variables here
    for convdoc in alldocs:
        #text_bucket[convdoc.FileName] = ""   
        text_bucket[convdoc.FileName + '     ' + convdoc.TemplateKey] = ""           
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
                    #es gäbe die Möglichkeiten Listen (Aufzählungen in MS Word) gesondert zu behandeln, wird hier nicht gemacht
                    #AnnotatedTextV2 gibt es nur bei den neueren Dokumenten
                    for ann in p.AnnotatedTextV2:
                        if ann.OrgText == '.':
                            text_bucket[convdoc.FileName + '     ' + convdoc.TemplateKey] = text_bucket[convdoc.FileName + '     ' + convdoc.TemplateKey].rstrip()
                            text_bucket[convdoc.FileName + '     ' + convdoc.TemplateKey] += ann.OrgText + ' ' 
                        else:                                
                            text_bucket[convdoc.FileName + '     ' + convdoc.TemplateKey] += ann.OrgText + ' ' 
        
        # remove stop words, store cleaned texts in text_bucket_cleaned
        """
        doc = nlp(text_bucket[convdoc.UploadFileId])
        text_bucket_cleaned[convdoc.UploadFileId] = ""
        for token in doc:
            if (token.is_stop == False):    
                text_bucket_cleaned[convdoc.UploadFileId] += ' ' + token.text
        """

    tagged_document_list = []
    FileName_list = text_bucket.keys() 
    token_list = []
    #special treatment from gensim tutorial
    for i, line in enumerate(text_bucket.values()):
        #FileNameKey = FileName_list[i]
        tokens = gensim.utils.simple_preprocess(line)

        if tokens_only:
            token_list.append(tokens)
        else:
            # For training data, add tags
            tagged_document_list.append(gensim.models.doc2vec.TaggedDocument(tokens, [i]))
        #    , FileNameKey
    
    #iterator wurde entfernt und durch liste ersetzt    
    if tokens_only:
        return token_list
    else:
        return tagged_document_list, FileName_list

