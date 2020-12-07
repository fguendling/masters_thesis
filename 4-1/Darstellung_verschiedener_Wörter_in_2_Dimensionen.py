# a.k.a training_our_own_word_vectors.py
# this script is also used to calculate how many words are covered by the spaCy model
import numpy as np
import spacy
import gensim.models
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec

from MyCorpus import *

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.decomposition import PCA

sentences = MyCorpus()
model = gensim.models.Word2Vec(sentences=sentences)
#ignore words that appear less times than min_count (parameter in Word2Vec, unused)
#vector will have dimension of size parameter

#model = gensim.models.Word2Vec(min_count=1)
model.build_vocab(sentences)
model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)

nlp = spacy.load("en_core_web_md")
#tokens = (next(x for x in sentences))
#print((next(x for x in sentences)))

total_number=0
oov_number=0
for doc in sentences:
    for w in doc:
        total_number += 1
        if(w.is_oov==True):
            oov_number += 1

print('total number of words_ ', total_number, 'oov words: ', oov_number, 'percentage: ', (oov_number/total_number)*100)

"""
for word in model.wv.vocab:
    print(word)
"""

try:
    vec_dhl = model.wv['test']
except:
    print('oov!')
    
# from https://web.stanford.edu/class/cs224n/materials/Gensim%20word%20vector%20visualization.html
def display_pca_scatterplot(model, words=None, sample=0):
    if words == None:
        if sample > 0:
            #words = np.random.choice(list(model.vocab.keys()), sample)
            return
        else:
            words = [ word for word in model.wv.vocab ]
    """    
    word_vectors_list = []
    
    for w in words:
        try:
            print(w)
            word_vectors_list.append(model[w])
            print(model[w])
        except: 
            continue
    """
    
    word_vectors = np.array([model[w] for w in words])
    
    twodim = PCA().fit_transform(word_vectors)[:,:2]
    
    plt.figure(figsize=(6,6))
    plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
    for word, (x,y) in zip(words, twodim):
        plt.text(x+0.05, y+0.05, word)
        
display_pca_scatterplot(model, 
                        [ 'dhl', 'ocnt', 
                         'system', 'package', 'shipment', 'order',
                         'delivered', 'returned', 'gref'
                         ])