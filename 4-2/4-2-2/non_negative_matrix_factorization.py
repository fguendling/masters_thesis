# -*- coding: utf-8 -*-

# https://sanjayasubedi.com.np/nlp/nlp-with-python-topic-modeling/
# der Plan:
# Topics aus Trainingskorpus extrahieren und visualisieren. 
# Dann das Topic von Testdokumenten schaetzen, mit dem Modell.
# in scrapeweb/scripts/main_testwordvec.py werden pdfs zur Verarbeitung gelesen:
# speichert u.a average vektoren in dgdai.ontwordvectors und
# die Texte aus den PDFs in collection traincorpus
# Ausführung dieses Scripts über spyder ist am einfachsten (F5) - nur möglich wenn eine Datenbankverbindung hergestellt werden konnte.

from sklearn.datasets import load_files
import pandas as pd
import spacy
import sys
# sys.path.append('../')  # geloest durch Anpassung des pythonpath ueber den python path manager in spyder
from prep.connectdb import *
from scripts.main_testwordvec1 import TextCorpus
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

pdf_texts = []
try:
    # already preprocessed documents are available in dgdai.traincorpus.LemmaNoStopText
    connectdb()
    print('connected to db')
    
    _LemmaNoStopText = TextCorpus.objects()
    for texts in _LemmaNoStopText:
        pdf_texts.append(texts.LemmaNoStopText)
except:
    print('err')

my_df = pd.DataFrame({'texts': pdf_texts})

# we are working only with nouns because topics are made up of nouns only
def only_nouns(texts):
    output = []
    for doc in nlp.pipe(texts):
        noun_text = " ".join(token.lemma_ for token in doc if token.pos_ == 'NOUN')
        output.append(noun_text)
    return output

my_df['texts'] = only_nouns(my_df['texts'])

# number of topics to extract
n_topics = 4

#sklearn functions
vec = TfidfVectorizer(max_features=5000, stop_words="english", max_df=0.95, min_df=2)
features = vec.fit_transform(my_df.texts)

cls = NMF(n_components=n_topics, random_state=0)
cls.fit(features)
#Our model is now trained and is ready to be used.

feature_names = vec.get_feature_names()

# number of most influencing words to display per topic
n_top_words = 15

topics_wo_score = []
for i, topic_vec in enumerate(cls.components_):
    # topic_vec.argsort() produces a new array
    # in which word_index with the least score is the
    # first array element and word_index with highest
    # score is the last array element. Then using a
    # fancy indexing [-1: -n_top_words-1:-1], we are
    # slicing the array from its end in such a way that
    # top `n_top_words` word_index with highest scores
    # are returned in descending order
    twords = ''
    for fid in topic_vec.argsort()[-1:-n_top_words-1:-1]:
        #print(feature_names[fid], end=' ')
        twords = twords + ' ' + feature_names[fid]
    topics_wo_score.append((i, twords))

print(topics_wo_score)
    
# Visualisation of topics from https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/
# funktioniert mit 4 topics

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'
cloud = WordCloud(background_color='white',
                  width=5000,
                  height=3600,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    #topic_words = dict(topics_wo_score[i][1])
    #cloud.generate_from_frequencies(topic_words, max_font_size=300)
    cloud.generate_from_text(topics_wo_score[i][1]) #, max_font_size=300)
    #plt.figure( figsize=(20,20), facecolor='k')
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')

plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()


# sample prediction
test_chapters = [
    "Playstation network was down so many people were angry",
    "Germany scored 7 goals against Brazil in worldcup semi-finals"
]
    
my_predictions = cls.transform(vec.transform(test_chapters)).argsort(axis=1)[:,-1]
print(my_predictions)

# real evaluation (Daten werden nicht bereitgestellt)
#test_df = pd.read_csv('manually_assembled_test_dataset.csv')

#for index, row in test_df:    
#    print('prediction for ', index)
#    print(cls.transform(vec.transform(row['text'])).argsort(axis=1)[:,-1])

# es gab auch eine Variante in der verschiedene Texte direkt hardcodiert hier in der Liste gespeichert wurden. Die Texte wurden entfernt.
test_list = []

my_predictions = cls.transform(vec.transform(test_list)).argsort(axis=1)[:,-1]
print(my_predictions)

   