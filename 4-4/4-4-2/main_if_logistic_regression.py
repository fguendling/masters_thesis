# -*- coding: utf-8 -*-
from re import I
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from sklearn import metrics
from nltk import sent_tokenize

df = pd.read_csv("/Users/felix/FRA_UAS/master/DHL_Document_Gap_Detector/scrapeweb/scripts/_20202310_Pretraining_test.csv", sep=',', error_bad_lines=False)

texts = df.text.tolist()
labels = df.label.tolist()

X_train, X_test, y_train, y_test = train_test_split( texts, labels, test_size=0.25, random_state=42)

#print(X_test)

# CountVectorizer includes preprocessing, tokenizing, filtering of stopwords
count_vect = CountVectorizer(ngram_range=(1, 2))
# instead of "occurences" tf idf can be used: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

# Extracting features from text files  ("return term-document-matrix")
X_train_counts = count_vect.fit_transform(X_train) 

# logistic regression, from https://www.snorkel.org/get-started/ chapter 5
lr_clf = LogisticRegression(solver="lbfgs").fit(X=X_train_counts, y=y_train)

# multinomial naive bayes as alternative to logistic regression
# mnv_clf = MultinomialNB().fit(X=X_train_counts, y=y_train)

# apply transformations to test set and use the model for predictions
X_test_counts = count_vect.transform(X_test)
predicted = lr_clf.predict(X_test_counts)

predict_train = lr_clf.predict(X_train_counts)

# simple evaluation of accuracy
# print(np.mean(predicted == y_test))

# detailed evaluation
# print(metrics.classification_report(y_train, predict_train))

# manual evaluation (would be based on one example document that was not part of training or testing)

doc = "example sentence."
eval_sentences = sent_tokenize(doc)

X_eval_counts = count_vect.transform(eval_sentences)

predict_eval = lr_clf.predict(X_eval_counts)

eval_df = pd.DataFrame()

eval_df["sentence"] = eval_sentences

filtered = [a for a in predict_eval if a != 0]

eval_df["prediction"] = predict_eval



