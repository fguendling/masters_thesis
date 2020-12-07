# -*- coding: utf-8 -*-
from gensim.test.utils import datapath
from gensim import utils
import spacy

nlp = spacy.load("en_core_web_lg")

class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __iter__(self):
        # this file contains raw text of about 25 text documents. it will not be provided here
        corpus_path = './Corpus/20200916_manually_assembled.corpus'
        for line in open(corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            #old (gensim style)
            # yield utils.simple_preprocess(line)
            #new (spaCy)
            yield nlp(line)

