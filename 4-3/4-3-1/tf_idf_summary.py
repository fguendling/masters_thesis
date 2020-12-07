#from https://towardsdatascience.com/text-summarization-using-tf-idf-e64a0644ace3
import nltk
import math
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def _create_frequency_matrix(sentences):
# "frequency of words in each sentence"
    frequency_matrix = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:15]] = freq_table

    return frequency_matrix

def _create_tf_matrix(freq_matrix):
# TF(t) = (Number of times term t appears in a document) / 
         #(Total number of terms in the document)

    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix

def _create_documents_per_words(freq_matrix):
# "in how many sentences does a word appear?"
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table

def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
# Calculate IDF and generate a matrix
# We’ll find the IDF for each word in a paragraph
# IDF(t) = log_e(Total number of documents / 
          #Number of documents with term t in it)
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix

def _create_tf_idf_matrix(tf_matrix, idf_matrix):
# Calculate TF-IDF and generate a matrix
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix

def _score_sentences(tf_idf_matrix) -> dict:
    """
    score a sentence by its word's TF
    Basic algorithm: adding the TF frequency of every non-stop word in a sentence 
    divided by total no of words in a sentence.
    """

    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    return sentenceValue

def _find_average_score(sentenceValue) -> int:
    """
    Find the average score from the sentence value dictionary. 
    Used for threshold calculation.
    """
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original summary_text
    average = (sumValues / len(sentenceValue))

    return average

def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary

def apply_tf_idf_algo(text, test_text):    
    # text is the whole corpus. tf-idf values are calculated based on this text.
    # test_text is a single text-document, that should be summarized.

    sentences = sent_tokenize(text) # NLTK function
    total_documents = len(sentences)  

    test_sentences = sent_tokenize(test_text)    

    # 2 Create the Frequency matrix of the words in each sentence.
    freq_matrix = _create_frequency_matrix(sentences)
    
    '''
    Term frequency (TF) is how often a word appears in a document, divided by how many words are there in a document.
    '''
    # 3 Calculate TermFrequency and generate a matrix
    tf_matrix = _create_tf_matrix(freq_matrix)
    
    # 4 creating table for documents per words
    count_doc_per_words = _create_documents_per_words(freq_matrix)
    
    '''
    Inverse document frequency (IDF) is how unique or rare a word is.
    '''
    # 5 Calculate IDF and generate a matrix
    idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
    
    # 6 Calculate TF-IDF and generate a matrix
    tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)
    
    # 7 Important Algorithm: score the sentences
    sentence_scores = _score_sentences(tf_idf_matrix)
    
    # 8 Find the threshold
    threshold = _find_average_score(sentence_scores)
    
    # 9 Important Algorithm: Generate the summary
    # USE ONLY SENTENCEES OF TARGET (Test) TEXT HERE
    summary = _generate_summary(test_sentences, sentence_scores, 1.3 * threshold)
    
    # this idea was not good (better to just return the summary)
    """
    if with_chapters==1:
        # assemble with chapters
        with_chapters = {}
                
        # chapters are available via head in lAllSubs
        #example data structure
        example = dict([
            ("1.1", "Ensure that breakbulk scan transactions in the Scan "), 
            ("3.1", "ACA should delete all transaction logs of PID")
            ])       
        
        for tc in convdoc.RootChapter:
            lAllSubs = convdoc.GetAllParasOfTocId(tc.InternalId)                                    
            for (_tcid, lvl, head, paralist) in lAllSubs:  
                chap = convdoc.GetChapterById(_tcid)
                print("new chapter %s %s" % (head, chap.Text))
                with_chapters[head] = "sentence of that chapter"
        
        return with_chapters
    
    else:
    """
    
    print(summary)
    return '<div style="color:red;">(Bisher wird als Corpus nur ein einziges Dokument berücksichtigt)</div>' + summary

# earlier several variables existed here, e.g.
# doc1 = "this is the full text of doc1"
# these have been removed.

test_corpus = doc1

train_corpus_3 = doc2 + doc3 
train_corpus_5 = train_corpus_3 + doc4 + doc5
train_corpus_7 = train_corpus_5 + doc6   +  doc7
train_corpus_9 = train_corpus_7 + doc8 + doc9 
train_corpus_11 = train_corpus_9 + doc10 + doc11
train_corpus_13 = train_corpus_11 + doc12 + doc13
train_corpus_15 = train_corpus_13 + doc14 + doc15

apply_tf_idf_algo(train_corpus_11, test_corpus)
