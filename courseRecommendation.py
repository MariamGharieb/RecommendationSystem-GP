# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 01:02:43 2023

@author:Mariam
"""

# importing important libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI
from pydantic import BaseModel

df = pd.read_csv('F:/Graduation Project/Second_Term/udemy_course_recommendation_system-main/courses.csv')
# prepare data
import nltk
# package provides various tokenizer models for tokenizing text into words or sentences.
# nltk.download('punkt')
# package used to identify the grammatical parts of a sentence, such as nouns, verbs, adjectives.
# nltk.download('averaged_perceptron_tagger')
# package provides a large lexical database of English words and their semantic relationships.
# nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

# base word
lemmatizer = WordNetLemmatizer()
# unuseful words
from nltk.corpus import stopwords

# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
VERB_CODES = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
len(df)
finaldata = df[["course_title", "subject"]]


# nltk.download('omw-1.4')

# function for eliminate unwanted words
def preprocess_sentences(text):
    # all text to lower case
    text = text.lower()
    temp_sent = []
    words = nltk.word_tokenize(text)
    # return list of tuples with word and its type (noun, verb, ..)
    tags = nltk.pos_tag(words)
    # lemmatize and remove stop words
    for i, word in enumerate(words):
        if tags[i][1] in VERB_CODES:
            lemmatized = lemmatizer.lemmatize(word, 'v')
        else:
            lemmatized = lemmatizer.lemmatize(word)
        if lemmatized not in stop_words and lemmatized.isalpha():
            temp_sent.append(lemmatized)

    finalsent = ' '.join(temp_sent)

    return finalsent


finaldata["course_title_processed"] = finaldata["course_title"].apply(preprocess_sentences)

# Compute TF-IDF Scores
# Vectorizing pre-processed using TF-IDF
tfidfvec = TfidfVectorizer()
# compute tf-idf scores
tfidf_course = tfidfvec.fit_transform((finaldata["course_title_processed"]))

# Finding cosine similarity between vectors
from sklearn.metrics.pairwise import cosine_similarity

sig = cosine_similarity(tfidf_course, tfidf_course)
# Reverse mapping of indices and movie titles
indices = pd.Series(finaldata.index, index=finaldata['course_title']).drop_duplicates()


# Recommendation function

def give_recommnedation(title, sig=sig):
    # Get the index corresponding to original_title
    idx = indices[title]

    # Get the pairwsie similarity scores
    sig_scores = list(enumerate(sig[idx]))

    # Sort the courses
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar courses
    sig_scores = sig_scores[1:11]

    # course indices
    course_indices = [i[0] for i in sig_scores]

    # Top 10 most similar courses
    return finaldata['course_title'].iloc[course_indices]


# create api
app = FastAPI()


class ModelInput(BaseModel):
    course_title: str


# loading saved model
@app.get("/")
def read_root(course_title: str):
    return {"recommended_courses": give_recommnedation(course_title).tolist()}


# @app.get("/recommend")
# def recommend_course(course_title: str):
#     recommended_courses = give_recommnedation(course_title).tolist()
#
#     return {"recommended": recommended_courses}
