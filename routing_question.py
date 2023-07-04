# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 01:02:43 2023

@author:Mariam
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI
from pydantic import BaseModel

users = pd.read_csv('F:/Graduation Project/Second_Term/dataset/users.csv')
posts = pd.read_csv('F:/Graduation Project/Second_Term/dataset/postsTag.csv')

"""# Preparing data"""

postsData = posts[["Title", "topic", "Body"]]
postsData = postsData.dropna()
postsData.head(10)

usersData = users[["Username", "topic", "rate", "Bio"]]
usersData.head(10)

import nltk
import re

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
# pattern = re.compile(r'\b\d+\b')
VERB_CODES = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}


def preprocess_sentences(text):
    # all text to lower case
    # text.SentimentText=text.SentimentText.astype(str)
    text = text.lower()

    temp_sent = []
    words = nltk.word_tokenize(text)
    # return list of tuples with word and its type (noun, verb, ..)
    tags = nltk.pos_tag(words)
    # lemmatized = [token for token in words if not pattern.match(token)]
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


"""# Recommend experts based on tags"""

# recommend based on tags in posts and topics in users
from sklearn.metrics.pairwise import cosine_similarity

users_data = users
posts_data = posts
merged_data = pd.merge(users_data, posts_data, on='topic')
# print(merged_data[merged_data['Username'] == 'samy'])
# Create TF-IDF vectorizer
tfidf_vec = TfidfVectorizer()

# Fit and transform post titles to create TF-IDF matrix
tfidf_matrix = tfidf_vec.fit_transform(merged_data['topic'])

# Calculate pairwise cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# Function to get recommended users for a given user
def RecommendExpertByTag(tag, top_n=5):  # tag that is provided by the student
    # Get the index of the user
    user_index = merged_data[merged_data['topic'] == tag].index
    # print(user_index)
    recommended_users = merged_data.loc[user_index, 'Username'].tolist()
    unique_recommended_users = list(set(recommended_users))
    return unique_recommended_users


# Example usage
tag = "java"
recommended_users = RecommendExpertByTag(tag)
print("Recommended experts for question by post tag", recommended_users)

"""# Recommend experts based on Bio of instructor and the body of the post"""

# Concatenate the instructor bio and post description into a single column
postsData["processedBody"] = postsData["Body"].apply(preprocess_sentences)
usersData["processedBio"] = usersData["Bio"].apply(preprocess_sentences)
merged_data = pd.merge(postsData, usersData, on='topic')
data = merged_data[['processedBody', 'processedBio']]
# print(data)
# Concatenate the instructor and post data
# data = pd.concat([instructors[['Username', 'content']], posts[['content']]])
# print(data)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the content into TF-IDF matrix for postsData
tfidf_matrix1 = vectorizer.fit_transform(postsData["processedBody"])

# Fit and transform the content into TF-IDF matrix for usersData
tfidf_matrix2 = vectorizer.transform(usersData["processedBio"])

# Calculate pairwise cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix1, tfidf_matrix2)
max_sim_index = cosine_sim.argmax()
# Convert the index to row and column indices
row_index = max_sim_index // cosine_sim.shape[1]
col_index = max_sim_index % cosine_sim.shape[1]

# Get the corresponding values from the datasets
value2 = postsData.loc[col_index, 'Body']


def recommend_instructors(post_id, top_n=1):
    # Get the index of the post
    post_index = post_id  # Adjust the index to account for instructors

    # Get the cosine similarity scores for the post
    sim_scores = list(enumerate(cosine_sim[post_index]))

    # Sort the instructors based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top similar instructors
    top_instructors = [i for i, _ in sim_scores[1:top_n + 1]]  # Exclude the same post

    # Return the recommended instructors
    recommended_instructors = users.loc[top_instructors, 'Username'].tolist()
    return recommended_instructors


# Example usage
post_id = 1
# print(posts.loc[1])
recommended_instructors = recommend_instructors(post_id)
# print(posts.loc[post_id])
print(recommended_instructors)

# create api
app = FastAPI()


class ModelInput(BaseModel):
    Tag: str
    post_id: int


# @app.get("/")
# def read_root(Tag: str):
#     return {"Instructor ": RecommendExpertByTag(Tag)}

# post_id = 1
# # print(posts.loc[1])
# recommended_instructors = recommend_instructors(post_id)
# # print(posts.loc[post_id])
# print("Recommended instructor by post id = ", recommended_instructors)

"""
@app.get("/recommendByTag")
def getInstructorByTag(Tag: str):
    return {"Instructor ": RecommendExpertByTag(Tag)}


@app.get("/recommendByPost")
def recommend_Instructor(post_id: int):
    recommended_instructors = recommend_instructors(post_id)
    return {"recommend_Instructor": recommended_instructors}
"""
# http://127.0.0.1:8000/?course_title=Forex Trading Secrets of the Pros With Amazon's AWS
