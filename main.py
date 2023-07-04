# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 01:02:43 2023

@author:Mariam
"""

# importing important libraries
from fastapi import FastAPI
from pydantic import BaseModel
import courseRecommendation
import routing_question

# create api
app = FastAPI()


class ModelInput(BaseModel):
    course_title: str
    Tag : str
    post_id : int

# loading saved model
@app.get("/recommendCourse")
def getCourses(course_title: str):
    return {"recommended_courses": courseRecommendation.give_recommnedation(course_title).tolist()}


@app.get("/recommendByTag")
def getInstructorByTag(Tag: str):
    return {"Instructor ": routing_question.RecommendExpertByTag(Tag)}


@app.get("/recommendByPost")
def recommend_Instructor(post_id: int):
    recommended_instructors = routing_question.recommend_instructors(post_id)
    return {"recommend_Instructor": recommended_instructors}
