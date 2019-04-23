#CSCE 578 Final Project - Ely Yonce & Erik Zorn
#In this project we will be doing cluster analysis on a dataset of fake news aricles
#We want to find common themes in the fake news based on term frequency
###########################################################################
#Get term frequency from each one of the articles
#import needed libraries
import pandas as pd
import numpy as np

#read in the data file into pandas dataframe
fakeNewsDF = pd.read_csv('./fake.csv')
#get rid of punctuations
fakeNewsDF.replace('-','')
#split the text on white space for each one of the articles
splitArticles = fakeNewsDF['text'].str.lower().str.split(' ')
#init dictionary to hold tf vals
termsDict = {}
#loop through all of the articles
index = 0
for article in splitArticles:
    articleSeries = pd.Series(article)
    termsDict[index] = articleSeries.value_counts()
    index += 1
print(termsDict)