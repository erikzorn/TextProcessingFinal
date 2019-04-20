#CSCE 578 Final Project - Ely Yonce & Ely Yonce
#In this project we will be doing cluster analysis on a dataset of fake news aricles
#We want to find common themes in the fake news based on term frequency
###########################################################################
#import needed libraries
import pandas as pd
import numpy as np

#read in the data file into pandas data frame
fakeNewsDF = pd.read_csv('./fake.csv')
#split the text on white space for each one of the articles
splitArticles = fakeNewsDF['text'].str.split(' ')
