#CSCE 578 Final Project - Ely Yonce & Erik Zorn
#In this project we will be doing cluster analysis on a dataset of fake news aricles
#We want to find common themes in the fake news based on term frequency
###########################################################################
#Get term frequency from each one of the articles
#import needed libraries
import pandas as pd
import numpy as np
import pprint, heapq
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt # Python defacto plotting library


#read in the data file into pandas dataframe
fakeNewsDF = pd.read_csv('./fake.csv', nrows=20)
#list of stopwords
stopwords = ["a", "about", " ", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount", "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as", "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]
#get rid of punctuations in the articles
fakeNewsDF["clean_text"] = fakeNewsDF['text'].str.replace('[^\w\s]','')
#split the text on white space for each one of the articles
splitArticles = fakeNewsDF['clean_text'].str.lower().str.split(' ')
#init dictionary to hold tf vals
termsDictNums = {}
termDict = {}
termlist = []
#loop through all of the articles
index = 0
termListInd = 0
for article in splitArticles:
    articleSeries = pd.Series(article)
    valCount = articleSeries.value_counts()
    termsDictNums[index] = valCount
    for item in valCount.keys():
        if item not in termlist and item not in stopwords:
            #print(item)
            termlist.append(item)
            termDict[item]  = termListInd
            termListInd += 1
    index += 1
rowObj = []
fullMat = []
tf_matrix = [[0 for n in range(termListInd)] for m in range(index)]


'''
termDictNums = {0: {term: count, term1: count1, ... , termN: countM}}
OR {<articleNum> : { <term> : <count> }}

			0		1		2				N
termList = ["dog", "cat", "mouse", ... , "fish"]
termDict = {"dog" : 0, "cat": 1, "mouse": 2, ... , "fish": N}

'''
print("index = ", index, " (aka the number of articles)")
print("Keys in termDictNums: ", len(termsDictNums.keys()))

print("\nKeys in termDict: ", len(termDict.keys()))
print("termList length: ", len(termlist))
print("termListInd = ", termListInd)

print("\ntf_matrix row count: ", len(tf_matrix) )
print("tf_matrix column count: ", len(tf_matrix[0]) )


for row, words in termsDictNums.items():
    for word, count in words.items():
        if (word not in stopwords):
            #print(word)
            tf_matrix[row][termDict[word]] = count

'''							words 	 					if you have a word, use termDict[<word>] 
tf_matrix = [	 0  1  2  3  4  5  6  7  8  9  10 11	to get the index it belongs at			

		0		[0, 0, 1, 2, 1, 5, 0, 0, 1, 1, 3, 1],
	a	1		[0, 0, 0, 1, 0, 1, 2, 1, 5, 1, 3, 1],
	r	2		[1, 5, 0, 0, 1, 1, 0, 0, 1, 2, 3, 1],
	t	3		[0, 0, 1, 2, 1, 5, 0, 0, 1, 1, 3, 1],
	i	4		[0, 0, 1, 2, 1, 5, 0, 0, 1, 1, 3, 1],
	c	5		[0, 5, 0, 0, 1, 0, 1, 2, 1, 1, 3, 1],
	l	6		[0, 0, 1, 2, 1, 5, 0, 0, 1, 1, 3, 1],
	e	7		[0, 1, 1, 3, 1 0, 1, 2, 1, 5, 0, 0,]
			]	
'''

top_3_word_list = []
for row in tf_matrix:
	# input_list = [1,2,3,4,5]
	number_of_elements = 3
	max_3 = heapq.nlargest(number_of_elements, row)
	top_3_word_list.append(termlist[row.index(max_3[0])] + ", " + termlist[row.index(max_3[1])] + ", " + termlist[row.index(max_3[2])])


# initialize pca variable with 3 components. And fit tf-idf matrix using PCA
pca = PCA(n_components=3)
X = pca.fit_transform(tf_matrix)

print(X)


if (len(X) > 20):
	X = X[:20,:]
# Print the variances produced
print(pca.explained_variance_ratio_)



# initialize kmeans clustering using 5 clusters
# 5 was chosen after testing with different clusters and determining that 5 clusters
# the results most effectively for the chosen dataset
kmeans = KMeans(n_clusters=3)

# fit the pca data using kmeans with 5 clusters
X_clustered = kmeans.fit_predict(X)

# define color codes
LABEL_COLOR_MAP = {
	0 : 'red',
	1 : 'blue',
	2 : 'green',
	3 : 'purple',
	4 : 'orange'
}	
label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]

# define the size of the visualization
plt.figure(figsize = (10,10))

# iterate through file_list and print file_name and top 3 tfidf weights of that file
# at the coordinates that the PCA tranformation produced
# for i,type in enumerate(file_list):
for i in range(index):
	if (i >= 19):
		break
	x = X[:,0][i]
	y = X[:,2][i]
	plt.scatter(x, y, color=label_color[i], alpha=0.5)
	plt.text(x + .0005, y, "Doc: " + str(i) + " --- Top 3 Words: " + top_3_word_list[i], fontsize=8)#+ " --- Top 3 Words: " + top_3_word_list[i], fontsize=8)

# create graph labels and title and then show the scatter plot
plt.xlabel('Principal Component 1', fontsize = 15)
plt.ylabel('Principal Component 2', fontsize = 15)
plt.title('2 component PCA', fontsize = 20)
plt.show()












