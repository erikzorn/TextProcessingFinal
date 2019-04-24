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


num_rows = input("How many articles would you like to read. (Min 15 and Max 12999 articles)\n")
try:
	num_rows = int(num_rows)
	if (num_rows > 12999 or num_rows < 15):
		print("Defaulting to max. Articles = 12999...")
		num_rows = 12999
except:
	num_rows = 25
	print("Invalid input: Defaulting to 25 articles...")

#read in the data file into pandas dataframe
fakeNewsDF = pd.read_csv('./fake.csv', nrows=num_rows)
#list of stopwords
stopwords = ["a", "about", " ", "\n", "i", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount", "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as", "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]
#get rid of punctuations in the articles
fakeNewsDF["clean_text"] = fakeNewsDF['text'].str.replace('[^\w\s]','')
#split the text on white space for each one of the articles
splitArticles = fakeNewsDF['clean_text'].str.lower().str.split()
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
            termlist.append(item)
            termDict[item]  = termListInd
            termListInd += 1
    index += 1
rowObj = []
fullMat = []
tf_matrix = [[0 for n in range(termListInd)] for m in range(index)]


'''
termDictNums = {0: {term: count, term1: count1, ... , termN: countM}, 1: {term: count, term1: count1, ... , termN: countM}}
OR {<articleNum> : { <term> : <count> }}

			0		1		2				N
termList = ["dog", "cat", "mouse", ... , "fish"]
termDict = {"dog" : 0, "cat": 1, "mouse": 2, ... , "fish": N}

'''
# print(termsDictNums[12])

print("index = ", index, " (aka the number of articles)")
print("Keys in termDictNums: ", len(termsDictNums.keys()))

print("\nKeys in termDict: ", len(termDict.keys()))
print("termList length: ", len(termlist))
print("termListInd = ", termListInd)

print("\ntf_matrix row count: ", len(tf_matrix) )
print("tf_matrix column count: ", len(tf_matrix[0]) )


# this generates the matrix
# each tf weight is divided by the max tf weight in that row. This is inteded to slightly
# normalize the weights in that case that there is a document way bigger than other docs
# so if the mac tf weight is for "cat" appearing 40 times and then the word "dog" appears
# 4 times... the new weight for "cat" will be 1 and the weight for "dog" will be .1

# ^^^ im pretty sure this will help when some docs have a weight for words that is huge becuase a lot of
# words are in it because it will make there be a max of wieght 1 for any term. 

# if we decide this is pointless, then we will remove the two lines indicsated in the for loop below
for row, words in termsDictNums.items():
    for word, count in words.items():
        if (word not in stopwords):
            #print(word)
            tf_matrix[row][termDict[word]] = count
    # comment these next to lines out if we want to remove normalizations
    divisor = float(max(tf_matrix[row]))
    if (divisor != 0):
    	tf_matrix[row][:] = [x / divisor for x in tf_matrix[row]]

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
rowsPrinted = 1

# code below is used for calculating the 3 highest term frequencies
# per document. These 3 terms are paired with the data points in the graph
# to indicate what the article talks about the most
for row in tf_matrix:
	try: 
		startAt = -1
		number_of_elements = 3
		max_3 = heapq.nlargest(number_of_elements, row)
		top3string = ''
		for i in range(len(max_3)):
			top3string += termlist[row.index(max_3[i], startAt+1)]
			top3string += ", "
			startAt = row.index(max_3[i], startAt+1)
		startAt = -1
		rowsPrinted +=1
		top_3_word_list.append(top3string)

	except:
		startAt = -1
		continue



# initialize pca variable with 3 components. And fit tf-idf matrix using PCA
pca = PCA(n_components=3)
X = pca.fit_transform(tf_matrix)


# Below is a within sum squares plot
# this will be shown to the user and they will choose the elbow
# to indicate the number of cluster they would like to choose
sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    sum_of_squared_distances.append(km.inertia_)

plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k\n*Close Plot to Input # of Clusters in Terminal*')
plt.show()

clusters = input("How man clusters would you like to use (Max 5)? Typically people choose the x-value where the elbow is in the graph.\n")
try:
	clusters = int(clusters)
	if (clusters < 1 and clusters > 5):
		print("Invalid input: defaulting to 4 clusters...")
		clusters = 4
except:
	clusters = 4
	print("Invalid input: defaulting to 4 clusters...")

print(X)


# Print the variances produced
print(pca.explained_variance_ratio_)

# initialize kmeans clustering using 5 clusters
# 5 was chosen after testing with different clusters and determining that 5 clusters
# the results most effectively for the chosen dataset
kmeans = KMeans(n_clusters=clusters)

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
print("LENGTH 3 word list: ", len(top_3_word_list))
for i in range(index):
	try:
		x = X[:,0][i]
		y = X[:,2][i]
		plt.text(x + .0005, y, "Doc: " + str(i) + " --- Top Words: " + top_3_word_list[i], fontsize=8)#+ " --- Top 3 Words: " + top_3_word_list[i], fontsize=8)
		plt.scatter(x, y, color=label_color[i], alpha=0.5)
	# plt.text(x + .0005, y, "Doc: " + str(i) + " --- Top Words: " + top_3_word_list[i], fontsize=8)#+ " --- Top 3 Words: " + top_3_word_list[i], fontsize=8)
	except:
		continue
# create graph labels and title and then show the scatter plot
plt.xlabel('Principal Component 1', fontsize = 15)
plt.ylabel('Principal Component 2', fontsize = 15)
plt.title('2 component PCA', fontsize = 20)
plt.show()












