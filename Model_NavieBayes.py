#Hacker earth

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing for train data
dataset = pd.read_csv('train.csv')
dataset = dataset.drop('currency',axis = 1)
dataset = dataset.drop('country',axis = 1)
nl = dataset.iloc[:, 4].values 
dataset = dataset.drop('keywords',axis = 1)
X = dataset.iloc[:, 3:9].values
y = dataset.iloc[:, 10].values
# Data Preprocessing for test data
dataset_test = pd.read_csv('test.csv')
dataset_test = dataset_test.drop('currency',axis = 1)
dataset_test = dataset_test.drop('country',axis = 1)
nl_test = dataset_test.iloc[:, 4].values
dataset_test = dataset_test.drop('keywords',axis = 1)
X_test= dataset_test.iloc[:, 3:9].values

#Encoding
from sklearn.preprocessing import LabelEncoder
#Encoding disable_communication for train data
labelencoder_X_coun = LabelEncoder()
X[:,1] = labelencoder_X_coun.fit_transform(X[:,1])
#Encoding disable_communication for test data
labelencoder_X_coun_test = LabelEncoder()
X_test[:,1] = labelencoder_X_coun_test.fit_transform(X_test[:,1])


# Data Preprocessing for Keywords for train data
#Cleaning data
import re
import nltk
#nltk.download('all')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus =[]
for i in range(0,108129):
    review = nl[i]
    review =review.lower()
    review = review.split('-')
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)          
    corpus.append(review)
 
#Creating Bag of words counts
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features =1500)#limiting to 1500 columns
X1 = cv.fit_transform(corpus).toarray()

#PCA
from sklearn.decomposition import PCA
pca =PCA(n_components = 2)#(n_components =None)
X1 = pca.fit_transform(X1)
explained_variance = pca.explained_variance_ratio
#Append X and X1
all_data = np.append(X, X1, axis =1)

# Data Preprocessing for Keywords in test data
#Cleaning data
corpus_test =[]
for i in range(0,63465):
    review1 = nl_test[i]
    review1 =review1.lower()
    review1 = review1.split('-')
    ps2 = PorterStemmer()
    review1 = [ps2.stem(word2) for word2 in review1 if not word2 in set(stopwords.words('english'))]
    review1 = ' '.join(review1)          
    corpus_test.append(review1)
 
#Creating Bag of words counts
cv2 = CountVectorizer(max_features =1500)#limiting to 1500 columns
X2 = cv2.fit_transform(corpus_test).toarray()
#PCA
X2 = pca.transform(X2)

#Append X and X1
all_data_test = np.append(X_test, X2, axis =1)


#Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
all_data = sc_X.fit_transform(all_data)
all_data_test = sc_X.fit_transform(all_data_test)

# Fitting Navie Bayes Classification to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(all_data, y)

# Predicting the Test set results
y_pred = classifier.predict(all_data_test)
my_df = pd.DataFrame(all_data_test)
my_df.to_csv('all_data_test.csv', index=False, header=False)
