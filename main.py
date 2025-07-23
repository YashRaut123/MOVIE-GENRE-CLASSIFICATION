
import numpy as np

import pandas as pd

import regex as re

import nltk

import matplotlib.pyplot as plt



from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

from gensim.models import Word2Vec



import warnings

warnings.filterwarnings('ignore')
train_data = pd.read_csv(r"C:\Users\yashr\Downloads\archive\Genre Classification Dataset\train_data.txt",delimiter=':::',names=['Title','Genre','Description'])

test_data = pd.read_csv(r"C:\Users\yashr\Downloads\archive\Genre Classification Dataset\test_data.txt",delimiter=':::',names=['Title','Description'])

test_sol = pd.read_csv(r"C:\Users\yashr\Downloads\archive\Genre Classification Dataset\test_data_solution.txt",delimiter=':::',names=['Title','Genre','Description'])
train_data.info()
test_data.info()
train_data.isnull().sum()
columns = ['Title','Description','Genre']

train_data = train_data[columns]

train_data.head()
train_data['Genre'].nunique()
train_data['Genre'].unique()
train_data['Genre'].value_counts()
plt.figure(figsize=(9,6))

plt.bar(train_data['Genre'].unique(),train_data['Genre'].value_counts())

plt.title('Movies released across each genre')

plt.xlabel('Genre')

plt.ylabel('No of movies')

plt.xticks(rotation = 70)

plt.show()
x_train = train_data.iloc[:,1].values # Considering only the movie description as a feature to predict the genre of the movie 

y_train = train_data.iloc[:,-1].values



# Selecting the testing data

x_test = test_data.iloc[:,-1].values # Considering only the movie description as a feature to predict the genre of the movie
def process_text(text):

    ''' This function performing text cleaning & converts words into it's base form '''

    

    # Convert to lowercase

    sentence = text.lower()

    

    # Removing username

    sentence = re.sub(r'@[a-zA-Z0-9_.]+','',sentence) 

    

    # Removing URLs

    sentence = re.sub(r"https?://\S+|www\.\S+",'',sentence)

    

    # Character normalization 

    sentence = re.sub(r"([a-zA-Z])\1{2,}", r'\1', sentence)

    

    # Removing punctuations 

    sentence = re.sub(r'[^\w\s]','',sentence)

    

    # Removing stopwords & converting to lowercase

    english_stop = stopwords.words('english')

    sentence = ' '.join([word for word in sentence.split() if word not in english_stop])

    

    # Performing word tokenization

    sentence = word_tokenize(sentence) # Returns a list of words

    

    # Performing lemmatization

    lemmatizer = WordNetLemmatizer()

    sentence = ' '.join([lemmatizer.lemmatize(word) for word in sentence])

    

    return sentence

# Applying text cleaning & preprocessing techniques on the text data



x_train = [process_text(desc) for desc in x_train]

x_test = [process_text(desc) for desc in x_test]
x_train_low = [desc.split() for desc in x_train]

x_test_low = [desc.split() for desc in x_test]
vocab = []



for desc in x_train_low:

    for word in desc:

        if word not in vocab:

            vocab.append(word)
print("Size of vocabulary: ",len(vocab))

tf_idf = TfidfVectorizer(max_features = 5000)

train_vector = tf_idf.fit_transform(x_train)

test_vector = tf_idf.transform(x_test)
model = Word2Vec(sentences = x_train_low, vector_size = 100, epochs = 5, workers = 5)
words_in_model = model.wv.index_to_key
def word_in_vocab(sentence):

    ''' This fn checks if all the words present in the sentence is part of the vocabulary of the model or not '''

    

    total = 0

    no_of_words = len(sentence)

    

    for word in sentence:

        if word in words_in_model:

            total += 1

            

    if total != no_of_words: 

    # If all words are not present in vocab of the model, we create a numpy array of zeros of same dimension as the word vector

        return False 

    else:

        return True
train_wv = [model.wv[sentence].sum(axis = 0) if len(sentence) != 0 and word_in_vocab(sentence) else np.zeros((100)) for sentence in x_train_low]

test_wv = [model.wv[sentence].sum(axis = 0) if len(sentence) != 0 and word_in_vocab(sentence) else np.zeros((100)) for sentence in x_test_low]
log_reg = LogisticRegression()

log_reg.fit(train_vector,y_train)
log_pred = log_reg.predict(train_vector)

print('Accuracy Score: ',round(accuracy_score(log_pred,y_train),2))
nb = GaussianNB()

nb.fit(train_vector.todense(),y_train)
nb_pred = nb.predict(train_vector.todense())

print('Accuracy Score: ',round(accuracy_score(nb_pred,y_train),2))
log_r = LogisticRegression()

log_r.fit(train_wv,y_train)
log_p = log_r.predict(train_wv)

print('Accuracy Score: ',round(accuracy_score(log_p,y_train),2))
nb = GaussianNB()

nb.fit(train_wv,y_train)
nb_pred = nb.predict(train_wv)

print('Accuracy Score: ',round(accuracy_score(nb_pred,y_train),2))
test_pred = log_reg.predict(test_vector)
test_sol['Predicted'] = test_pred
test_sol.head()



total_movies = len(test_sol)

correct_predictions = 0



for movie in range(1,total_movies+1):

    if test_sol['Genre'][movie] == test_sol['Predicted'][movie]:

        correct_predictions += 1

    

print("Total no of movie genre predicted: ",total_movies,"\nNo of movie genre predicted right: ",correct_predictions,

      "\nPercentage of correct predictions: ",round((correct_predictions/total_movies)*100,0),"%")
