#!/usr/bin/env python
# coding: utf-8

# <h1 align="center">Zewail University of Science and Technology</h1>
# <h2 align="center">CIE 417 (Fall 2018)</h2>
# <h2 align="center">Ham or Spam Email Classifier</h3>
# <h3 align="center">Logistic Regression</h3>

# In[47]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import string
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer 
from sklearn.linear_model import LogisticRegression
import nltk


# <h3 align="left">Load Dataset</h3>

# In[48]:


dataset = pd.read_csv('SMSSpamCollection.csv', encoding='latin-1')
dataset = dataset.loc[:,['Label','Email']]
dataset.head()


# <h3 align="left">Change Ham/Spam to 0/1</h3>

# In[49]:


d={'ham':0, 'spam':1}
dataset.Label = list(map(lambda x:d[x],dataset.Label))
dataset.head()


# <h3 align="left">Extract the TFIDF Feature From the Emails</h3>

# In[50]:


#let the features be "words", exclude common words found in the stop-words list, and set the maximum number of features
vectorizer = TfidfVectorizer(analyzer='word',
                             stop_words = 'english',
                             max_features = 5000)

ps = PorterStemmer()
def stem_string(s):
        #remove punctuation
        s = re.sub(r'[^\w\s]',' ',s)
        #split into words
        tokens = word_tokenize(s)
        #get the stem of words then return
        return ' '.join([ps.stem(w) for w in tokens])


# In[51]:


feature = vectorizer.fit_transform(stem_string(s) for s in dataset.Email)


# <h3 align="left">Split the Dataset into Training Set and Test Set</h3>

# In[52]:


Xtrain, Xtest, ytrain, ytest = train_test_split(feature, dataset.Label, test_size=0.2, random_state=1)


# <h3 align="left">Determine the Best Penalty (l1/l2) and the Best Regularization Parameter</h3>

# In[53]:


lambdas = np.linspace(0.1,3,num=20)
best_penalty = 'l1'
best_lambda = 0.1
best_f1_score = 0

for p in ('l1','l2'):
    for l in lambdas:
        LR = LogisticRegression(penalty=p, C=l, class_weight='balanced')
        scores = cross_val_score(LR, Xtrain, ytrain, scoring='f1', cv=5)
        if ((np.mean(scores)) > best_f1_score):
            best_f1_score = np.mean(scores)
            best_lambda = l
            best_penalty = p

print('The best average of F1 scores on the training set is: ', best_f1_score, ', and it happens with:\nRegularization Parameter: ', best_lambda, '\nPenalty: ', best_penalty)


# <h3 align="left">Fit the Chosen Model and Calculate Accuracy and F1 Score on Test Set</h3>

# In[54]:


model = LogisticRegression(penalty=best_penalty, C=best_lambda, class_weight='balanced')
model.fit(Xtrain,ytrain)
prediction1 = model.predict(Xtest)
f1score1 = f1_score(ytest, prediction1)
accuracy1 = accuracy_score(ytest,prediction1)

print('Test Accuracy: ', accuracy1, '\nTest F1 Score: ', f1score1)


# <h3 align="left">Test the Model on The Two Following Email Contents</h3>

# In[55]:


test_sample = [
              "['URGENT!] Your Mobile No 398174814449 was awarded a vacation",
              "Hello my friend, how are you?"
              ]

feature2 = vectorizer.transform(stem_string(s) for s in test_sample)
prediction2 = model.predict(feature2)
print('The first email in the sample is: ')
if (prediction2[0]==0):
    print('Ham')
else:
    print('Spam')
print('The second email in the sample is: ')
if (prediction2[1]==0):
    print('Ham')
else:
    print('Spam')

