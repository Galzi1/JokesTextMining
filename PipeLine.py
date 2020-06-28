#!/usr/bin/env python
# coding: utf-8

# # Text Mining  pipeline

# ### let's import a few free-open source tools to our convenience 

# In[72]:


import nltk
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize.punkt import PunktSentenceTokenizer as PST
import string

# nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()


# ### Read data

# In[73]:


data = pd.read_json("stupidstuff.json")
data[0:10]


# In[74]:


data['body'] = data['body'].str.strip()


# ### Feature eng.

# - number of words
# - number of characters
# - number of sentences
# - number of punctuations
# - number of all CAPS-LOCK words
# - ratio of all CAPS-LOCK words to total words
# - number of stopwords
# - mean length of words
# - number of syntax errors
# - number of new-lines? /isn't sentence
# - specific punctuations: ?, !
# - repetition of words? / BoW-TfIdf

# In[75]:


def to_lowercase(text):
    return text.lower()


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_numbers(text):
    return ''.join(i for i in text if not i.isdigit())


def remove_special_characters(text):
    return ''.join(c for c in text if c.isalnum())


# In[76]:


data['body_no_punct'] = data['body'].apply(remove_punctuation)


# In[77]:


def words_counter(text):
    words = nltk.word_tokenize(text)
    words_no_punct = [word for word in words if not any(char in set(string.punctuation) for char in word)]
    return len(words_no_punct)
data['words_count'] = data['body'].apply(words_counter)


# In[78]:


def characters_counter(text):
    return len(text)
data['characters_count'] = data['body'].apply(characters_counter)


# In[79]:


def sentences_counter(text, pst): 
    sentences = [sentence for sentence in pst.sentences_from_text(text, False) if sentence not in string.punctuation]
    return len(sentences)

pst = PST()
data['sentences_count'] = data['body'].apply(sentences_counter, args=(pst,))


# In[80]:


def punct_counter(text):
    puncts = [c for c in text if c in string.punctuation]
    return len(puncts)

data['punct_count'] = data['body'].apply(punct_counter)


# In[81]:


def all_caps_counter(text):
    words = nltk.word_tokenize(text)
    words_no_punct = [word for word in words if not any(char in set(string.punctuation) for char in word)]
    all_caps_words = [word for word in words_no_punct if (
            remove_special_characters(word).isupper() and len(remove_special_characters(word)) > 1 and not bool(
                re.search('(24:00|2[0-3]:[0-5][0-9]|[0-1][0-9]:[0-5][0-9])', word)))]
    return len(all_caps_words)
data['all_caps_count'] = data['body'].apply(all_caps_counter)


# In[82]:


def stopwords_counter(text):
    stops = [word for word in nltk.word_tokenize(text) if word in stopwords]
    return len(stops)
data['stopwords_count'] = data['body'].apply(stopwords_counter)


# In[83]:


def mean_length(text):
    return sum(map(len, text)) / len(text)

data['mean_len_word'] = data['body'].apply(lambda x: mean_length(x))


# In[84]:


#def dot_counter(text): 
#    return text.count(".")
#data['dot_count'] = data['body'].apply(dot_counter)


# In[85]:


# Gal: I suspect that this feature will not be very helpful
def comma_counter(text): 
    return text.count(",")
data['comma_count'] = data['body'].apply(comma_counter)


# In[86]:


def qmark_counter(text): 
    return text.count("?")
data['qmark_count'] = data['body'].apply(qmark_counter)


# In[87]:


def excmark_counter(text): 
    return text.count("!")
data['excmark_count'] = data['body'].apply(excmark_counter)


# In[88]:


# New feature
def quotes_counter(text):
    return text.count('\"')
data['quotes_count'] = data['body'].apply(quotes_counter)


def currency_counter(text):
    curr = [c for c in text if c in "$£€₦₨￥"]
    return len(curr)
data['currency_count'] = data['body'].apply(currency_counter)


def stopwords_cleaner(text):
    words = nltk.word_tokenize(text)
    text = [word for word in words if word not in stopwords]
    return text


data['text_prepared'] = data['body_no_punct'].apply(to_lowercase)
data['text_prepared'] = data['text_prepared'].apply(stopwords_cleaner)
data['text_prepared'] = data['text_prepared'].apply(remove_special_characters)
data['text_prepared'] = data['text_prepared'].apply(remove_numbers)

print(data.head())
# In[89]:


def stem_text(text):
    text = [ps.stem(word) for word in text]
    return text

Bow = CountVectorizer(analyzer=stem_text)
X_Bow = Bow.fit_transform(data['text_prepared'])


# In[90]:


X_Data = data.copy()


# In[91]:


X_Data = pd.get_dummies(X_Data, columns=['category'])


# In[92]:


list(X_Data.columns.values)


# In[93]:


del X_Data['body']
del X_Data['id']
del X_Data['body_no_punct']
del X_Data['text_prepared']
del X_Data['rating']


# In[96]:


X_features_ = pd.concat([X_Data, pd.DataFrame(X_Bow.toarray())], axis=1)
X_features_.head()


# ### ML TIME!

# In[97]:


from sklearn.linear_model import Lasso
#from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error


# In[98]:


X_train, X_test, y_train, y_test = train_test_split(X_features_, data['rating'], test_size=0.3)


# In[99]:


param_grid = { 
    'alpha': [0.05, 0.1, 0.2],
    'fit_intercept': [True, False],
    'normalize' : [True, False],
    'max_iter' :[500, 1000],
}


# In[ ]:


model=Lasso()
CV_model = GridSearchCV(estimator=model, param_grid=param_grid, cv= 5, scoring= 'neg_mean_absolute_error', n_jobs=-1)
CV_model.fit(X_train, y_train)
print(CV_model.best_params_)


# In[ ]:


model1=Lasso(alpha=CV_model.best_params_['alpha'],
                fit_intercept= CV_model.best_params_['fit_intercept'],
                normalize=CV_model.best_params_['normalize'],
                max_iter=CV_model.best_params_['max_iter'])


# In[ ]:


model1.fit(X_train, y_train)
pred=model1.predict(X_test)


# In[ ]:


mae1 = mean_absolute_error(y_test,pred)

print(mae1)

pred=model.predict(X_test)


# In[ ]:


mae = mean_absolute_error(y_test,pred)

print(mae)

# ### Eval train set as well

# In[31]:


X_Subset_ = X_train[0::17]
Y_Subset_ = y_train[0::17]

pred_sub=model1.predict(X_Subset_)
mean_absolute_error(Y_Subset_,pred_sub)


# ### Feature Evaluation TBD

# In[35]:


import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(y_test, density=True, bins=30, label="Data")
mn, mx = plt.xlim()
plt.xlim(mn, mx)
kde_xs = np.linspace(mn, mx, 301)
kde = st.gaussian_kde(y_test)
plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")
plt.legend(loc="upper left")
plt.ylabel('Probability')
plt.xlabel('Data')
plt.title("Histogram");


# In[36]:


import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(pred, density=True, bins=30, label="Data")
mn, mx = plt.xlim()
plt.xlim(mn, mx)
kde_xs = np.linspace(mn, mx, 301)
kde = st.gaussian_kde(pred)
plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")
plt.legend(loc="upper left")
plt.ylabel('Probability')
plt.xlabel('Data')
plt.title("Histogram");


# In[ ]:




