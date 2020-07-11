#!/usr/bin/env python
# coding: utf-8

# # Text Mining  pipeline

# ### let's import a few free-open source tools to our convenience 

# In[1]:


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

# In[2]:


import pandas as pd
import numpy as np
data = pd.read_json("stupidstuff.json")


# In[3]:


data['body'] = data['body'].str.strip()
data.head()


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

# In[4]:


def to_lowercase(text):
    return text.lower()

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_numbers(text):
    return ''.join(i for i in text if not i.isdigit())

def remove_special_characters(text):
    return ''.join(c for c in text if c.isalnum())


# In[5]:


data['body_no_punct'] = data['body'].apply(remove_punctuation)


# In[6]:


def words_counter(text):
    words = nltk.word_tokenize(text)
    return len(words)
data['words_count'] = data['body_no_punct'].apply(words_counter)


# In[7]:


def characters_counter(text):
    return len(text)
data['characters_count'] = data['body'].apply(characters_counter)


# In[8]:


def sentences_counter(text, pst): 
    sentences = [sentence for sentence in pst.sentences_from_text(text, False) if not sentence in string.punctuation]
    return len(sentences)

pst = PST()
data['sentences_count'] = data['body'].apply(sentences_counter, args = (pst,))


# In[9]:


def punct_counter(text):
    puncts = [c for c in text if c in string.punctuation]
    return len(puncts)

data['punct_count'] = data['body'].apply(punct_counter)


# In[10]:


def all_caps_counter(text):
    words = nltk.word_tokenize(text)
    words_no_punct = [word for word in words if not any(char in set(string.punctuation) for char in word)]
    all_caps_words = [word for word in words_no_punct if (
            remove_special_characters(word).isupper() and len(remove_special_characters(word)) > 1 and not bool(
                re.search('(24:00|2[0-3]:[0-5][0-9]|[0-1][0-9]:[0-5][0-9])', word)))]
    return len(all_caps_words)
data['all_caps_count'] = data['body'].apply(all_caps_counter)


# In[11]:


def stopwords_counter(text):
    stops = [word for word in nltk.word_tokenize(text) if word in stopwords]
    return len(stops)
data['stopwords_count'] = data['body'].apply(stopwords_counter)


# In[12]:


def mean_length(text):
    text = nltk.word_tokenize(text)
    return (sum( map(len, text) ) / len(text))

data['mean_len_word'] = data['body_no_punct'].apply(lambda x: mean_length(x))


# In[13]:


def comma_counter(text): 
    return text.count(",")
data['comma_count'] = data['body'].apply(comma_counter)


# In[14]:


def qmark_counter(text): 
    return text.count("?")
data['qmark_count'] = data['body'].apply(qmark_counter)


# In[15]:


def excmark_counter(text): 
    return text.count("!")
data['excmark_count'] = data['body'].apply(excmark_counter)


# In[16]:


def quotes_counter(text):
    return text.count('\"')
data['quotes_count'] = data['body'].apply(quotes_counter)


# In[17]:


def currency_counter(text):
    curr = [c for c in text if c in "$£€₦₨￥"]
    return len(curr)
data['currency_count'] = data['body'].apply(currency_counter)


# In[18]:


def stopwords_cleaner(text):
    words = nltk.word_tokenize(text)
    text =[]
    for word in words:
        a = word.lower()
        if a not in stopwords:
            text.append(a)
    return text


# In[19]:
def stem_text(text):
    stem_words = np.vectorize(ps.stem)
    words = nltk.word_tokenize(text)
    text = ' '.join(stem_words(words))
    return text


data['body_no_punct'] = data['body_no_punct'].apply(to_lowercase)
data['body_no_punct'] = data['body_no_punct'].apply(stopwords_cleaner)
data['body_no_punct'] = data['body_no_punct'].apply(remove_special_characters)
data['body_no_punct'] = data['body_no_punct'].apply(remove_numbers)
data['body_no_punct'] = data['body_no_punct'].apply(stem_text)


# In[20]:



Bow = CountVectorizer(max_features=1000, ngram_range=(1, 2))
Bow.fit(data['body_no_punct'])
# X_Bow = Bow.fit_transform(data['body_no_punct'])


# In[21]:


X_Data = data.copy()


# In[22]:


X_Data = pd.get_dummies(X_Data, columns=['category'], drop_first=True)


# In[23]:


list(X_Data.columns.values)


# In[24]:

X_Data.drop(['body', 'id', 'rating'], axis=1, inplace=True)


# In[25]:
from sklearn import preprocessing

def normalize_columns(df_cols):
    names = df_cols.columns
    x = df_cols.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_normalized = pd.DataFrame(x_scaled)
    df_normalized.columns = names
    return df_normalized


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor


def fit_feature_selector(x_train, y_train):
    # configure to select a subset of features
    fs = SelectFromModel(RandomForestRegressor(criterion="mae"))
    # learn relationship from training data
    fs.fit(x_train, y_train)
    # transform train input data
    # X_train_fs_df = pd.DataFrame(fs.transform(X_train))
    cols = fs.get_support(indices=True)
    return cols


def select_features(x, cols):
    features_df_fs = x.iloc[:, cols]
    return features_df_fs
# ### ML TIME!

# In[27]:


from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, classification_report


# In[28]:


#y = pd.factorize(data['category'])[0]


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X_Data, data['rating'], test_size=0.2)

X_train_text = X_train['body_no_punct']
X_train_feats = X_train.drop('body_no_punct', axis=1)

X_train_feats_norm = normalize_columns(X_train_feats)
selected_cols = fit_feature_selector(X_train_feats_norm, y_train)
X_train_feats_sel = select_features(X_train_feats_norm, selected_cols)

X_test_text = X_test['body_no_punct']
X_test_feats = X_test.drop('body_no_punct', axis=1)

X_test_feats_norm = normalize_columns(X_test_feats)
X_test_feats_sel = select_features(X_test_feats_norm, selected_cols)

Train_X_Bow = Bow.transform(X_train_text)
Test_X_Bow = Bow.transform(X_test_text)

X_train_full = pd.concat([X_train_feats_sel, pd.DataFrame(Train_X_Bow.toarray())], axis=1)
# X_features_ = pd.concat([X_Data, pd.DataFrame(X_Bow.toarray())], axis=1)
# X_train_full.head()

X_test_full = pd.concat([X_test_feats_sel, pd.DataFrame(Test_X_Bow.toarray())], axis=1)

# In[31]:


param_grid = dict(C=[0.01, 0.1, 1, 10],
                    # kernel=['linear', 'rbf', 'sigmoid', 'poly'],
                    kernel=['linear', 'rbf', 'sigmoid'],
                    # degree=[2, 3, 5, 10],
                    gamma=['auto', 'scale', 0.1, 1, 10],
                    #epsilon=[0, 0.01, 0.1, 1]
                  )

# In[ ]:


model=SVR()
# CV_model = GridSearchCV(estimator=model, param_grid=param_grid, cv= 5, scoring= 'neg_mean_absolute_error', verbose=10)
CV_model = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=10, n_jobs=-1)
CV_model.fit(X_train_full, y_train)
cvbp = CV_model.best_params_

pred=CV_model.predict(X_test_full)

# In[ ]:


# model1=SVR(kernel=CV_model.best_params_['kernel'])


# In[ ]:


# model1.fit(X_train, y_train)
# pred=model1.predict(X_test)


# In[ ]:


mae = mean_absolute_error(y_test, pred)


# ### Eval train set as well

# In[34]:


X_Subset_ = X_train[0::17]
Y_Subset_ = y_train[0::17]

# pred_sub=model1.predict(X_Subset_)
# mean_absolute_error(y_test,pred)


# ### Feature Evaluation TBD

# In[37]:


import scipy.stats as st
import matplotlib.pyplot as plt

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(y_test, density=True, bins=30, label="Data")
mn, mx = plt.xlim()
plt.xlim(mn, mx)
#kde_xs = np.linspace(mn, mx, 301)
#kde = st.gaussian_kde(y_test)
#plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")
plt.legend(loc="upper left")
plt.ylabel('Probability')
plt.xlabel('Data')
plt.title("Histogram")


# In[36]:


import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython
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
plt.title("Histogram")


# In[ ]:




