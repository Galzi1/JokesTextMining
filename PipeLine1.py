# Gal Ziv,          ID: 205564198
# Gilad Leshem      ID: 037994480

import nltk
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize.punkt import PunktSentenceTokenizer as PST
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import string
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, classification_report


# Preprocessing functions
def to_lowercase(text):
    """
    :param text: a string represents a joke's body
    :return: text converted to lower-case
    """
    return text.lower()


def remove_punctuation(text):
    """
    :param text: a string represents a joke's body
    :return: text with punctuation removed
    """
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_numbers(text):
    """
    :param text: a string represents a joke's body
    :return: text with numbers and digits removed
    """
    return ''.join(i for i in text if not i.isdigit())


def remove_special_characters(text):
    """
    :param text: a string represents a joke's body
    :return: text with all 'special characters' (non-alphanumeric) removed
    """
    return ''.join(c for c in text if c.isalnum())


def stopwords_cleaner(text):
    """
    :param text: a string represents a joke's body
    :return: text with all stopwords removed
    """
    words = nltk.word_tokenize(text)
    return [word for word in words if word not in stopwords][0]


def stem_text(text):
    """
    :param text: a string represents a joke's body
    :return: text with all words stemmed
    """
    stem_words = np.vectorize(ps.stem)
    words = nltk.word_tokenize(text)
    text = ' '.join(stem_words(words))
    return text


def normalize_columns(df_cols):
    """
    :param df_cols: subset of columns from dataframe. Must all be numeric
    :return: a dataframe of the given columns with all columns normalized from 0 to 1
    """
    names = df_cols.columns
    x = df_cols.values
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_normalized = pd.DataFrame(x_scaled)
    df_normalized.columns = names
    return df_normalized


def preprocess(df_text_col):
    """
    :param df_text_col: a pandas series containing plain text
    :return: df_text_col with all featured preprocessed and cleaned
    """
    df_text_col = df_text_col.apply(remove_punctuation)
    print("Ciao 1")
    df_text_col = df_text_col.apply(remove_special_characters)
    print("Ciao 2")
    # df = df.apply(normalise_text, axis=1)
    df_text_col = df_text_col.apply(remove_numbers)
    print("Ciao 3")
    df_text_col = df_text_col.apply(to_lowercase)
    print("Ciao 4")
    df_text_col = df_text_col.apply(stopwords_cleaner)
    print("Ciao 5")
    df_text_col = df_text_col.apply(stem_text)
    # df = df.apply(lemm_text, axis=1)
    return df_text_col


# Feature creation functions
def words_counter(text):
    """
    :param text: a string represents a joke's body
    :return: number of words in text
    """
    words = nltk.word_tokenize(text)
    return len(words)


def characters_counter(text):
    """
    :param text: a string represents a joke's body
    :return: number of characters in text
    """
    return len(text)


def sentences_counter(text, pst):
    """
    :param text: a string represents a joke's body
    :return: number of sentences in text using PST to separate sentences
    """
    sentences = [sentence for sentence in pst.sentences_from_text(text, False) if not sentence in string.punctuation]
    return len(sentences)


def punct_counter(text):
    """
    :param text: a string represents a joke's body
    :return: number of punctuation in text
    """
    puncts = [c for c in text if c in string.punctuation]
    return len(puncts)


def all_caps_counter(text):
    """
    :param text: a string represents a joke's body
    :return: number of all-caps words in text
    """
    words = nltk.word_tokenize(text)
    words_no_punct = [word for word in words if not any(char in set(string.punctuation) for char in word)]
    all_caps_words = [word for word in words_no_punct if (
            remove_special_characters(word).isupper() and len(remove_special_characters(word)) > 1 and not bool(
                re.search('(24:00|2[0-3]:[0-5][0-9]|[0-1][0-9]:[0-5][0-9])', word)))]
    return len(all_caps_words)


def stopwords_counter(text):
    """
    :param text: a string represents a joke's body
    :return: number of stopwords in text
    """
    stops = [word for word in nltk.word_tokenize(text) if word in stopwords]
    return len(stops)


def mean_length(text):
    """
    :param text: a string represents a joke's body
    :return: mean length of the words in text
    """
    words = nltk.word_tokenize(text)
    return sum(map(len, words)) / len(words)


def comma_counter(text):
    """
    :param text: a string represents a joke's body
    :return: number of commas in text
    """
    return text.count(",")


def qmark_counter(text):
    """
    :param text: a string represents a joke's body
    :return: number of question marks in text
    """
    return text.count("?")


def excmark_counter(text):
    """
    :param text: a string represents a joke's body
    :return: number of exclamation marks in text
    """
    return text.count("!")


def quotes_counter(text):
    """
    :param text: a string represents a joke's body
    :return: number of quotes in text
    """
    return text.count('\"')


def currency_counter(text):
    """
    :param text: a string represents a joke's body
    :return: number of currency symbols (most frequently-used kinds) in text
    """
    curr = [c for c in text if c in "$£€₦₨￥"]
    return len(curr)


def get_sentiment(text):
    """
    :param text: a string represents a joke's body
    :return: the sentiment of the text using TextBlob
    """
    return TextBlob(text).sentiment


def create_features(df):
    """
    :param df: a dataframe containing a text column named 'body'
    :return: df with new added columns consisting of new features
    """
    df['words_count'] = df['body'].apply(words_counter)
    df['characters_count'] = df['body'].apply(characters_counter)
    pst = PST()
    df['sentences_count'] = df['body'].apply(sentences_counter, args=(pst,))
    df['punct_count'] = df['body'].apply(punct_counter)
    df['all_caps_count'] = df['body'].apply(all_caps_counter)
    df['stopwords_count'] = df['body'].apply(stopwords_counter)
    df['mean_len_word'] = df['body'].apply(mean_length)
    df['comma_count'] = df['body'].apply(comma_counter)
    df['qmark_count'] = df['body'].apply(qmark_counter)
    df['excmark_count'] = df['body'].apply(excmark_counter)
    df['quotes_count'] = df['body'].apply(quotes_counter)
    df['currency_count'] = df['body'].apply(currency_counter)

    df['Sentiment_Score'] = df['body'].apply(get_sentiment)
    sentiment_series = df['Sentiment_Score'].tolist()
    columns = ['Polarity', 'Subjectivity']
    temp_df = pd.DataFrame(sentiment_series, columns=columns, index=df.index)
    df['Polarity'] = temp_df['Polarity']
    df['Subjectivity'] = temp_df['Subjectivity']
    df = df.drop(['Sentiment_Score'], axis=1)
    return df


def fit_feature_selector(_x_train, _y_train):
    """
    Trains a feature selector from a RandomForestRegressor
    :param _x_train: independent features subset for model training
    :param _y_train: dependent features subset for model training
    :return: list of most important features
    """
    # configure to select a subset of features
    fs = SelectFromModel(RandomForestRegressor(n_estimators=1000, criterion="mae"))
    # learn relationship from training data
    fs.fit(_x_train, _y_train)
    # get most important features
    cols = fs.get_support(indices=True)
    return cols


def select_features(x, cols):
    """
    :param x: a dataframe
    :param cols: columns to select from x
    :return: a subset of x by cols
    """
    features_df_fs = x.iloc[:, cols]
    return features_df_fs


# Uncomment this in the first run in order to download stopwords corpus from nltk
# nltk.download('stopwords')
#

# Getting stop-words from nltk library
stopwords = nltk.corpus.stopwords.words('english')

# Initializing porter stemmer
ps = nltk.PorterStemmer()

# Reading data from pre-scraped json file into a pandas dataframe
data = pd.read_json("stupidstuff.json")

# Removing leading and trailing spaces from 'body' column strings
data['body'] = data['body'].str.strip()

print("Hi 1")
data = create_features(data)
print("Hi 2")
data['body'] = preprocess(data['body'])
print("Hi 3")

Bow = CountVectorizer(max_features=1000, ngram_range=(1, 2))
Bow.fit(data['body'])

X_Data = data.copy()

X_Data = pd.get_dummies(X_Data, columns=['category'], drop_first=True)

X_Data.drop(['id', 'rating'], axis=1, inplace=True)

print("Hi 4")

X_train, X_test, y_train, y_test = train_test_split(X_Data, data['rating'], test_size=0.2)

X_train_text = X_train['body']
X_train_feats = X_train.drop('body', axis=1)

X_train_feats_norm = normalize_columns(X_train_feats)

print("Hi 5")
selected_cols = fit_feature_selector(X_train_feats_norm, y_train)
X_train_feats_sel = select_features(X_train_feats_norm, selected_cols)

X_test_text = X_test['body']
X_test_feats = X_test.drop('body', axis=1)

print("Hi 6")

X_test_feats_norm = normalize_columns(X_test_feats)
X_test_feats_sel = select_features(X_test_feats_norm, selected_cols)

Train_X_Bow = Bow.transform(X_train_text)
Test_X_Bow = Bow.transform(X_test_text)

X_train_full = pd.concat([X_train_feats_sel, pd.DataFrame(Train_X_Bow.toarray())], axis=1)

X_test_full = pd.concat([X_test_feats_sel, pd.DataFrame(Test_X_Bow.toarray())], axis=1)

param_grid = dict(C=[0.01, 0.1, 1, 10],
                    # kernel=['linear', 'rbf', 'sigmoid', 'poly'],
                    kernel=['linear', 'rbf', 'sigmoid'],
                    # degree=[2, 3, 5, 10],
                    gamma=['auto', 'scale', 0.1, 1, 10],
                    #epsilon=[0, 0.01, 0.1, 1]
                  )

model = SVR()
# CV_model = GridSearchCV(estimator=model, param_grid=param_grid, cv= 5, scoring= 'neg_mean_absolute_error', verbose=10)
CV_model = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=10, n_jobs=-1)
CV_model.fit(X_train_full, y_train)
cvbp = CV_model.best_params_

pred = CV_model.predict(X_test_full)

mae = mean_absolute_error(y_test, pred)

X_Subset_ = X_train[0::17]
Y_Subset_ = y_train[0::17]

# pred_sub=model1.predict(X_Subset_)
# mean_absolute_error(y_test,pred)

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




