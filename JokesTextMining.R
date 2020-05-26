# Gal Ziv,          ID: 205564198

setwd("D:/ть/RProgramming/JokesTextMining")

rm(list = ls()) # remove all variables from global environment
cat("\014") # clear the screen

# Libraries
library(tm)
library(SnowballC)
library(jsonlite)
library(qdap)

# Read jokes from JSON to dataframe
jokes.df <- fromJSON("stupidstuff.json")

# Remove rows with na values
jokes.df <- na.omit(jokes.df)

# Remove jokes with empty body
jokes.df <- subset(jokes.df, body != "")

# Remove jokes with whole number rating (eg. 0.0, 3.0, 5.0) - likely indicates very low number of votes
jokes.df <- subset(jokes.df, rating != 0.00 & rating != 1.00 & rating != 2.00
                   & rating != 3.00 & rating != 4.00 & rating != 5.00)

# Remove duplicate jokes
jokes.df <- jokes.df[!duplicated(jokes.df$body),]

# Fix spelling in jokes (mainly to eliminate false diffrentiation of words that originated in spelling mistakes)
sapply(jokes.df$body, which_misspelled)

# Convert into corpus
jokes.corpus <- VCorpus(VectorSource(jokes.df$body))

# Remove whitespaces
jokes.corpus <- tm_map(jokes.corpus, stripWhitespace)

# Remove punctuation
jokes.corpus <- tm_map(jokes.corpus, removePunctuation)

# Convert to lower case
jokes.corpus <- tm_map(jokes.corpus, content_transformer(tolower))

# Remove common stop words in English
jokes.corpus <- tm_map(jokes.corpus, removeWords, stopwords("english"))

# Stem words
jokes.corpus <- tm_map(jokes.corpus, stemDocument)

# Convert to DTM
dtm <- DocumentTermMatrix(jokes.corpus)
inspect(dtm)

# Remove sparse terms from DTM
smallDtm <- removeSparseTerms(dtm, .9)
inspect(smallDtm)

df <- data.frame(text = unlist(sapply(jokes.corpus, `[`, "content")), stringsAsFactors = F)
jokes.df$body <- data.frame(text = unlist(sapply(jokes.corpus, `[`, "content")), stringsAsFactors = F)