# Gal Ziv,          ID: 205564198

setwd("D:/גל/RProgramming/JokesTextMining")

rm(list = ls()) # remove all variables from global environment
cat("\014") # clear the screen

library(tm)
library(SnowballC)
library(jsonlite)

jokes.df <- fromJSON("stupidstuff.json")

## TODO:
# Remove rows with na values
jokes.df <- na.omit(jokes.df)

# Remove jokes with empty body
jokes.df <- subset(jokes.df, body != "")

# Remove jokes with whole number rating (eg. 0.0, 3.0, 5.0) - likely indicates very low number of votes
jokes.df <- subset(jokes.df, rating != 0.00 & rating != 1.00 & rating != 2.00
                   & rating != 3.00 & rating != 4.00 & rating != 5.00)

# Remove duplicate jokes
jokes.df <- jokes.df[!duplicated(jokes.df$body),]

# Regular text mining stuff
jokes.corpus <- VCorpus(VectorSource(jokes.df$body))
jokes.corpus <- tm_map(jokes.corpus, stripWhitespace)
jokes.corpus <- tm_map(jokes.corpus, removePunctuation)
jokes.corpus <- tm_map(jokes.corpus, content_transformer(tolower))
jokes.corpus <- tm_map(jokes.corpus, removeWords, stopwords("english"))
jokes.corpus <- tm_map(jokes.corpus, stemDocument)

dtm <- DocumentTermMatrix(jokes.corpus)
inspect(dtm)

smallDtm <- removeSparseTerms(dtm, .9)
inspect(smallDtm)

jokes.df$body <- data.frame(text = unlist(sapply(jokes.corpus, `[`, "content")), stringsAsFactors = F)

# Fix spelling mistakes?