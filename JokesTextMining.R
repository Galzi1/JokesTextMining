# Gal Ziv,          ID: 205564198

setwd("D:/ть/RProgramming/JokesTextMining")

rm(list = ls()) # remove all variables from global environment
cat("\014") # clear the screen

# Libraries
library(tm)
library(SnowballC)
library(jsonlite)
library(qdap)
library(ggplot2)
library(wordcloud)
library(reshape2)
library(igraph)
library(textclean)
library(rvest)
library(dplyr)
library(splitstackshape)
library(textstem)
library(forcats)

### Custom Functions:

#' @param string
#' @return same string with all punctuations removed except for '?' and '!'
rm_punct_except_qm_em <- function(x){
  return(gsub("[^[:alnum:][:space:]?!]", " ", x))
}

#' @param string
#' @return same string with a space between punctuations and other characters
sep_punct_text <- function(x){
  left  <- "([[:alpha:]])([[:punct:]])"
  right <- "([[:punct:]])([[:alpha:]])"
  struct_format <- "\\1 \\2"
  return(gsub(left, struct_format, gsub(right, struct_format, x)))
}

# stem_simple_past <- function(doc, dict){
#   ### past forms become present form
#   ### Collect past forms
#   past <- dict$Past[which(dict$Past %in% doc)]
#   
#   ### Collect infinitive forms of past forms
#   inf1 <- dict$Infinitive[which(dict$Past %in% doc)]
#   
#   ### Identify the order of past forms in temp
#   ind <- match(doc, past)
#   ind <- ind[is.na(ind) == FALSE]
#   
#   ### Where are the past forms in temp?
#   position <- which(doc %in% past)
#   
#   doc[position] <- inf1[ind]
# }
# 
# stem_past_participle <- function(doc, dict){
#   ### Collect past participle forms
#   pp <- dict$PP[which(dict$PP %in% doc)]
#   inf2 <- dict$Infinitive[which(dict$PP %in% doc)]
#   ind <- match(doc, pp)
#   ind <- ind[is.na(ind) == FALSE]
#   position <- which(doc %in% pp)
#   doc[position] <- inf2[ind]
# }

# Read jokes from JSON to dataframe
jokes.df <- fromJSON("stupidstuff.json")

# Remove rows with na values
jokes.df <- na.omit(jokes.df)

# Remove jokes with empty body
jokes.df <- subset(jokes.df, body != "")

ggplot(jokes.df, aes(x = fct_infreq(factor(category)))) +
  geom_bar(width=0.7, fill="steelblue") + 
  xlab("category") + 
  coord_flip()

# Remove jokes with whole number rating (eg. 0.0, 3.0, 5.0) or exactly in-half (eg. 1.5, 3.5)
# These ratings likely indicates very low number of votes
jokes.df <- subset(jokes.df, rating != 0.00 & rating != 1.00 & rating != 2.00
                   & rating != 3.00 & rating != 4.00 & rating != 5.00)
jokes.df <- subset(jokes.df, rating != 1.50 & rating != 2.50
                   & rating != 3.50 & rating != 4.5)

# Remove duplicate jokes
jokes.df <- jokes.df[!duplicated(jokes.df$body),]

### Text Processing

#WARNING: The following code does not remove spelling mistakes, only prints them to the Console...
# # Fix spelling in jokes (mainly to eliminate false diffrentiation of words that originated in spelling mistakes)
# sapply(jokes.df$body, which_misspelled)

#TODO plots that describe the jokes rating (distribution)
# ggplot(jokes.df, aes(x = rating)) + 
#   geom_density(color = "darkblue", fill = "lightblue")

ggplot(jokes.df, aes(x = fct_infreq(factor(category)))) +
  geom_bar(width=0.7, fill="steelblue") + 
  xlab("category") + 
  coord_flip()

#TODO Add sentiment/polarity of the joke (or sentences in it) as a variable in the dataframe

# Convert into corpus
jokes.corpus <- VCorpus(VectorSource(jokes.df$body))

# Replace contractions with long form
jokes.corpus <- tm_map(jokes.corpus, content_transformer(replace_contraction))

#TODO Consider keeping punctuations, or at least '!' and '?' characters that can alter the whole sentence meaning
# Remove punctuation
jokes.corpus <- tm_map(jokes.corpus, content_transformer(rm_punct_except_qm_em))

# separate punctuations and other characters with a space
jokes.corpus <- tm_map(jokes.corpus, content_transformer(sep_punct_text))

# remove numbers
jokes.corpus <- tm_map(jokes.corpus, removeNumbers)

# Remove whitespaces
jokes.corpus <- tm_map(jokes.corpus, stripWhitespace)

# apply(X = jokes.df,MARGIN = 1,function(t){sum(grepl(pattern = "!",x = t,fixed = TRUE))})


#TODO Consider keeping all-caps words (they can possibly make the joke funny/not funny)
# Convert to lower case
jokes.corpus <- tm_map(jokes.corpus, content_transformer(tolower))

# Remove common stop words in English
jokes.corpus <- tm_map(jokes.corpus, removeWords, stopwords("english"))

# Lemmatize words
jokes.corpus <- tm_map(jokes.corpus, content_transformer(lemmatize_strings), dictionary = lexicon::hash_lemmas)

# Stem words
jokes.corpus <- tm_map(jokes.corpus, content_transformer(stem_strings), stemmer = "en")

##### Advanced stemming
### Create a database
# x <- read_html("http://www.englishpage.com/irregularverbs/irregularverbs.html")
# 
# x %>%
#   html_table(header = TRUE) %>%
#   bind_rows %>%
#   rename(Past = `Simple Past`, PP = `Past Participle`) %>%
#   filter(!Infinitive %in% LETTERS) %>%
#   cSplit(splitCols = c("Past", "PP"),
#          sep = " / ", direction = "long") %>%
#   filter(complete.cases(.)) %>%
#   mutate_each(funs(gsub(pattern = "\\s\\(.*\\)$|\\s\\[\\?\\]",
#                         replacement = "",
#                         x = .))) -> mydic
# 
# jokes.corpus <- tm_map(jokes.corpus, content_transformer(stem_simple_past), mydic)
# jokes.corpus <- tm_map(jokes.corpus, content_transformer(stem_past_participle), mydic)

# Convert to DTM
dtm <- DocumentTermMatrix(jokes.corpus)
inspect(dtm)

# Remove sparse terms from DTM
smallDtm <- removeSparseTerms(dtm, .9)
inspect(smallDtm)

jokes.df$body <- data.frame(text = unlist(sapply(jokes.corpus, `[`, "content")), stringsAsFactors = F)

# Term frequency
termFreq <- colSums(as.matrix(smallDtm))
termFreqDf <- data.frame(term = names(termFreq), frequency = termFreq)

# Find top terms
topTerms <- termFreq[order(termFreq, decreasing = T)][1:20]
topTermsDf <- data.frame(term = names(topTerms), frequency = topTerms)

# plot 
ggplot(topTermsDf, aes(x = reorder(term, frequency), y = frequency)) +
  geom_bar(stat = "identity", fill = 'darkred') +
  coord_flip() + 
  xlab("term")

# TFxIDF
smallDTM_tfIdf <- DocumentTermMatrix(jokes.corpus, control = list(weighting = weightTfIdf))

termFreq_tfIdf <- colSums(as.matrix(smallDTM_tfIdf))
topTerms_tfIdf <- termFreq_tfIdf[order(termFreq, decreasing = T)][1:20]
topTerms_tfIdf_Df <- data.frame(term = names(topTerms_tfIdf), TFxIDF = topTerms_tfIdf)


ggplot(topTerms_tfIdf_Df, aes(x = reorder(term, TFxIDF), y = TFxIDF)) +
  geom_bar(stat="identity", fill='darkred') +
  coord_flip() + 
  xlab("term") 

# Word cloud
wordcloud(termFreqDf$term,termFreqDf$frequency, max.words = 100, colors = c('black','darkred'))

