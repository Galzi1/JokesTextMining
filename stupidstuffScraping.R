# Gal Ziv,          ID: 205564198

setwd("D:/ть/RProgramming/JokesTextMining")

rm(list = ls()) # remove all variables from global environment
cat("\014") # clear the screen

# Libraries
library(rvest)
library(jsonlite)

# Helper functions to get text from html but preserve <br> elements as newline
html_text_collapse <- function(x, trim = FALSE, collapse = "\n"){
  UseMethod("html_text_collapse")
}

html_text_collapse.xml_nodeset <- function(x, trim = FALSE, collapse = "\n"){
  vapply(x, html_text_collapse.xml_node, character(1), trim = trim, collapse = collapse)
}

html_text_collapse.xml_node <- function(x, trim = FALSE, collapse = "\n"){
  paste(xml2::xml_find_all(x, ".//text()"), collapse = collapse)
}

# Read jokes from JSON to dataframe
jokes.df <- fromJSON("stupidstuff.json")

jokes_num <- length(jokes.df$body)

url_template <- "http://stupidstuff.org/jokes/joke.htm?jokeid="

for (i in jokes.df$id) {
  curr_url <- paste(url_template, i, sep = "")
  webpage <- read_html(curr_url)
  joke_body_html <- html_nodes(webpage, xpath = '//table[@bgcolor="#ffffff" and @width="470"]//table[@class="scroll"]//td')
  joke_body_html_text <- html_text_collapse(joke_body_html)
  if (length(joke_body_html_text) > 1) {
    joke_body_html_text <- paste(joke_body_html_text, sep = " ", collapse = " ")
  }
  jokes.df[jokes.df$id == i, "body"] <- gsub("[\r\n]", " ", joke_body_html_text)
  print(i)
}

write(toJSON(jokes.df), "stupidstuff.json")
