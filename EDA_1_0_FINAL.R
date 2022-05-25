##### WD #####
setwd(WORKING_DIRECTORY)
##### LIBRARYS #####
library(wordcloud)
library(ggplot2)
#library(scales)
library(numbers)
library(dplyr)
library(tidyr)
library(BBmisc)
library(tokenizers)
library(NLP)
##### FUNKTIONEN #####



#++++++++++++++++++++++++++++++++++
# rquery.wordcloud() : Word cloud generator
# - http://www.sthda.com
#+++++++++++++++++++++++++++++++++++
# x : character string (plain text, web url, txt file path)
# type : specify whether x is a plain text, a web page url or a file path
# lang : the language of the text
# excludeWords : a vector of words to exclude from the text
# textStemming : reduces words to their root form
# colorPalette : the name of color palette taken from RColorBrewer package, 
# or a color name, or a color code
# min.freq : words with frequency below min.freq will not be plotted
# max.words : Maximum number of words to be plotted. least frequent terms dropped
# value returned by the function : a list(tdm, freqTable)
rquery.wordcloud <- function(x, type=c("text", "url", "file"), 
                             lang="english", excludeWords=NULL, 
                             textStemming=FALSE,  colorPalette="Dark2",
                             min.freq=3, max.words=200)
{ 
  library("tm")
  library("SnowballC")
  library("wordcloud")
  library("RColorBrewer") 
  
  if(type[1]=="file") text <- readLines(x)
  else if(type[1]=="url") text <- html_to_text(x)
  else if(type[1]=="text") text <- x
  
  # Load the text as a corpus
  docs <- Corpus(VectorSource(text))
  # Convert the text to lower case
  docs <- tm_map(docs, content_transformer(tolower))
  # Remove numbers
  docs <- tm_map(docs, removeNumbers)
  # Remove stopwords for the language 
  docs <- tm_map(docs, removeWords, stopwords(lang))
  # Remove punctuations
  docs <- tm_map(docs, removePunctuation)
  # Eliminate extra white spaces
  docs <- tm_map(docs, stripWhitespace)
  # Remove your own stopwords
  if(!is.null(excludeWords)) 
    docs <- tm_map(docs, removeWords, excludeWords) 
  # Text stemming
  if(textStemming) docs <- tm_map(docs, stemDocument)
  # Create term-document matrix
  tdm <- TermDocumentMatrix(docs)
  m <- as.matrix(tdm)
  v <- sort(rowSums(m),decreasing=TRUE)
  d <- data.frame(word = names(v),freq=v)
  # check the color palette name 
  if(!colorPalette %in% rownames(brewer.pal.info)) colors = colorPalette
  else colors = brewer.pal(8, colorPalette) 
  # Plot the word cloud
  set.seed(1234)
  wordcloud(d$word,d$freq, min.freq=min.freq, max.words=max.words,
            random.order=FALSE, rot.per=0.35, 
            use.r.layout=FALSE, colors=colors)
  
  invisible(list(tdm=tdm, freqTable = d))
}
#++++++++++++++++++++++
# Helper function
#++++++++++++++++++++++
# Download and parse webpage
html_to_text<-function(url){
  library(RCurl)
  library(XML)
  # download html
  html.doc <- getURL(url)  
  #convert to plain text
  doc = htmlParse(html.doc, asText=TRUE)
  # "//text()" returns all text outside of HTML tags.
  # We also don’t want text such as style and script codes
  text <- xpathSApply(doc, "//text()[not(ancestor::script)][not(ancestor::style)][not(ancestor::noscript)][not(ancestor::form)]", xmlValue)
  # Format text vector into one character string
  return(paste(text, collapse = " "))
}


##### NUR GEMATCHTE DATEN #####
DATA = filter(DATA, filename %in% MATCHED_ID)
EDA_DATA = filter(EDA_DATA, filename %in% MATCHED_ID)
####### EDA #######
UNMATCHED_LABELS = LABELS %>% filter(., Id %in% UNMATCHED_ID)
MATCHED_LABELS = LABELS %>% filter(., Id %in% MATCHED_ID)

## 5. % der Sätze mit Labels in Abschnitten von Gesamtmenge #####
subset = filter(DATA, section_ID %% 2 == 1)
sum(subset$nLabels)/sum(DATA$nLabels)
# 5.1 Untersuchung des Datensatz LABELS#####

MATCHED_LABELS = LABELS %>% filter(., Id %in% MATCHED_ID)

# unterschiedliche Labeltokens
unique(MATCHED_LABELS$cleaned_label)

subset = MATCHED_LABELS %>% 
  group_by(., dataset_label)  %>% 
  summarise(., n = n()) %>% 
  mutate(freq = n / sum(n)) %>% 
  arrange(desc(freq))

ordered_dataset_label = subset$dataset_label[1:20]

# Abbildung 11
ggplot(data = subset, aes(x = dataset_label, y = freq)) + 
  geom_bar(stat = "identity") + 
  scale_y_continuous(name = "Häufigkeit in Publikationen in %", limits = c(0,0.3)) + 
  scale_x_discrete(name = "Labeltokens",limits = ordered_dataset_label) + 
  coord_flip()

# Tabelle 3
summary(subset$n)

# Abbildung 12
rquery.wordcloud(MATCHED_LABELS$cleaned_label, min.freq = 1, max.words = 50, colorPalette = "RdYlGn")

##### DATA #####
# 5.2.1. Anzahl Sätze / Wörter / Labels pro Publikation #####

# Anzahl Sätze
subset = DATA
subset = DATA %>% group_by(., filename) %>% summarise(Anz_Saetze = n())

# Abbildung 13
dev.new()
ggplot(subset, aes(y = Anz_Saetze)) + 
  geom_boxplot() + 
  scale_x_discrete(name = "")+ 
  scale_y_continuous(name = "",limits = c(0,750))+ coord_flip()

# durchschnittliche Sätze pro Publikation
summary(subset$Anz_Saetze)

# Abbildung 14
dev.new()
ggplot(DATA, aes(x = nWords)) + 
  geom_histogram(aes(y = ..count../sum(..count..)), binwidth = 1) + 
  scale_y_continuous(name = "rel. Häufigkeit") + 
  scale_x_continuous(name = "Satzlänge", limits =c(1,100))

# Tabelle 4
summary(DATA$nWords)

subset = DATA$nWords


# % erreichte Wörter bei T = 50
length(subset[subset <= 50]) / length(subset) 

# % der Sätze sind Labels 
nrow(filter(DATA, nLabels > 0)) / nrow(DATA)

# % der Wörter sind Labels 
sum(DATA$nLabels) / sum(DATA$nWords)


# 5.2.2. Wo treten die Labels im Text auf? #####

subset = DATA %>% 
  group_by(.,filename) %>% 
  mutate(.,position = BBmisc::normalize(ID, method = "range", on.constant = "quiet"))

subset = subset %>% filter(., nLabels > 0)

# Abbildung 15
ggplot(subset, aes(x = position)) + 
  geom_histogram(aes(y = ..count../sum(..count..)), binwidth = 0.05) + 
  scale_y_continuous(name = "rel. Häufigkeit in Publikationen") + 
  scale_x_continuous(name = "Position des Satzes mit nLabels > 0 im Text", limits = c(0,1))

subset = DATA %>% 
  group_by(.,filename,section_ID) %>% 
  mutate(., position = BBmisc::normalize(sentence_ID, method = "range", on.constant = "quiet")) %>%
  filter(., nLabels > 0)

# Abbildugn 16
ggplot(subset, aes(x = position)) + 
  geom_histogram(aes(y = ..count../sum(..count..)), binwidth = 0.05) + 
  scale_y_continuous(name = "rel. Häufigkeit in Publikationen") + 
  scale_x_continuous(name = "Position des Satzes mit Anz. Labels > 0 im Abschnitt", limits = c(0,1))

# 5.2.3. In welchem Abschnitt treten die Labels auf #####

# Für ungerade Section IDs gilt, dass es zuvor in den Rohdaten LABELS eine "section" war
subset = DATA %>% 
  filter(., nLabels > 0) %>% 
  mutate(., section_ID = ifelse(section_ID %% 2 == 0, section_ID - 1, section_ID)) %>%
  group_by(filename, section_ID) %>% 
  summarise(., anz = n()) 

# Rückumwandlung Section IDs in Abschnittsbezeichnungen und zählen der Häufigkeiten
anz.labels.pro.abschnitt = merge(DATA, subset) %>% 
  group_by(text) %>% 
  summarise(anz = n()) %>%
  mutate(., anz = anz / sum(anz)) %>%
  arrange(., desc(anz))

# Abbildung 17
anz.labels.pro.abschnitt.vct = anz.labels.pro.abschnitt$text[1:20]
dev.new()
ggplot(data = anz.labels.pro.abschnitt, aes(x = text, y = anz)) + 
  geom_bar(stat = "identity") + 
  scale_y_continuous(name = "rel. Häufigkeit an Sätzen in welchen Labels sind",limits = c(0,0.1)) + 
  scale_x_discrete(name = "Abschnittstitel", limits = anz.labels.pro.abschnitt.vct) + 
  coord_flip()

# 5.2.4 einzigartige Wörter / Zeichen #####
subset = clean_text(unlist(lapply(DATA$text, tokenize_ptb))) #TODO: dauert vermutlich bis zu eine Stunde

#unterschiedliche Wörter
length(unique(subset))

# Wie häufig jedes Wort vorkommt
haufigkeit = table(subset)

inv.vf = function(x){
  result = c()
  for(i in x){
    result = c(result, length(haufigkeit[haufigkeit > i]))
  }
  result = result /length(haufigkeit)
  return(result)
}

#inv.vf(0:100)

# Test wie viele Wörter erreicht werden für haufigkeit > x
length(haufigkeit[haufigkeit > 31])
length(haufigkeit[haufigkeit > 32])

# % Erreichte Wörter
inv.vf(31)

# Abbildung 18
dev.new()
plot(inv.vf(0:50), type = "l", xlab = "Häufigkeit der Wörter", ylab = expression(paste("F"^"-1","(X)")))


##### EDA_DATA #####
# 5.3.1 text feature #####
# Häufigkeiten Zählen
subset = EDA_DATA %>% filter(., labels == 1) %>% group_by(word_ID) %>% summarise(., anz_labels = n())

#  % an erwischten Labels
vf = function(x){
  return(sum(filter(subset, word_ID <= x)$anz_labels)/nrow(filter(EDA_DATA, labels == 1)))
}

vf(50)

set.seed(3)
filesname_subset = sample(MATCHED_ID, 1000)

subset = filter(EDA_DATA, nLabels > 0, labels == 0, filename %in% filesname_subset) #Sätze in welchen Labels vorhanden sind aber ohne die Labelwörter selbst
subset_2 = filter(EDA_DATA, nLabels == 0, filename %in% filesname_subset) #Sätze ohne Labelwörter



# Abbildung 19 Teil 1
dev.new()
rquery.wordcloud(subset$text, max.words = 50, colorPalette = "RdYlGn", excludeWords = c("number"))

# Abbildung 19 Teil 2
dev.new()
rquery.wordcloud(subset_2$text, max.words = 50, colorPalette = "RdYlGn", excludeWords = c("number"))

# 5.3.2 POS Feature #####

subset = filter(EDA_DATA, labels == 1) %>% group_by(POS) %>% summarise(., n = n()) %>% mutate(freq = n / sum(n)) %>% arrange(desc(freq))
DATA_2 = filter(EDA_DATA, labels == 0) %>% group_by(POS) %>% summarise(., n = n()) %>% mutate(freq = n / sum(n)) %>% arrange(desc(freq))

ordered_POS = subset$POS[1:10]

# Abbildung 20 Teil 1
dev.new()
ggplot(subset, aes(x = POS, y = freq)) + geom_bar(stat = "identity") + scale_y_continuous(name = "Anz. Wörter mit isLabel = 1") + scale_x_discrete(limits = ordered_POS)

# Abbildung 20 Teil 2
dev.new()
ggplot(DATA_2, aes(x = POS, y = freq)) + geom_bar(stat = "identity") + scale_y_continuous(name = "Anz. Wörter mit isLabel = 0", limits = c(0,0.7)) + scale_x_discrete(limits = ordered_POS)

# 5.3.3 isUpper #####
subset = filter(EDA_DATA, labels == 1) %>% group_by(isFirstUpper) %>% summarise(., n = n()) %>% mutate(freq = n / sum(n)) %>% arrange(desc(freq))
subset_2 = filter(EDA_DATA, labels == 0) %>% group_by(isFirstUpper) %>% summarise(., n = n()) %>% mutate(freq = n / sum(n)) %>% arrange(desc(freq))

ordered_Classes = subset$isFirstUpper

# Abbildung 22 Teil 1
dev.new()
ggplot(subset, aes(x = isFirstUpper, y = freq)) + geom_bar(stat = "identity") + scale_y_continuous(name = "Anz. Wörter mit isLabel = 1", limits = c(0,1)) + scale_x_discrete(limits = ordered_Classes)

# Abbildung 22 Teil 2
dev.new()
ggplot(subset_2, aes(x = isFirstUpper, y = freq)) + geom_bar(stat = "identity") + scale_y_continuous(name = "Anz. Wörter mit isLabel = 0", limits = c(0,1)) + scale_x_discrete(limits = ordered_Classes)

# 5.3.4 upperFeature #####
subset = filter(EDA_DATA, labels == 1) %>% group_by(upperFeature) %>% summarise(., n = n()) %>% mutate(freq = n / sum(n)) %>% arrange(desc(freq))
subset_2 = filter(EDA_DATA, labels == 0) %>% group_by(upperFeature) %>% summarise(., n = n()) %>% mutate(freq = n / sum(n)) %>% arrange(desc(freq))

ordered_Classes = subset$upperFeature

# Abbildung 25 Teil 1
dev.new()
ggplot(subset, aes(x = upperFeature, y = freq)) + geom_bar(stat = "identity") + scale_y_continuous(name = "Anz. Wörter mit isLabel = 1", limits = c(0,1)) #+ scale_x_discrete(limits = ordered_Classes)

# Abbildung 25 Teil 2
dev.new()
ggplot(subset_2, aes(x = upperFeature, y = freq)) + geom_bar(stat = "identity") + scale_y_continuous(name = "Anz. Wörter mit isLabel = 0", limits = c(0,1)) #+ scale_x_discrete(limits = ordered_Classes)

