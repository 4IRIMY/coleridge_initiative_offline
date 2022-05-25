##### LIBRARYS #####
library(rjson)
library(tidyr)
library(dplyr)
library(stringr)
library(tidyverse)
library(tokenizers)
library(lexRankr)
library(openNLP)
library(rJava)
library(NLP)
library(qdapRegex)

##### FUNKTIONEN #####

# Preprocessfunktionen
label_matching = function(wrd_lvl_dataset){
  
  wrd_lvl_dataset = wrd_lvl_dataset  %>% mutate(., label = 0) %>% mutate(label_text = clean_text(text)) # Zunächst neue Spalte und alle Labels auf 0 setzen
  filename = wrd_lvl_dataset$filename[1]
    
  to_be_tested_labels = (LABELS %>% filter(., Id == filename))$dataset_label
    
  for(label in to_be_tested_labels){ 
      
    label_words = unlist(tokenize_ptb(label))                                                           # Zu testendes Label in Wörter aufteilen 
      
    laenge = length(label_words)
      
    cleaned_label_words = clean_text(label_words)                                                       # Label Wörter kleinschreiben und Zeichen entfernen
      
    results_for_first_label_word = filter(wrd_lvl_dataset, label_text == cleaned_label_words[1])
    start_ids = results_for_first_label_word$ID
      
    for(start_id in start_ids){
      to_be_tested_indizes = seq(start_id,start_id + laenge - 1)                                        # entsprechend länge des labels die nächsten indizes nach start_id prüfen
        
      to_be_tested_data = filter(wrd_lvl_dataset, ID %in% to_be_tested_indizes)$label_text              # Falls label der untersuchten Reihe entspricht --> label = 1
        
      is_Same = isTRUE(all.equal(to_be_tested_data, cleaned_label_words))                               # Testen ob Label == Reihe im Datensatz
      if(is_Same){
        wrd_lvl_dataset$label[to_be_tested_indizes] = 1
      }
    }
  }
  return(wrd_lvl_dataset)
}                      # Fügt Labels mittels des String Matching Algorithmus ein

is_upper = function(input_string){
  first_character = substring(input_string, 1,1)
  return(as.numeric(str_detect(first_character, "[[:upper:]]")))
}                               # Schaut, ob erstes Zeichen großgeschrieben ist

count_upper = function(input_string){
  return(str_count(input_string, "[[:upper:]]"))
}                            # Zählt wie viele Zeichen großgeschrieben sind

preprocessing_to_text_level = function(json_dataset, file_name){
  section_ID = c()
  text = c()
  is_section = c()
  filename = c()
  
  z = 1
  
  input_json = json_dataset[[file_name]]
  
  for(i in 1:length(input_json)){
    for(j in c(1,2)){
      
      section_ID[z] = z
      filename[z] = sub(".json", "", file_name)
      
      #print(paste("i: ",i,"j: ",j))
      text[z] = input_json[[i]][[j]]
      if(j == 1){
        is_section[z] = 1
      }else{
        is_section[z] = 0
      }
      z = z + 1
    }
  }
  return(data.frame(filename, section_ID, text, is_section, stringsAsFactors = FALSE))
} # Erstellt Dataframe aus json Dateien mit text und is_section feature

clean_text = function(text){
  # PYTHON: re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower())
  
  text = gsub("[^A-Za-z0-9]", " ", tolower(text))
  return(text)
}                                     # Entfernt Zeichen und konvertiert Strings klein

preprocessing_to_word_level = function(txt_lvl_dataset){
  for(row in 1:length(txt_lvl_dataset$text)){
    
    split_text = paste(unlist(tokenize_ptb(txt_lvl_dataset$text[[row]])),collapse= "|")
    
    txt_lvl_dataset$text[[row]] = split_text
  }
  return(separate_rows(txt_lvl_dataset,"text",sep = "[|]"))
}         # Bricht Textdataset in einzelne Wörter/Zeichen herunter

progress = function(context, file_name, z){
  cat("\014")
  print(context)
  print(file_name)
  print(paste(round(z/length(FILES)*100, digits = 2),"%"))
}                      # Gibt % Fortschritt des Preprocessings auf Konsole aus


# Top Level Funktionen
full_preprocess = function(files){
  
  ### PIPELINE
  DATA = list()
  ANZ_MATCHES = data.frame()
  
  z = 0
  for(file_name in files){
    
    progress("(1/12) JSON einlesen: ", file_name, z)
    file_path = paste(JSON_DATA_PATH, "/",file_name,sep="")
    DATA[[file_name]] = fromJSON(paste(readLines(file_path), warn = FALSE,collapse=""))
    
    progress("(2/12) In Textebene zusammenfassen: ", file_name, z)
    DATA[[file_name]] = preprocessing_to_text_level(DATA, file_name)
    
    progress("(3/12) In Satzebene zusammenfassen: ", file_name, z)
    DATA[[file_name]] = filter(DATA[[file_name]], text != "") %>% unnest_sentences("text", "text", output_id = sentence_ID)
    
    progress("(4/12) In Wortebene zusammenfassen: ", file_name, z)
    DATA[[file_name]] = preprocessing_to_word_level(DATA[[file_name]]) %>% rowid_to_column(., "ID")
    
    progress("(5/12) Labels einfügen: ", file_name, z)
    DATA[[file_name]] = label_matching(DATA[[file_name]]) %>% filter(.,text != "")
    
    progress("(7/12) ANZ_MATCHES berechnen: ", file_name, z)
    ANZ_MATCHES = bind_rows(ANZ_MATCHES, summarise(DATA[[file_name]], anz_labels = sum(label), file_name = file_name))
    
    progress("(8/12)Features einfügen: ", file_name, z)
    DATA[[file_name]]$isFirstUpper = sapply(DATA[[file_name]]$text, is_upper)
    nUpper = sapply(DATA[[file_name]]$text, count_upper)
    DATA[[file_name]]$nUpper = nUpper
    nChar = sapply(DATA[[file_name]]$text, nchar)
    DATA[[file_name]]$nChar = nChar
    DATA[[file_name]] = DATA[[file_name]] %>% mutate(., upperFeature = round(nUpper/nChar*3, digits = 0))
    
    progress("(9/12)In Sätze zusammenfassen: ", file_name, z)
    DATA[[file_name]] = DATA[[file_name]] %>% 
      group_by(sentence_ID, section_ID) %>% 
      summarise(
        nWords = n(), 
        text = paste(text, collapse = " "), 
        labels = paste(label, collapse = " "), 
        nLabels = sum(label == 1),
        isFirstUpper = paste(isFirstUpper, collapse = " "),
        nUpper = paste(nUpper, collapse = " "),
        nChar = paste(nChar, collapse = " "),
        upperFeature = paste(upperFeature, collapse = " ")
      )%>% 
      rowid_to_column(., "ID")
    
    progress("(10/12)POS Annotations: ", file_name, z)
    sent_token_annotator = Maxent_Sent_Token_Annotator()
    word_token_annotator = Maxent_Word_Token_Annotator()
    pos_tag_annotator = Maxent_POS_Tag_Annotator()
    DATA[[file_name]]$POS = ""
    text_vector = DATA[[file_name]]$text
    result = lapply(text_vector, annotate,list(sent_token_annotator, word_token_annotator,pos_tag_annotator))
    
    for(sentence in seq_along(result)){
      annotations = subset(result[[sentence]], type == "word")
      tag_list = sapply(annotations$features, "[[", "POS")
      #print(paste(tag_list,collapse = " "))
      DATA[[file_name]]$POS[sentence] = paste(tag_list,collapse = " ")
    }
    
    progress("(11/12) write filename in rows: ", file_name, z)
    DATA[[file_name]] = DATA[[file_name]] %>% mutate(., filename = sub(".json", "", file_name))
    
    progress("(12/12) Mask Numbers: ", file_name, z)
    DATA[[file_name]] = DATA[[file_name]] %>% mutate(text = rm_number(text, replacement = "NUMBER"))
    
    z = z + 1
    gc()
  }
  DATA = bind_rows(DATA)
  return(list(DATA, ANZ_MATCHES))
}                               # Führt Preprocessing bis auf Stufe Satzebene durch

EDA_preprocess = function(files){
  
  seperate_in_words = function(dataset){
    dataset = mutate(dataset, count_POS = lengths(strsplit(POS, " "))) %>% filter(., count_POS == nWords) # Entfernen von unsauberen Zeilen
    dataset = dataset %>% separate_rows(., c("text","labels", "isFirstUpper", "nUpper", "nChar", "upperFeature", "POS"),sep = " ")
    dataset$count_POS = NULL
    dataset$ID = NULL
    dataset = dataset %>% group_by(sentence_ID, filename, section_ID) %>% mutate(word_ID = row_number(sentence_ID)) 
    return(dataset)
  } # Bricht Sätze in Wörter auf

  liste = list()
  z = 0
  set.seed(z+1)
  for(file in files){
    
    file_name = sub(".json", "", file)
    
    progress("Subseting... ", file_name, z)
    dataset = filter(DATA, filename == file_name)
    
    # Im Verhältnis 1:1 gelabelte Sätze und ungelabelte Sätze verwenden
    LABELED_ROWS = filter(dataset, nLabels >0)
    UNLABELED_ROWS = filter(dataset, nLabels == 0)
    
    if(nrow(LABELED_ROWS) >= nrow(UNLABELED_ROWS)){                                         # Ausnahme: Ein Datensatz besteht nur aus Sätzen mit Labels, wird übersprungen
      progress("Separating... ", file_name, z)
      liste[[file_name]] = LABELED_ROWS
      z = z + 1
      next
    }
    
    UNLABELED_ROWS = UNLABELED_ROWS[sample(nrow(UNLABELED_ROWS), 1 * nrow(LABELED_ROWS)),] # Gleicher Anteil Gelabelte und ungelabelte Sätze
    
    
    dataset = bind_rows(LABELED_ROWS,UNLABELED_ROWS)
    
    progress("Separating... ", file_name, z)
    liste[[file_name]] = seperate_in_words(dataset)
    z = z + 1
  }
  return(bind_rows(liste))
}                                # Führt Preprocessing für EDA_DATA (Wortebene mit allen Features) durch



##### DATEIPFADE #####
WORKING_DIRECTORY = "C:/Users/morit/OneDrive/UNI/SS21/BA/workspace_2" #TODO: Eingabe
setwd(WORKING_DIRECTORY)

DATA_ROOT_PATH = "./Data"
JSON_DATA_PATH = paste(DATA_ROOT_PATH,"/DATA",sep = "")
LABELS_PATH = paste(DATA_ROOT_PATH, "/LABELS.csv", sep = "")

##### MAIN #####
#### Slicing #####

# Alle
files_in_folder = list.files(JSON_DATA_PATH)
id_of_files = sub(".json", "", files_in_folder)

#TODO: opt. nur Teil der DAten einlesen
indizes = 1:30#length(files_in_folder)
FILES = files_in_folder[indizes]
ID = id_of_files[indizes]
#### DATA (EDA & BiLSTM) #####

# Daten neu pipen
LABELS = read.csv2(LABELS_PATH, header = TRUE, sep = ",")
result = full_preprocess(FILES)

DATA = result[[1]]         # Datensatz B.
ANZ_MATCHES = result[[2]]

result = NULL

UNMATCHED_FILES = unique((ANZ_MATCHES %>% filter(anz_labels == 0))$file_name)
UNMATCHED_ID = sub(".json", "", UNMATCHED_FILES)

MATCHED_FILES = setdiff(FILES, UNMATCHED_FILES)
MATCHED_ID = sub(".json", "", MATCHED_FILES)

## TRAIN / TEST / VALIDATE Split (BiLSTM) #####

set.seed(1)
TRAIN_FILES = sample(MATCHED_FILES, round(0.8 * length(MATCHED_FILES), digits = 0))
TRAIN_ID = sub(".json", "", TRAIN_FILES)

uebrige = setdiff(MATCHED_FILES, TRAIN_FILES)
VAL_FILES = sample(uebrige, round(0.5 * length(uebrige), digits = 0))
VAL_ID = sub(".json", "", VAL_FILES)

TEST_FILES = setdiff(uebrige, VAL_FILES)
TEST_ID = sub(".json", "", TEST_FILES)

TEST = filter(DATA, filename %in% TEST_ID)
TRAIN = filter(DATA, filename %in% TRAIN_ID)
VALIDATE = filter(DATA, filename %in% VAL_ID)

## Wortebene (EDA) #####
EDA_DATA = EDA_preprocess(FILES)
