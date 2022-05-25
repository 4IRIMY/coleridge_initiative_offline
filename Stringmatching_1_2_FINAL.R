##### WD #####
setwd(WORKING_DIRECTORY)

##### LIBRARYS #####

library(rjson)
library(tidyr)
library(dplyr)
library(stringr)
library(tidyverse)
library(tokenizers)
library(ggplot2)
library(rjson)
library(lexRankr)


##### FUNKTIONEN #####

micro_fbeta = function(tp,fp,fn,b = 0.5){
  top = (1 + b^2) * tp
  bottom = (1 + b^2) * tp + b^2 * fn + fp
  return(top / bottom)
}                                     # berechnet F-0.5 Score

jaccard = function(yt_clean, yp_clean){ # erwartet Vektoren mit einzelnen Wörtern
  
  a = sapply(yt_clean, tolower)
  b = sapply(yp_clean, tolower)
  c = generics::intersect(a,b)
  return(as.double(as.double(length(c)) / (length(a) + length(b) - length(c))))
}                                       # berechnet Jaccard Score

clean_text = function(text){
  # PYTHON: re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower())
  
  text = gsub("[^A-Za-z0-9]", " ", tolower(text))
  return(text)
}                                                  # Bereinigen vom Text entsprechend des Kaggle Wettkampfes: https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/overview/evaluation 

string_matching = function(file, to_be_tested_labels){
  
  file_name = sub(".json", "", file)
  
  wrd_lvl_dataset = wrd_lvl_preprocess(file)[[file]] %>% mutate(label_text = clean_text(text))  # Daten auf Wortebene herunterbrechen
  
  result = c()                                                                                       # Lösungsmenge: Hier werden identifizierte Datensatznamen gespeichert
  
  for(label in to_be_tested_labels){ 
    
    label_words = unlist(tokenize_ptb(label))                                                        # Zu testendes Labeltoken in Woerter aufteilen 
    
    laenge = length(label_words)                                                                     # Anzahl der Woerter, aus welchem der Datensatzname besteht
    
    cleaned_label_words = clean_text(label_words)                                                    # entsprechend des Kaggle Wettkampfes Datensatzname bereinigen
    
    results_for_first_label_word = filter(wrd_lvl_dataset, label_text == cleaned_label_words[1])
    start_ids = results_for_first_label_word$ID
    
    # Testen ob Label == Reihe im Datensatz
    for(start_id in start_ids){
      to_be_tested_indizes = seq(start_id,start_id + laenge - 1)                                     # entsprechend laenge des Tokens die naechsten indizes nach start_id pruefen
      
      to_be_tested_data = filter(wrd_lvl_dataset, ID %in% to_be_tested_indizes)$label_text           # Filtern des Datensatzes nach der zu ueberpruefenden Reihe
      
      is.same = isTRUE(all.equal(to_be_tested_data, cleaned_label_words))                            # Entspricht die Reihe dem Datensatztoken?
      
      if(is.same){
        result = c(result,label)                                                                     # Falls Reihe dem Datensatznamen entspricht, in Loesungsmenge aufnehmen
        break
      }
    }
  }
  
  return(result)
  
}                        # Untersucht Text auf bekannte Datensatznamen und gibt diese zurück

predict_me_str_mtch = function(file, to_be_tested_labels){
  
  # Hilfsfunktionen
  
  sort_list = function(liste){
    for(i in seq_along(liste)){
      liste[[i]] = sort(liste[[i]])
    }
    return(liste)
  }                            # sortiert Vektoren innerhalb der Liste
  
  append_sequenced_predictions = function(liste){
    
    return(lapply(liste,paste, collapse = " "))
  }         # Erstellt aus einzelnen Wörtern in Liste zusammenhengenden String
  
  evaluate_me = function(yp_sorted, yt_sorted){ # erwartet 2 Strings (nicht einzelne wörter)
    
    unmatched_pred = yp_sorted
    unmatched_gt = yt_sorted
    
    tp = 0
    fp = 0
    fn = 0
    
    for(ground_truth in unmatched_gt){
      jaccard_scores = c()
      if(length(unmatched_pred) == 0){
        fn = fn + length(unmatched_gt)
        break
      }
      for(prediction in unmatched_pred){
        split_pred = str_split(prediction, " ")
        split_true = str_split(ground_truth, " ")
        jaccard_scores = c(jaccard_scores, jaccard(split_pred, split_true))
      }
      if(max(jaccard_scores) >= 0.5){
        unmatched_pred = unmatched_pred[-which.max(jaccard_scores)]
        unmatched_gt = unmatched_gt[-which.max(jaccard_scores)]
        tp = tp + 1
      }else{
        fn = fn + length(unmatched_gt)
        break
      }
    }
    fp = fp + length(unmatched_pred[unmatched_pred != ""])
    return(list(tp,fp,fn))
  }           # Evaluiert TP, FP, FN anhand der yp und yt
  
  
  # Predictions
  
  yp_raw = string_matching(file, to_be_tested_labels)
  
  yp_clean = lapply(yp_raw, clean_text)                       # alles klein, satzzeichen weg
  
  yp_clean = unique(yp_clean)                                 # Nur einmal label vorhersagen, falls mehrmals das gleiche gefunden wurde
  
  yp_clean = unlist(yp_clean)
  
  
  # Labeltexte
  
  yt_sorted = filter(LABELS, Id == sub(".json", "", file))$cleaned_label # Datensatznamen der aktuellen Publikation einlesen
  
  
  # Evaluierung
    
  result = evaluate_me(yp_clean,yt_sorted)
  
  # Ergebnis zusammenfassen
  RETURN = data.frame(
    tp = result[[1]], 
    fn = result[[3]], 
    fp = result[[2]], 
    yt = paste(yt_sorted, collapse = "|"), 
    yp = paste(yp_clean, collapse = "|"),
    file_name = file
  )
  return(RETURN)
}                    # Führt string_matching aus, evaluiert TP, FP, FN und erstellt Ergebnistabelle

wrd_lvl_preprocess = function(files){
  
  ### PIPELINE
  DATA = list()
  ANZ_matches = data.frame()
  
  z = 0
  for(file_name in files){
    
    #progress("(1/11) JSON einlesen: ", file_name, z)
    file_path = paste(JSON_DATA_PATH, "/",file_name,sep="")
    DATA[[file_name]] = fromJSON(paste(readLines(file_path), warn = FALSE,collapse=""))
    
    #progress("(2/11) In Textebene zusammenfassen: ", file_name, z)
    DATA[[file_name]] = preprocessing_to_text_level(DATA, file_name)
    
    
    #progress("(3/11) In Satzebene zusammenfassen: ", file_name, z)
    DATA[[file_name]] = filter(DATA[[file_name]], text != "") %>% unnest_sentences("text", "text")
    
    #progress("(4/11) In Wortebene zusammenfassen: ", file_name, z)
    DATA[[file_name]] = preprocessing_to_word_level(DATA[[file_name]]) %>% rowid_to_column(., "ID")
    
    #progress("(11/11) write filename in rows: ", file_name, z)
    #DATA[[file_name]] = DATA[[file_name]] %>% mutate(., filename = sub(".json", "", file_name))
    
    z = z + 1
    gc()
  }
  #DATA = bind_rows(DATA)
  return(DATA)
}                                         # generiert Datensatz bis zur Wortebene (ohne alle Features)
    
    

    
### EVALUIERUNG A) TESTDATEN ##### 

# Labels aus train.csv laden
labels = LABELS %>% filter(., Id %in% TRAIN_ID)                  
to_be_tested_labels = unique(as.character(LABELS$dataset_label)) 

result = tibble()                                                    # Tabelle zum Speichern der Predictions, Labels und Evaluationsgroessen
    
z = 1
for(file in TEST_FILES){
      
  # Predicitons erzeugen und berechnen der Evaluationsgroessen 
  result = rbind(result, predict_me_str_mtch(file, to_be_tested_labels))
      
  cat("\014")
  print(paste("File ", z, "/",length(TEST_FILES)))
      
  z = z+1
}

micro_fbeta(sum(result$tp), sum(result$fp), sum(result$fn))         # Güte nur Testdaten

### EVALUIERUNG A) TESTDATEN & UNGEMATCHTE DATEN #####

# Labels aus train.csv laden
labels = LABELS %>% filter(., Id %in% TRAIN_ID)                  
to_be_tested_labels = unique(as.character(LABELS$dataset_label)) 

result_2 = tibble()                                                    # Tabelle zum Speichern der Predictions, Labels und Evaluationsgroessen

z = 1
for(file in UNMATCHED_FILES){
  
  # Predicitons erzeugen und berechnen der Evaluationsgroessen 
  result_2 = rbind(result_2, predict_me_str_mtch(file, to_be_tested_labels))
  
  cat("\014")
  print(paste("File ", z, "/",length(UNMATCHED_FILES)))
  
  z = z+1
}

micro_fbeta(sum(result_2$tp) + sum(result$tp), sum(result_2$fp) + sum(result$tp), sum(result_2$fn) + sum(result_2$fp))       # Güte Testdaten + ungematchte Daten

### UNGEMATCHTE LABELS ###
unmatched_labels = LABELS %>% 
  filter(., Id %in% UNMATCHED_ID) %>%
  group_by(., cleaned_label) %>%
  summarise(., nUnmatched = n())
### EVALUIERUNG B) TESTDATEN #####

# Neue Aufteilung der Daten mit ungematchten LabelsTest
set.seed(2)
TRAIN_FILES = sample(FILES, round(0.8 * length(FILES), digits = 0))
TRAIN_ID = sub(".json", "", TRAIN_FILES)

uebrige = setdiff(FILES, TRAIN_FILES)
VAL_FILES = sample(uebrige, round(0.5 * length(uebrige), digits = 0))
VAL_ID = sub(".json", "", VAL_FILES)

TEST_FILES = setdiff(uebrige, VAL_FILES)
TEST_ID = sub(".json", "", TEST_FILES)

TEST = filter(DATA, filename %in% TEST_ID)
TRAIN = filter(DATA, filename %in% TRAIN_ID)
VALIDATE = filter(DATA, filename %in% VAL_ID)




# Labels aus train.csv laden
labels = LABELS %>% filter(., Id %in% TRAIN_ID)                  
to_be_tested_labels = unique(as.character(LABELS$dataset_label)) 

result_3 = tibble()                                                    # Tabelle zum Speichern der Predictions, Labels und Evaluationsgroessen

z = 1
for(file in TEST_FILES){
  
  # Predicitons erzeugen und berechnen der Evaluationsgroessen 
  result_3 = rbind(result, predict_me_str_mtch(file, to_be_tested_labels))
  
  cat("\014")
  print(paste("File ", z, "/",length(TEST_FILES)))
  
  z = z+1
}

micro_fbeta(sum(result_3$tp), sum(result_3$fp), sum(result_3$fn))