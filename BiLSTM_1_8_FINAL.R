##### WD #####
setwd(WORKING_DIRECTORY)
######### UTILITY - NÜTZLICHE BEFEHLE - NUR BEI BEDARF #####
##### SPEICHERN #####
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  sample_weight_mode="temporal"
)                                                   # Modell muss neu compiliert werden, da sonst speicherung nicht moeglich ist
model %>% save_model_hdf5("./model.h5")
model = NULL
save_text_tokenizer(TOKENIZER, "text_tokenizer_50000")
save_text_tokenizer(TOKENIZER_2, "POS_tokenizer_50000")
TOKENIZER = NULL
TOKENIZER_2 = NULL
save.image("overfitted_BiLSTM_Data.RData")
##### LADEN #####
load("BiLSTM_Data.RData")

model = load_model_hdf5("./Testdurchlauf_1/model_dropout_0_0.5.h5")
TOKENIZER = load_text_tokenizer("text_tokenizer_50000")
TOKENIZER_2 = load_text_tokenizer("POS_tokenizer_50000")
######### NN - Daten Preprocessing Teil 2 ###################
##### LIBRARYS #####
library(reticulate)
use_condaenv("ba_env", required = TRUE) #TODO: Eingabe
library(tensorflow)
library(keras)
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


##### FUNKTIONEN #####

pad_x_sequences = function(text, tokenizer){
  tokenized_text = tokenizer$texts_to_sequences(text)
  return(keras::pad_sequences(tokenized_text, MAX_LEN, padding = PADDING, truncating = TRUNCATING))
}                           # Wandelt Text in Indizes des Tokenizers um und schneidet Sätze ab / fuellt nullen auf

pad_x_feature = function(text){
  splitted = str_split(text, pattern = " ")
  return(pad_sequences(splitted, MAX_LEN, padding = PADDING, truncating = TRUNCATING))
}                                        # Schneidet Sätze von Features ab / Fuellt nullen auf

pad_y_sequences = function(labels){
  y_train_sequence = lapply(str_split(labels, " "), as.integer)
  return(pad_sequences(y_train_sequence, MAX_LEN, padding = PADDING, truncating = TRUNCATING))
}                                    # Schneidet "Saetze" von Labels ab / Fuellt nullen auf

create_subsequences = function(y_pos){
  result = list()
  stop = FALSE
  i = 1
  while(TRUE){
    if(length(y_pos) == 0){
      break
    }
    
    result[[i]] = y_pos[1]
    y_pos = y_pos[-1]
    
    while(TRUE){
      if(length(y_pos) == 0){
        break
      }
      if(result[[i]][length(result[[i]])] + 1 == y_pos[1]){
        result[[i]] = cbind(result[[i]],y_pos[1])
        y_pos = y_pos[-1]
      }else{
        break
      }
    }
    #print(i)
    i = i + 1
  }
  return(result)
}                                 # Erstellt aus Positionen wo yhat = 1 ist reihen, sofern die folgenden Positionen auch yhat = 1 sind

create_sample_weights = function(y_train, weight_0, weight_1){
  data = y_train
  data[which(y_train == 0)] = weight_0
  data[which(y_train == 1)] = weight_1
  return(data)
}         # Erstellt für jedes y Gewichte fuer den Weighted Loss (Class Imbalance)

micro_fbeta = function(tp,fp,fn,b = 0.5){
  top = (1 + b^2) * tp
  bottom = (1 + b^2) * tp + b^2 * fn + fp
  return(top / bottom)
}                              # Berechnet F-Beta Score

evaluate_me = function(yp_sorted, yt_sorted){
  
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
}                          # Berechnet TP, FP, FN, analog Stringmatching

predict_me = function(x_features, threshold, file, yt_sorted){
  
  
  clean_text = function(text){
    # PYTHON: re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower())
    
    text = gsub("[^A-Za-z0-9]", " ", tolower(text))
    
    return(text)
  }                                                # Bereinigt Text
  
  jaccard = function(yt_clean, yp_clean){ 
    
    a = sapply(yt_clean, tolower)
    b = sapply(yp_clean, tolower)
    c = generics::intersect(a,b)
    return(as.double(as.double(length(c)) / (length(a) + length(b) - length(c))))
  }                                     # Berechnet Jaccard Score auf Basis 2er Strings
  
  convert_pos_to_text = function(preprocessed_sequence, pos_of_predictions){
    positive_predictions = t(preprocessed_sequence)[pos_of_predictions] #Transpose damit reihenweise indizierung und nicht spaltenweise
    text = TOKENIZER$index_word[unlist(positive_predictions)]
    return(text)
  }  # Übersetzt Positionsvektoren zu Text zurück
  
  get_pos_of_classified = function(x_features, threshold){
    raw_prediction = model$predict(x_features)
    pos_of_prediction = which(t(raw_prediction[,,1]) > threshold)
    return(pos_of_prediction)
  }                    # Erstellt Prognosevektoren (Indiziert) und gibt Positionen der labels wo yhat = 1 zurück
  
  make_subsequenced_texts = function(x_preprocessed, subsequences){               # Erstellt aus prognostizierten Vokabularindexreihen Tokens (durch Rueckuebersetzung)
    
    yp_text = list()
    for(i in seq_along(subsequences)){
      yp_text[[i]] = convert_pos_to_text(x_preprocessed, as.vector(subsequences[[i]]))
    }
    
    return(yp_text)
  }
  
  clean_y = function(y_text){
    result = list()
    for(i in seq_along(y_text)){
      result[[i]] = clean_text(as.vector(y_text[i][[1]]))
      if(any(result[[i]] == " ")){
        result[[i]] = result[[i]][-which(result[[i]] == " ")]
      }
    }
    result = unique(result)
    return(result)
  }                                                 # Bereinigt Prognosevektoren
  
  sort_list = function(liste){
    for(i in seq_along(liste)){
      liste[[i]] = sort(liste[[i]])
    }
    return(liste)
  }                                                # Sortiert Prognosen alphabetisch
  
  append_sequenced_predictions = function(vektor){
    
    return(sapply(vektor,paste, collapse = " "))
  }                            # Hängt Prognosetexte aneinander, welche als Vektoren vorliegen 
  
  
  x_text = x_features[[1]]
  
  yp_pos = get_pos_of_classified(x_features, threshold = threshold)
  
  yp_subsequenced = create_subsequences(yp_pos)
  
  yp_text = make_subsequenced_texts(x_text, yp_subsequenced)
  
  yp_clean = clean_y(yp_text)
  
  yp_appended = append_sequenced_predictions(yp_clean)
  
  yp_unique = unique(yp_appended)                                        # Jede Prediction nur einmal (wegen satzweiser Auswertung)
  
  yp_sorted = sort_list(yp_unique)
  
  # Whitespaces entfernen
  yp_sorted = gsub("\\s+"," ",yp_sorted)                                 
  yp_sorted = yp_sorted[yp_sorted != " "]
  
  measures = evaluate_me(yp_sorted, yt_sorted)
  tp = measures[[1]]
  fp = measures[[2]]
  fn = measures[[3]]
    
  RETURN = data.frame(
    tp = tp, 
    fn = fn, 
    fp = fp, 
    yt = paste(yt_sorted, collapse = "|"), 
    yp = paste(yp_sorted, collapse = "|"),
    yp_pos = paste(yp_pos, collapse = "|"),
    yp_raw = paste(yp_text, collapse = "|"),
    file_name = file
  )
  return(RETURN)
}         # Erstellt Prognosen aus dem BiLSTM Modell

create_model = function(in_drop, out_drop){
  

  
  ##### MODELL #####
  ### PARAMETER #####
  dimPOSEmbedding = 25
  dimWordEmbedding = 50
  dimIsUpper = 1
  dimUpperFeature = 2
  dropoutInput = in_drop
  dropoutOutput = out_drop
  dimHiddenLSTM = 18
  ### ARCHITEKTUR #####
  
  # INPUT & EMBEDDINGS #####
  text_input = 
    layer_input(
      shape = c(MAX_LEN),
      dtype = "int32",
      name = "text"
    ) 
  
  text_embedding = 
    layer_embedding(
      input_dim = dimVocabulary + 1, 
      output_dim = dimWordEmbedding, 
      input_length = MAX_LEN
    ) 
  
  POS_input =
    layer_input(
      shape = c(MAX_LEN),
      dtype = "int32",
      name = "POS"
    )
  
  POS_embedding = 
    layer_embedding(
      input_dim = dimPOSVocabulary + 1, 
      output_dim = dimPOSEmbedding, 
      input_length = MAX_LEN
    )
  
  is_uppercase_input = 
    layer_input(
      shape = c(MAX_LEN),
      dtype = "int32",
      name = "is_uppercase"
    ) 
  
  is_uppercase_embedding =
    layer_embedding(
      input_dim = 2,
      output_dim = dimIsUpper,
      input_length = MAX_LEN
    )
  
  upperFeature_input = 
    layer_input(
      shape = c(MAX_LEN),
      dtype = "int32",
      name = "upperFeature"
    ) 
  
  upperFeature_embedding =
    layer_embedding(
      input_dim = 4,
      output_dim = dimUpperFeature,
      input_length = MAX_LEN
    )
  
  # Zusammenhängen
  text = text_input %>% text_embedding %>%
    layer_dropout(
      noise_shape = c(1),
      rate = dropoutInput
    )
  is_uppercase = is_uppercase_input %>% is_uppercase_embedding
  POS = POS_input %>% POS_embedding
  upperFeature = upperFeature_input %>% upperFeature_embedding
  
  # CONCATENATE #####
  concatenated_embeddings = layer_concatenate(
    list(text, is_uppercase, POS, upperFeature),
    axis = 2
  )
  
  # BiLSTM & DENSE #####
  Network = concatenated_embeddings %>% 
    bidirectional(
      layer_lstm(
        units = dimHiddenLSTM, 
        return_sequences = TRUE
      ) 
    ) %>%
    layer_dropout(
      rate = dropoutOutput 
    )%>%
    time_distributed(
      layer_dense(
        units = 1, 
        activation = "sigmoid"
      )
    ) 
  
  model = keras_model(list(text_input, is_uppercase_input, POS_input, upperFeature_input), Network)
  
  summary(model)
  
  
  model %>% compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = c(
      f_beta_0.9,
      f_beta_0.975,
      f_beta_0.99,
      f_beta_0.995,
      f_beta_0.999
      #f_beta_0.9999
      #metric_false_negatives(thresholds = tshld), 
      #metric_false_positives(thresholds = tshld), 
      #metric_true_positives(thresholds = tshld)
    ), 
    sample_weight_mode="temporal"
  )
  return(model)
}                            # ERSTELLT BILSTM - Modell

progress = function(context, file_name, z, files){
  cat("\014")
  print(context)
  print(file_name)
  print(paste(round(z/length(files)*100, digits = 2),"%"))
}                      # Gibt % Fortschritt des Preprocessings auf Konsole aus

evaluate_BiLSTM = function(ID, Datensatz, threshold, return_resultset = FALSE){
  result = data.frame()
  #names[test_indizes[10:20]]
  z = 0
  for(name in ID){
    
    data = filter(Datensatz, filename == name)
    
    # Text
    x_text_ = pad_x_sequences(data$text, TOKENIZER)
    
    # isUpper
    x_isUpper_ = pad_x_feature(data$isFirstUpper)
    
    # POS
    x_pos_ = pad_x_sequences(data$POS, TOKENIZER_2)
    
    # y-Labels
    yt_sorted = sort(filter(LABELS, Id == name)$cleaned_label)
    
    #upperFeature
    upperFeature_ = pad_x_feature(data$upperFeature)
    
    result = bind_rows(result, predict_me(list(x_text_,x_isUpper_,x_pos_,upperFeature_), threshold, name, yt_sorted))
    z = z + 1
    progress("Evaluiere...", name, z, ID)
  }
  if(!return_resultset){
    return(print(micro_fbeta(sum(result$tp),sum(result$fp),sum(result$fn))))
  }else{
    return(result)
  }
  
} # Wertet BiLSTM Modell bezügich der gegebenen Dateien in ID aus

##### Preprocessing #####
### TOKENIZER #####

dimVocabulary = 50000 

TOKENIZER = keras::text_tokenizer(
  num_words = dimVocabulary,
  filters = "" ,           #"!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n",
  lower = TRUE,
  split = " ",
  char_level = FALSE,
  oov_token = "OOV"
)

TOKENIZER = fit_text_tokenizer(TOKENIZER, TRAIN$text)                   # Erstellt Vokabular fuer Text Feature

TOKENIZER_2 = keras::text_tokenizer(
  num_words = NULL,
  filters = "" ,           #"!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n",
  lower = FALSE,
  split = " ",
  char_level = FALSE,
  oov_token = "OOV"
)

TOKENIZER_2 = fit_text_tokenizer(TOKENIZER_2, TRAIN$POS)                # Erstellt Vokabular fuer POS Feature

dimPOSVocabulary = length(TOKENIZER_2$index_word)


### PADDING PARAMETER#####

MAX_LEN = 50        #max Wörter pro Satz
PADDING = "post"    #wo auffüllen
TRUNCATING = "post" #wo abschneiden

### UNDERSAMPLING #####
verhaeltnis_unlabeled_labeled = 2
set.seed(1)

LABELED_ROWS = filter(TRAIN, nLabels >0)
UNLABELED_ROWS = filter(TRAIN, nLabels == 0)
UNLABELED_ROWS = UNLABELED_ROWS[sample(nrow(UNLABELED_ROWS), verhaeltnis_unlabeled_labeled * nrow(LABELED_ROWS)),]
TRAIN = bind_rows(LABELED_ROWS,UNLABELED_ROWS)

LABELED_ROWS = filter(VALIDATE, nLabels >0)
UNLABELED_ROWS = filter(VALIDATE, nLabels == 0)
UNLABELED_ROWS = UNLABELED_ROWS[sample(nrow(UNLABELED_ROWS), verhaeltnis_unlabeled_labeled * nrow(LABELED_ROWS)),]
VALIDATE = bind_rows(LABELED_ROWS,UNLABELED_ROWS)
### PADDING #####
## X - TRAIN #####
# TEXT
x_text = pad_x_sequences(TRAIN$text, TOKENIZER)

# isUpper
x_isUpper = pad_x_feature(TRAIN$isFirstUpper)

# POS
x_pos = pad_x_sequences(TRAIN$POS, TOKENIZER_2)

# upperFeature
x_upperFeature = pad_x_feature(TRAIN$upperFeature)

## X - Validation #####
# TEXT
val_text = pad_x_sequences(VALIDATE$text, TOKENIZER)

# isUpper
val_isUpper = pad_x_feature(VALIDATE$isFirstUpper)

# POS
val_pos = pad_x_sequences(VALIDATE$POS, TOKENIZER_2)

# upperFeature
val_upperFeature = pad_x_feature(VALIDATE$upperFeature)

## Y - TRAIN #####
y_train = pad_y_sequences(TRAIN$labels)
## Y - Validation #####
y_val = pad_y_sequences(VALIDATE$labels)

### Weighted loss #####

## TRAIN #####
train_ratio = mean(as.vector(y_train))
train_SAMPLE_WEIGHTS = create_sample_weights(y_train, train_ratio, 1)
print(train_ratio)



## VALIDATE #####
val_ratio = mean(as.vector(y_val))
val_SAMPLE_WEIGHTS = create_sample_weights(y_val, val_ratio, 1)
print(val_ratio)

### 3D umwandlung labels #####

# Labels muessen fuer keras wegen sequentieller Daten 3 Dimensionen haben
y_train_3d = array(y_train, c(length(y_train[,1]), MAX_LEN, 1))          
y_val_train_3d = array(y_val, c(length(y_val[,1]), MAX_LEN, 1))          

dim(y_train_3d)
dim(y_val_train_3d)
##### Measures - NICHT TOKENBASIERT - NUR ZUM BEOBACHTEN #####
#https://towardsdatascience.com/f-beta-score-in-keras-part-i-86ad190a252f
f_beta_0.9 = custom_metric("f_beta_0.9", function(y_true, y_pred,beta = 0.5, threshold = 0.9, epsilon = 1e-7){
  
  beta_squared = beta^2
  
  y_true = tf$cast(y_true, tf$float32)
  y_pred = tf$cast(y_pred, tf$float32)
  
  y_pred = tf$cast(tf$greater_equal(y_pred, tf$constant(threshold)), tf$float32)
  
  tp = tf$reduce_sum(y_true * y_pred)
  pred_p = tf$reduce_sum(y_pred)
  act_p = tf$reduce_sum(y_true)
  
  precision = tp/(pred_p+epsilon)
  recall = tp/(act_p+epsilon)
  
  return((1+beta_squared)*precision*recall / (beta_squared*precision + recall + epsilon))
  
})
f_beta_0.975 = custom_metric("f_beta_0.975", function(y_true, y_pred,beta = 0.5, threshold = 0.975, epsilon = 1e-7){
  
  beta_squared = beta^2
  
  y_true = tf$cast(y_true, tf$float32)
  y_pred = tf$cast(y_pred, tf$float32)
  
  y_pred = tf$cast(tf$greater_equal(y_pred, tf$constant(threshold)), tf$float32)
  
  tp = tf$reduce_sum(y_true * y_pred)
  pred_p = tf$reduce_sum(y_pred)
  act_p = tf$reduce_sum(y_true)
  
  precision = tp/(pred_p+epsilon)
  recall = tp/(act_p+epsilon)
  
  return((1+beta_squared)*precision*recall / (beta_squared*precision + recall + epsilon))
  
})
f_beta_0.99 = custom_metric("f_beta_0.99", function(y_true, y_pred,beta = 0.5, threshold = 0.99, epsilon = 1e-7){
  
  beta_squared = beta^2
  
  y_true = tf$cast(y_true, tf$float32)
  y_pred = tf$cast(y_pred, tf$float32)
  
  y_pred = tf$cast(tf$greater_equal(y_pred, tf$constant(threshold)), tf$float32)
  
  tp = tf$reduce_sum(y_true * y_pred)
  pred_p = tf$reduce_sum(y_pred)
  act_p = tf$reduce_sum(y_true)
  
  precision = tp/(pred_p+epsilon)
  recall = tp/(act_p+epsilon)
  
  return((1+beta_squared)*precision*recall / (beta_squared*precision + recall + epsilon))
  
})
f_beta_0.995 = custom_metric("f_beta_0.995", function(y_true, y_pred,beta = 0.5, threshold = 0.995, epsilon = 1e-7){
  
  beta_squared = beta^2
  
  y_true = tf$cast(y_true, tf$float32)
  y_pred = tf$cast(y_pred, tf$float32)
  
  y_pred = tf$cast(tf$greater_equal(y_pred, tf$constant(threshold)), tf$float32)
  
  tp = tf$reduce_sum(y_true * y_pred)
  pred_p = tf$reduce_sum(y_pred)
  act_p = tf$reduce_sum(y_true)
  
  precision = tp/(pred_p+epsilon)
  recall = tp/(act_p+epsilon)
  
  return((1+beta_squared)*precision*recall / (beta_squared*precision + recall + epsilon))
  
})
f_beta_0.999 = custom_metric("f_beta_0.999", function(y_true, y_pred,beta = 0.5, threshold = 0.999, epsilon = 1e-7){
  
  beta_squared = beta^2
  
  y_true = tf$cast(y_true, tf$float32)
  y_pred = tf$cast(y_pred, tf$float32)
  
  y_pred = tf$cast(tf$greater_equal(y_pred, tf$constant(threshold)), tf$float32)
  
  tp = tf$reduce_sum(y_true * y_pred)
  pred_p = tf$reduce_sum(y_pred)
  act_p = tf$reduce_sum(y_true)
  
  precision = tp/(pred_p+epsilon)
  recall = tp/(act_p+epsilon)
  
  return((1+beta_squared)*precision*recall / (beta_squared*precision + recall + epsilon))
  
})
f_beta_0.9999 = custom_metric("f_beta_0.9999", function(y_true, y_pred,beta = 0.5, threshold = 0.9999, epsilon = 1e-7){
  
  beta_squared = beta^2
  
  y_true = tf$cast(y_true, tf$float32)
  y_pred = tf$cast(y_pred, tf$float32)
  
  y_pred = tf$cast(tf$greater_equal(y_pred, tf$constant(threshold)), tf$float32)
  
  tp = tf$reduce_sum(y_true * y_pred)
  pred_p = tf$reduce_sum(y_pred)
  act_p = tf$reduce_sum(y_true)
  
  precision = tp/(pred_p+epsilon)
  recall = tp/(act_p+epsilon)
  
  return((1+beta_squared)*precision*recall / (beta_squared*precision + recall + epsilon))
  
})
########## TRAINING UND TESTEN #####################

##### 7.3. Unregularisiertes Modell #####

input_dropouts = c(0)
output_dropouts = c(0)

result = data.frame()
for(in_drop in input_dropouts){
  
  for(out_drop in output_dropouts){
    model = create_model(in_drop, out_drop)
    history = history = model %>% fit(
      list(x_text, x_isUpper, x_pos, x_upperFeature), y_train_3d,
      epochs = 50,
      batch_size = 1024, 
      shuffle = TRUE,
      validation_data = list(list(val_text, val_isUpper, val_pos, val_upperFeature), y_val_train_3d, val_SAMPLE_WEIGHTS),
      sample_weight = train_SAMPLE_WEIGHTS
    )
    
    model_name = paste("./model_dropout_",in_drop,"_",out_drop,".h5", sep = "")
    Data = data.frame(
      model_name = rep(model_name,times = length(history$params$epochs)),
      epoch = 1:history$params$epochs,
      loss = history$metrics$loss, 
      val_loss = history$metrics$val_loss
    )
    result = bind_rows(result, Data)
    
    model %>% compile(
      optimizer = "rmsprop",
      loss = "binary_crossentropy",
      sample_weight_mode="temporal"
    )                                                   # Modell muss neu compiliert werden, da sonst speicherung nicht moeglich ist
    model %>% save_model_hdf5(model_name)
  }
}

Metrics = result %>% pivot_longer(., c(loss, val_loss), names_to = "metric_name",values_to = "value")
# Abbildung 27
dev.new()
ggplot(Metrics, aes(x = epoch, y = value, color = metric_name)) + 
  geom_line() + 
  scale_y_continuous(limits = c(0,0.01)) +
  labs(x = "Epoche", y = "Loss", color = "Metrik")


##### 7.6. Optimierung der Dropout Warscheinlichkeiten #####

early_stop = callback_early_stopping(
  monitor = "val_loss",
  restore_best_weights = TRUE,
  patience = 7
)

# Dropout Parameter
input_dropouts = c(0, 0.5, 0.75)
output_dropouts = c(0, 0.5)

# Trainieren der resultierenden 6 Modelle
result = data.frame()
for(in_drop in input_dropouts){
  
  for(out_drop in output_dropouts){
    model = create_model(in_drop, out_drop)
    history = history = model %>% fit(
      list(x_text, x_isUpper, x_pos, x_upperFeature), y_train_3d,
      epochs = 200,
      batch_size = 1024, 
      shuffle = TRUE,
      validation_data = list(list(val_text, val_isUpper, val_pos, val_upperFeature), y_val_train_3d, val_SAMPLE_WEIGHTS),
      sample_weight = train_SAMPLE_WEIGHTS,
      callbacks = early_stop
    )
    
    model_name = paste("./model_dropout_",in_drop,"_",out_drop,".h5", sep = "")
    metrics = history$metrics
    Data = data.frame(
      model_name = rep(model_name,times = length(history$params$epochs)),
      epoch = 1:history$params$epochs,
      loss = metrics$loss, 
      val_loss = metrics$val_loss,
      
      # F-Beta auf Wortebene, NICHT "Wettkampfsmetrik", also kein Jaccard Score zur Bestimmung ob TP, FP oder FN!
      f_beta_0.9 = metrics$f_beta_0.9, 
      f_beta_0.975 = metrics$f_beta_0.975,
      f_beta_0.995 = metrics$f_beta_0.995,
      f_beta_0.999 = metrics$f_beta_0.999
    )
    result = bind_rows(result, Data)
    
    model %>% compile(
      optimizer = "rmsprop",
      loss = "binary_crossentropy",
      sample_weight_mode="temporal"
    )                                                   # Modell muss neu compiliert werden, da sonst speicherung nicht moeglich ist
    model %>% save_model_hdf5(model_name)
  }
}

# Beste Modelle 
beste_modelle = result %>% 
  group_by(., model_name) %>% 
  summarise(., beste_guete = min(val_loss)) %>%
  arrange(., beste_guete)




# Threshold + Güte Schaetzung an 100 Daten 
thresholds = c(0.99, 0.995,0.999,0.9999)

threshold = 0.995

RESULT = data.frame()
for(in_drop in input_dropouts){
  
  for(out_drop in output_dropouts){
    # Modell erstellen
    modellname = paste("model_dropout_",in_drop,"_",out_drop,".h5", sep = "")
    modelpfad = paste("./Testdurchlauf_1/",modellname, sep = "")
    
    model = create_model(in_drop,out_drop)
    model = load_model_hdf5(modelpfad)
    
    # Thresholds testen
    scores = c()
    for(threshold in thresholds){
      score = evaluate_BiLSTM(VAL_ID[1:100], DATA, threshold)
      scores = c(scores, score)
    }
    result = data.frame(
      modellname = rep(modellname, times = length(thresholds)),
      threshold = thresholds,
      scores = scores
    )
    RESULT = bind_rows(RESULT, result)
  }
}

beste_thresholds = RESULT %>% 
  group_by(., modellname, threshold) %>% 
  summarise(., beste_guete = max(scores)) %>%
  arrange(., desc(beste_guete))

# Güte
model = load_model_hdf5("./Testdurchlauf_1/model_dropout_0_0.5.h5")

# EVALUIERUNG A) TESTDATEN
test_result = evaluate_BiLSTM(TEST_ID,DATA, 0.999, return_resultset = TRUE)
micro_fbeta(sum(test_result$tp),sum(test_result$fp),sum(test_result$fn))

# EVALUIERUNG A) UNMATCHED_DATA
unmatched_result = evaluate_BiLSTM(UNMATCHED_ID, DATA, 0.999, return_resultset = TRUE)
micro_fbeta(sum(unmatched_result$tp),sum(unmatched_result$fp),sum(unmatched_result$fn))

# Generalisierungstest
MATCHED_TRAIN_LABELS = LABELS %>% filter(., Id %in% MATCHED_ID, Id %in% TRAIN_ID)
MATCHED_TEST_LABELS = LABELS %>% filter(., Id %in% MATCHED_ID, Id %in% TEST_ID)

train_tokens = unique(MATCHED_TRAIN_LABELS$cleaned_label)
test_tokens = unique(MATCHED_TEST_LABELS$cleaned_label)

unmatched_tokens = setdiff(test_tokens, train_tokens)

# Anzahl TOkens in TEST welche nicht in TRAIN vorkommen
length(unmatched_tokens)