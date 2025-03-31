# Wczytanie bibliotek
library(tidyverse)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(pROC)
library(ggplot2)


# Wczytanie danych
data <- read.csv("heart.csv")

param_grid <- expand.grid(
  maxdepth = c(3, 5, 7, 9, 11),
  minsplit = c(2, 5, 10, 20),
  cp = seq(0.001, 0.1, by = 0.01)
)

# Przekształcenie danych kategorycznych na faktory
data$Sex <- as.factor(data$Sex)
data$ChestPainType <- as.factor(data$ChestPainType)
data$RestingECG <- as.factor(data$RestingECG)
data$ExerciseAngina <- as.factor(data$ExerciseAngina)
data$ST_Slope <- as.factor(data$ST_Slope)
data$HeartDisease <- as.factor(data$HeartDisease)

# Podział na zbiory treningowy i testowy (70% trening, 30% test)
set.seed(123)
trainIndex <- createDataPartition(data$HeartDisease, p = 0.7, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Funkcja do trenowania modelu i obliczania metryk z użyciem cross-walidacji
evaluate_model_cv <- function(params, trainData, folds = 10) {
  # Przygotowanie zbiorów do cross-walidacji
  cv_folds <- createFolds(trainData$HeartDisease, k = folds, list = TRUE)
  
  # Miejsce na metryki
  metrics <- data.frame(Accuracy = numeric(), AUC = numeric(), Precision = numeric(), Recall = numeric())
  
  # Cross-walidacja
  for (fold in cv_folds) {
    train_fold <- trainData[-fold, ]
    test_fold <- trainData[fold, ]
    
    # Trenowanie modelu
    model <- rpart(
      HeartDisease ~ ., 
      data = train_fold, 
      method = "class",
      control = rpart.control(
        maxdepth = as.numeric(params["maxdepth"]),
        minsplit = as.numeric(params["minsplit"]),
        cp = as.numeric(params["cp"])
      )
    )
    
    # Predykcja na zbiorze testowym
    predictions <- predict(model, newdata = test_fold, type = "class")
    probabilities <- predict(model, newdata = test_fold, type = "prob")[, 2]
    
    # Obliczanie metryk
    accuracy <- mean(predictions == test_fold$HeartDisease)
    roc_curve <- roc(test_fold$HeartDisease, probabilities, levels = levels(test_fold$HeartDisease))
    auc_score <- auc(roc_curve)
    cm <- confusionMatrix(predictions, test_fold$HeartDisease)
    precision <- cm$byClass["Precision"]
    recall <- cm$byClass["Recall"]
    
    # Zapis metryk
    metrics <- rbind(metrics, c(Accuracy = accuracy, AUC = auc_score, Precision = precision, Recall = recall))
  }
  
  # Średnie metryki z CV
  colMeans(metrics, na.rm = TRUE)
}

# Zastosowanie funkcji dla całej siatki parametrów
results <- param_grid
metric_results <- t(apply(param_grid, 1, function(row) evaluate_model_cv(as.list(row), trainData)))

# Poprawienie nazw kolumn w wynikowych metrykach
colnames(metric_results) <- c("Accuracy", "AUC", "Precision", "Recall")
results <- cbind(results, metric_results)

# Wyświetlenie wyników
print(results)

# Najlepsze parametry na podstawie AUC (można zmienić na "Accuracy", "Precision", "Recall")
best_params <- results[which.max(results$Accuracy), ]
print(best_params)

# Trenowanie modelu z najlepszymi parametrami na całym zbiorze danych
best_model <- rpart(
  HeartDisease ~ ., 
  data = trainData, 
  method = "class",
  control = rpart.control(
    maxdepth = best_params$maxdepth,
    minsplit = best_params$minsplit,
    cp = best_params$cp
  )
)

# Wizualizacja najlepszego drzewa
rpart.plot(best_model)

# Finalne oceny (jeśli masz zestaw testowy niezależny, możesz go tutaj użyć)
final_predictions <- predict(best_model, newdata = testData, type = "class")
final_probabilities <- predict(best_model, newdata = testData, type = "prob")[, 2]

# Finalna macierz pomyłek
final_conf_matrix <- confusionMatrix(final_predictions, testData$HeartDisease)

# Obliczenie ROC i AUC
roc_curve <- roc(testData$HeartDisease, final_probabilities, levels = levels(testData$HeartDisease))
final_auc <- auc(roc_curve)

# Wyświetlanie wyników
print(final_conf_matrix)
cat("Final AUC: ", final_auc, "\n")

# Konwersja macierzy pomyłek na tabelę i rysowanie heatmapy
conf_matrix <- as.table(final_conf_matrix$table)
conf_matrix_df <- as.data.frame(conf_matrix)

# Tworzenie heatmapy z liczbami
ggplot(data = conf_matrix_df, aes(x = Prediction, y = Reference)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "#f7fbff", high = "#08306b") +
  geom_text(aes(label = Freq), color = "black", size = 5) +
  labs(
    title = "Confusion Matrix Heatmap",
    x = "Predicted Label",
    y = "Actual Label",
    fill = "Frequency"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12)
  )



# Lasy losowe

param_grid <- expand.grid(
  mtry = c(2, 4, 6),          
  ntree = seq(100, 1000, by = 100),  
  nodesize = c(1, 5, 10)      
)

# Funkcja do trenowania modelu i obliczania metryk z cross-walidacją

evaluate_rf_model_cv <- function(params) {
  # Cross-walidacja
  folds <- createFolds(trainData$HeartDisease, k = 5, list = TRUE)
  
  metrics <- lapply(folds, function(fold_idx) {
    # Podział na dane treningowe i walidacyjne
    train_fold <- trainData[-fold_idx, ]
    val_fold <- trainData[fold_idx, ]
    
    # Trenowanie modelu
    model <- randomForest(
      HeartDisease ~ ., 
      data = train_fold,
      mtry = as.numeric(params["mtry"]),
      ntree = as.numeric(params["ntree"]),
      nodesize = as.numeric(params["nodesize"])
    )
    
    # Predykcja na zbiorze walidacyjnym
    val_predictions <- predict(model, newdata = val_fold, type = "response")
    
    # Obliczanie metryk
    val_roc_curve <- roc(val_fold$HeartDisease, as.numeric(val_predictions))
    val_auc <- auc(val_roc_curve)
    cm <- confusionMatrix(val_predictions, val_fold$HeartDisease)
    precision <- cm$byClass["Precision"]
    recall <- cm$byClass["Recall"]
    accuracy <- mean(val_predictions == val_fold$HeartDisease)
    
    return(c(Accuracy = accuracy, AUC = val_auc, Precision = precision, Recall = recall))
  })
  
  # Średnie wyniki cross-walidacji
  mean_metrics <- colMeans(do.call(rbind, metrics))
  return(mean_metrics)
}

# Cross-walidacja dla każdego zestawu hiperparametrów
cv_metrics <- t(apply(param_grid, 1, function(row) evaluate_rf_model_cv(as.list(row))))

# Dodanie wyników do tabeli
results_rf <- cbind(param_grid, cv_metrics)

# Wyświetlenie najlepszych parametrów
best_rf_params <- results_rf[which.max(results_rf$Accuracy), ]
print("Najlepsze parametry:")
print(best_rf_params)

# Trenowanie modelu z najlepszymi parametrami na całym zbiorze treningowym
final_rf_model <- randomForest(
  HeartDisease ~ ., 
  data = trainData, 
  mtry = best_rf_params$mtry,
  ntree = best_rf_params$ntree,
  nodesize = best_rf_params$nodesize
)


# Predykcja na zbiorze testowym
final_test_predictions <- predict(final_rf_model, newdata = testData, type = "response")

# Obliczenie metryk na zbiorze testowym
final_roc_curve <- roc(testData$HeartDisease, as.numeric(final_test_predictions))
final_auc <- auc(final_roc_curve)
final_cm <- confusionMatrix(final_test_predictions, testData$HeartDisease)
final_precision <- final_cm$byClass["Precision"]
final_recall <- final_cm$byClass["Recall"]
final_accuracy <- mean(final_test_predictions == testData$HeartDisease)

# Wyświetlenie wyników na zbiorze testowym
cat("Metryki na zbiorze testowym:\n")
cat("Accuracy:", final_accuracy, "\n")
cat("AUC:", final_auc, "\n")
cat("Precision:", final_precision, "\n")
cat("Recall:", final_recall, "\n")

# Wyświetlenie confusion matrix jako heatmapy
test_cm_matrix <- as.table(final_cm$table)
cm_data <- as.data.frame(test_cm_matrix)
colnames(cm_data) <- c("Predicted", "Actual", "Count")

# Tworzenie heatmapy z liczbami
ggplot(data = cm_data, aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile() +
  geom_text(aes(label = Count), color = "white", fontface = "bold", size = 6) +
  scale_fill_gradient(low = "black", high = "orange") +
  theme_minimal() +
  labs(title = "Confusion Matrix RandomForest", x = "Predicted", y = "Actual") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


###### Regresja

data <- read.csv("financial_regression.csv")
# Usuwamy zmienne: date, gold.open, gold.high, gold.low, gold.volume
data <- data[, !(names(data) %in% c("date", "gold.open", "gold.high", "gold.low", "gold.volume"))]
data <- data[!is.na(data$gold.close), ]

sum(is.na(data["gold.close"]))
set.seed(123)  # Ustawiamy ziarno losowania dla replikowalności
train_index <- sample(1:nrow(data), size = 0.7 * nrow(data))  # 70% danych jako zbiór uczący
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

library(dplyr)

# Funkcja imputująca brakujące wartości
train_data_imputed <- train_data %>%
  mutate(across(everything(), ~replace(., is.na(.), mean(., na.rm = TRUE))))

test_data_imputed <- test_data %>%
  mutate(across(everything(), ~replace(., is.na(.), mean(., na.rm = TRUE))))

evaluate_model_cv_rpart <- function(params, trainData, folds = 10) {
  # Przygotowanie zbiorów do cross-walidacji
  cv_folds <- createFolds(trainData$gold.close, k = folds, list = TRUE)
  
  # Miejsce na metryki
  metrics <- data.frame(RMSE = numeric(), MAE = numeric(), R2 = numeric(), MAPE = numeric())
  
  # Cross-walidacja
  for (fold in cv_folds) {
    train_fold <- trainData[-fold, ]
    test_fold <- trainData[fold, ]
    
    # Trenowanie modelu (drzewo decyzyjne)
    model <- rpart(
      gold.close ~ ., 
      data = train_fold, 
      method = "anova",  # Regresja
      control = rpart.control(
        cp = as.numeric(params["cp"]),
        maxdepth = as.numeric(params["maxdepth"]),
        minsplit = as.numeric(params["minsplit"])
      )
    )
    
    # Predykcja na zbiorze testowym
    predictions <- predict(model, newdata = test_fold)
    
    # Obliczanie metryk
    rmse <- sqrt(mean((predictions - test_fold$gold.close)^2))
    mae <- mean(abs(predictions - test_fold$gold.close))
    r2 <- cor(predictions, test_fold$gold.close)^2
    mape <- mean(abs((predictions - test_fold$gold.close) / test_fold$gold.close)) * 100
    
    # Zapis metryk
    metrics <- rbind(metrics, data.frame(RMSE = rmse, MAE = mae, R2 = r2, MAPE = mape))
  }
  
  # Średnie metryki z CV
  return(colMeans(metrics, na.rm = TRUE))
}

# Parametry do grid searcha dla drzewa decyzyjnego
grid_search_rpart <- expand.grid(
  cp = seq(0.01, 0.1, by = 0.01),
  maxdepth = c(5, 10, 15),
  minsplit = c(10, 20, 30)
)

# Zmienna, która będzie przechowywać wyniki
best_rpart_metrics <- data.frame()

# Grid search, iteracja po parametrach
for (i in 1:nrow(grid_search_rpart)) {
  params <- grid_search_rpart[i, ]
  metrics <- evaluate_model_cv_rpart(params, train_data_imputed, folds = 10)
  
  # Połączenie parametrów z metrykami bez użycia cbind
  result <- data.frame(
    cp = params$cp, 
    maxdepth = params$maxdepth, 
    minsplit = params$minsplit,
    RMSE = metrics["RMSE"], 
    MAE = metrics["MAE"], 
    R2 = metrics["R2"], 
    MAPE = metrics["MAPE"]
  )
  
  # Dodawanie wyniku do ramki danych (usuńmy nazwy wierszy)
  best_rpart_metrics <- bind_rows(best_rpart_metrics, result)
}

# Wyświetlenie najlepszych parametrów dla drzewa decyzyjnego (minimalny RMSE)
best_rpart_metrics_clean <- best_rpart_metrics
rownames(best_rpart_metrics_clean) <- NULL  # Usuwamy nazwy wierszy
best_params <- best_rpart_metrics_clean[which.min(best_rpart_metrics_clean$RMSE), ]


# Trenowanie modelu z najlepszymi parametrami
best_model <- rpart(
  gold.close ~ ., 
  data = train_data_imputed, 
  method = "anova",
  control = rpart.control(
    cp = best_params$cp,
    maxdepth = best_params$maxdepth,
    minsplit = best_params$minsplit
  )
)

final_predictions <- predict(best_model, newdata = test_data_imputed)

# Obliczanie metryk na zbiorze testowym
final_rmse <- sqrt(mean((final_predictions - test_data_imputed$gold.close)^2))
final_mae <- mean(abs(final_predictions - test_data_imputed$gold.close))
final_r2 <- cor(final_predictions, test_data_imputed$gold.close)^2
final_mape <- mean(abs((final_predictions - test_data_imputed$gold.close) / test_data_imputed$gold.close)) * 100

cat("Test Set Metrics:\n")
cat("RMSE:", final_rmse, "\n")
cat("MAE:", final_mae, "\n")
cat("R2:", final_r2, "\n")
cat("MAPE:", final_mape, "%\n")

# Wizualizacja wyników
results_df <- data.frame(
  Actual = test_data_imputed$gold.close,
  Predicted = final_predictions
)

rpart.plot(
  best_model, 
  type = 3,
  digits = 3,
  fallen.leaves = TRUE,
  main = "Regression Tree"
)

#Random Forest

evaluate_model_cv_rf <- function(params, trainData, folds = 10) {
  # Przygotowanie zbiorów do cross-walidacji
  cv_folds <- createFolds(trainData$gold.close, k = folds, list = TRUE)
  
  # Miejsce na metryki
  metrics <- data.frame(RMSE = numeric(), MAE = numeric(), R2 = numeric(), MAPE = numeric())
  
  # Cross-walidacja
  for (fold in cv_folds) {
    train_fold <- trainData[-fold, ]
    test_fold <- trainData[fold, ]
    
    # Trenowanie modelu (las losowy)
    model <- randomForest(
      gold.close ~ ., 
      data = train_fold, 
      mtry = as.numeric(params["mtry"]),
      ntree = as.numeric(params["ntree"]),
      nodesize = as.numeric(params["nodesize"])
    )
    
    # Predykcja na zbiorze testowym
    predictions <- predict(model, newdata = test_fold)
    
    # Obliczanie metryk
    rmse <- sqrt(mean((predictions - test_fold$gold.close)^2))
    mae <- mean(abs(predictions - test_fold$gold.close))
    r2 <- cor(predictions, test_fold$gold.close)^2
    mape <- mean(abs((predictions - test_fold$gold.close) / test_fold$gold.close)) * 100
    
    # Zapis metryk
    metrics <- rbind(metrics, data.frame(RMSE = rmse, MAE = mae, R2 = r2, MAPE = mape))
  }
  
  # Średnie metryki z CV
  return(colMeans(metrics, na.rm = TRUE))
}

# Parametry do grid searcha dla Random Forest
grid_search_rf <- expand.grid(
  mtry = c(2, 3, 4),       # Liczba zmiennych w rozgałęzieniu
  ntree = c(50, 100, 150), # Liczba drzew w lesie
  nodesize = c(5, 10, 20)  # Minimalna liczba próbek w liściu
)

# Zmienna, która będzie przechowywać wyniki
best_rf_metrics <- data.frame()

# Grid search, iteracja po parametrach
for (i in 1:nrow(grid_search_rf)) {
  params <- grid_search_rf[i, ]
  metrics <- evaluate_model_cv_rf(params, train_data_imputed, folds = 10)
  
  # Połączenie parametrów z metrykami bez użycia cbind
  result <- data.frame(
    mtry = params$mtry, 
    ntree = params$ntree, 
    nodesize = params$nodesize,
    RMSE = metrics["RMSE"], 
    MAE = metrics["MAE"], 
    R2 = metrics["R2"], 
    MAPE = metrics["MAPE"]
  )
  
  # Dodawanie wyniku do ramki danych (usuńmy nazwy wierszy)
  best_rf_metrics <- bind_rows(best_rf_metrics, result)
}

# Wyświetlenie najlepszych parametrów dla Random Forest (minimalny RMSE)
best_rf_metrics_clean <- best_rf_metrics
rownames(best_rf_metrics_clean) <- NULL  # Usuwamy nazwy wierszy
best_rf_metrics_clean[which.min(best_rf_metrics_clean$RMSE), ]

best_rf_params <- best_rf_metrics_clean[which.min(best_rf_metrics_clean$RMSE), ]

# Trenowanie modelu z najlepszymi parametrami
best_rf_model <- randomForest(
  gold.close ~ ., 
  data = train_data_imputed, 
  mtry = best_rf_params$mtry, 
  ntree = best_rf_params$ntree, 
  nodesize = best_rf_params$nodesize
)

# Predykcja na zbiorze testowym
rf_predictions <- predict(best_rf_model, newdata = test_data_imputed)

# Ewaluacja modelu na zbiorze testowym
rf_rmse <- sqrt(mean((rf_predictions - test_data_imputed$gold.close)^2))
rf_mae <- mean(abs(rf_predictions - test_data_imputed$gold.close))
rf_r2 <- cor(rf_predictions, test_data_imputed$gold.close)^2
rf_mape <- mean(abs((rf_predictions - test_data_imputed$gold.close) / test_data_imputed$gold.close)) * 100

# Wyświetlenie metryk
cat("Random Forest Test Set Evaluation:\n")
cat(sprintf("RMSE: %.3f\n", rf_rmse))
cat(sprintf("MAE: %.3f\n", rf_mae))
cat(sprintf("R^2: %.3f\n", rf_r2))
cat(sprintf("MAPE: %.3f%%\n", rf_mape))

# Wykres rozrzutu: wartości rzeczywiste vs przewidywane
scatter_plot <- ggplot(data = test_data_imputed, aes(x = gold.close, y = rf_predictions)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(
    title = "Random Forest: Actual vs Predicted",
    x = "Actual Values",
    y = "Predicted Values"
  ) +
  theme_minimal()

print(scatter_plot)



