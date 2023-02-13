rm(list = ls())

library(class)
library(caret)
library(readr)
library(pROC)

load('knn_data.rda')
load("C:/Users/clemc/OneDrive/Bureau/ESIEE/E4/P3/Machine learning avancé/TD/knn_data.rda")

dfw <- data
#///////////////////////////////////////////////////////////////////////////////
n_data           <- nrow(dfw)
frac_validation  <- 0.10
Y_name <- 'DIFF'

id_Y <- which( names(dfw)  %in% Y_name)
id_X <- which(!names(dfw)  %in% Y_name)


id_validation    <- createDataPartition(1:n_data, p = frac_validation, list = F) 
n_validation     <-length(id_validation)

XY_validation    <- dfw[id_validation,]
X_validation     <- dfw[id_validation, id_X] 
Y_validation     <- dfw[id_validation, id_Y]      # variable à prédire

XY_trainingtest   <- dfw[ - id_validation, ]
X_trainingtest    <- dfw[ - id_validation, id_X]
Y_trainingtest    <- dfw[ - id_validation, id_Y]

n_trainingtest    <- nrow(XY_trainingtest)

knn_val       <-  seq(1,90,1)
frac_training <-  0.80
folds         <-  40

MSE <- NULL
#chaque test utilise une sous-partie des données d'entraînement pour entraîner le modèle KNN et évalue les erreurs de prédiction 
for(i in 1: folds ) #nombre de tests fait 
{
  
  id_train <-  createDataPartition(1:n_trainingtest, p = frac_training, list = F) #séparation des données pour un ensemble d'entrainement
  
  X_train <- XY_trainingtest[id_train,id_X] #explicatives
  Y_train <- XY_trainingtest[id_train,id_Y] #que l'on veut prédire
  
  X_test <- XY_trainingtest[- id_train,id_X]
  Y_test <- XY_trainingtest[- id_train,id_Y]
  Err <- NULL
  for (ik in knn_val) #prédiction avec l'erreur associée
  {

    Y_pred <- knn(X_train,X_test,Y_train, ik) #prédiction des plus proches voisins en fonction des données connues X_train, X_test, Y_train
    
    Err <- c(Err, sum(Y_pred != Y_test) / length(Y_pred) ) #erreur = conparaison entre Y_pred et Y_test
  }
  
  MSE <- cbind(MSE, Err) 
}

#moyenne et écart-type calculées pour chaque k de knn_val
MSE      <- data.frame(MSE); row.names(MSE) <- paste0('k_',knn_val) ; names(MSE) <- paste0('fold_',1:folds) #mean square error
mean_mse <- apply(MSE,1, mean)
sd_mse   <- apply(MSE,1, sd)
knn_stat <- data.frame(mean =  mean_mse,sd = sd_mse)

#création du graphique 
stat <- data.frame (x = knn_val, mean = mean_mse, std = sd_mse)
gr_1 <- ggplot() + geom_line(data = stat, aes(x = x, y = mean))
gr_1 <- gr_1 + xlab('KPPV') + ylab("% Error") 
gr_1

opt_kppv <- which.min(mean_mse) ; cat('best k = \n', opt_kppv) # 15 observations (plus petite erreur de moyenne)

Ypred_valid   <- knn(X_trainingtest,X_validation,Y_trainingtest, opt_kppv)
confusionMatrix(factor(Ypred_valid), factor(Y_validation) ) #performances de classifications

