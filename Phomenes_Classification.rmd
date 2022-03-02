---
title: <div style="text-align:center;"> <strong> Rapport SY19 TP10 A21 </strong></div>
author:   "Lelong Philomène, Sarbout Ilias, Tran Quoc Hung"
dat: "r format(Sys.time(),'%d %B, %Y')"
output:
  pdf_document: default
  html_document:
    df_print: paged
editor_options: 
  chunk_output_type: inline
---
```{r setup, include=FALSE} 
knitr::opts_chunk$set(warning = FALSE, message = FALSE) 
```
\tableofcontents

# Introduction

Ce rapport présente les différentes méthodes appliquées pour la résolution de deux problèmes de classification dans une première partie puis pour un problème de régression dans une seconde. La méthode donnant le meilleur modèle pour chacun des problèmes a été retenue et déposer sur http://maggle.gi.utc/. Pour nos deux trois problèmes, un partitionnement des données en un ensemble de données d’apprentissage et un ensemble de données de test est effectué et une création de fonctions de validation croisée car cela permet la comparaison  objective des modèles construits. Nous avons fait le choix de créer nos modèles de classification et de régression avec des données non-standardisées. Le format du fichier .Rdata attendu rendait la tâche de standardisation de l'ensemble des données (données du fichier et test_set) compliquée. De plus, standardiser les données du fichier d'apprentissage et du fichier de test indépendamment montrait à l'inverse une tendance à accentuer le taux de mauvaise classification ou la MSE.

# Jeu de données 1 : Problème de classification

  Le problème traité dans cette première partie est un problème de classification. La variable à expliquer, y, est qualitative nominale à 5 modalités  {aa,ao,dcl,iy,sh}.  Pour prédire cette variable, on dispose de 256 variables explicatives quantitatives dont les valeurs semblent être centrées et réduites. Nous disposons de 2250 relevés complets. 
  Nous noterons n, le nombre de relevés et p, le nombre de variables explicatives. Ici nous avons donc n grand (n=2250) et p grand (p=256).

## Analyse Préliminaire et Pré-traitement des données

En nous intéressant au jeu de données, nous remarquons que les valeurs des prédicteurs sont en générales comprises entre -3 et 3, que leur moyenne est proche de 0 et que leur variance est proche de 1. Nous avons donc décidé de les conservés tel quels.  
  
```{r import-data,include = FALSE , cache = TRUE} 
library(methods)
library(MASS)
library(caret)
library(nnet)
library(mclust)
#traitement des données
setwd("/home/tranquochung/Desktop/SY19/TD10")  # path de TRAN
#setwd("Z:/Documents/SY19/sy19/DonnéesTP10")# path de philo
data = read.table("parole_train.txt")
data = data[complete.cases(data),]
n<- nrow(data)
p<- ncol(data)
row.names(data) <- 1:nrow(data)
lev = levels(as.factor(data$y))
```

```{r, cache = TRUE, include = FALSE}
summary(data[,1:3])
```

```{r, include = FALSE, cache = TRUE}
set.seed(42)#graine pour avoir les mêmes résultats sur différentes machines
rows <- sample(nrow(data)) 
data <- data[rows, ]
n    <- nrow(data)
nfolds = 10
#Création des ensemble test et apprentissage pour la validation croisée
folds = cut(seq(1,nrow(data)),breaks=nfolds,labels=FALSE) 
```

Dans un premier temps, nous avons réalisé une analyse en composante principales pour réduire le nombre de variables explicatives  afin de permettre de visualiser les données vu leur haute dimensionalité. Ensuite, nous avons créer les ensembles qui nous servirons par la suite pour la validation croisée.

![Texte alternatif](Z:/Documents/SY19/sy19/images/Screeplot_dataset1.png "Screeplot Dataset1")





Ce graphique réprésente le pourcentage de la variance expliquée par rapport au nombre de composantes principales. Nous remarquons que PC1 explique 55% de la variance totale et PC2 15 %. De ce fait, avec ces deux composantes nous pouvons représenter 70% des informations contenues par les 256 variables. Et de ce fait, obtenir une représentation plutôt fidèles des individus en 2 dimensions. La projection sur les axes factoriels révèle trois clusters dont deux sont facilement assimilables à des classes de phonemes distinctes.

![Texte alternatif](Z:/Documents/SY19/sy19/images/Rplot_dataset1.png "Screeplot Dataset1")

## Modèles testés :

### Méthodologie de comparaison de modèles:

Pour commencer, nous avons créer une fonction de validation croisée a 10-CV afin de comparer les résultats des différents modèles de classification. Cela à pour but d'évaluer la précision de notre modèle sur plusieurs sous ensembles de test puis d'en faire la moyenne et de ce fait d'obtenir des résultats plus fiables. Nous l'avons adaptée aux différents modèles testés.

```{r cross-validation, include = FALSE, cache=TRUE} 
library(nnet)
library(MASS)
library(mlbench)
library(glmnet)

cross_validation <- function(data, model, folds) {
  n <- nrow(data)
  p <- ncol(data)
  ntst <- n/folds
  
  set.seed(17)
  fold_ids <- rep(seq(folds), ceiling(n / folds))
  fold_ids <- fold_ids[1:n]
  fold_ids <- sample(fold_ids, length(fold_ids))
  
  CV_model_accuracy  <- vector(length = folds, mode = "numeric")
  
  ## Loop ##
  for (k in (1:folds)) {
    if(model == "LDA") {
      CV_model_accuracy[k] <- LDA_accuracy(y~., 
                                           traindata=data[which(fold_ids != k),], 
                                           testdata=data[which(fold_ids == k),]
                                           )
    }
    else if(model == "QDA") {
      CV_model_accuracy[k] <- QDA_accuracy(y~., 
                                           traindata=data[which(fold_ids != k),], 
                                           testdata=data[which(fold_ids == k),]
                                           )
    }
    else if(model == "NAIVE_BAYES") {
      CV_model_accuracy[k] <- NBAYES_accuracy(y~., 
                                           traindata=data[which(fold_ids != k),], 
                                           testdata=data[which(fold_ids == k),]
                                           )
    }
    else if(model == "RDA") {
      CV_model_accuracy[k] <- RDA_accuracy(y~., 
                                           traindata=data[which(fold_ids != k),], 
                                           testdata=data[which(fold_ids == k),]
                                           )
    }
    else if(model == "SVM_RADIAL"){
      CV_model_accuracy[k] <- SVM_RADIAL_accuracy(
                    traindata=data[which(fold_ids != k),], 
                    testdata=data[which(fold_ids == k),]
                    )
   }
   else if(model == "SVM_LINEAR"){
      CV_model_accuracy[k] <- SVM_LINEAR_accuracy(
                    traindata=data[which(fold_ids != k),], 
                    testdata=data[which(fold_ids == k),]
                    )
  }
  else if(model == "RF") {
      CV_model_accuracy[k] <- RF_accuracy(y~., 
                                           traindata=data[which(fold_ids != k),], 
                                           testdata=data[which(fold_ids == k),]
                                           )
  }
  else if(model == "KNN") {
    CV_model_accuracy[k] <- KNN_accuracy(y~., 
                                   traindata=data[which(fold_ids != k),], 
                                   testdata=data[which(fold_ids == k),])
  }
 }
   noquote(sprintf("Mean accuracy %s : %.3f",model, mean(CV_model_accuracy)))
}
```

### Les méthodes d'analyse discriminante

Nous nous nous sommes d’abord intéressées aux modèles discrimants. Ici, nous nous trouvons avec un p et n élevé.

L'analyse discriminante englobe des méthodes qui peuvent être utilisées à la fois pour la classification et la réduction de la dimensionnalité. L'analyse discriminante linéaire (LDA) est particulièrement populaire car elle est à la fois un classificateur et une technique de réduction de dimensionnalité. L'analyse discriminante quadratique (QDA) est une variante de LDA qui permet une séparation non linéaire des données. La QDA risque de ne pas être performante dans notre cas à cause de son grand nombre de paramètres. Les résultats présentés sont les MSE obtnues pour les différents modèles.

```{r, include=FALSE, cache = TRUE}
library(MASS)
library(naivebayes)
###### LDA ######
LDA_accuracy <- function(formula, traindata, testdata) {
  p <- ncol(traindata)
  model.lda <- lda(formula, data=traindata)
  pred.lda <- predict(model.lda, newdata=testdata[,-p])
  accuracy.lda <- mean(pred.lda$class == testdata[,p])
  return(accuracy.lda)
}

###### QDA ######
QDA_accuracy <- function(formula, traindata, testdata) {
  p <- ncol(traindata)
  model.qda <- qda(formula, data=traindata)
  pred.qda <- predict(model.qda, newdata=testdata[,-p])
  accuracy.qda <- mean(pred.qda$class == testdata[,p])
  return(accuracy.qda)
}


###### NAIVE BAYES ######

NBAYES_accuracy <- function(formula, traindata, testdata) {
  p <- ncol(traindata)
  traindata$y <- as.factor(traindata$y)
  testdata$y <- as.factor(testdata$y)
  model.nbayes <- naive_bayes(formula, data=traindata)
  pred.nbayes <- predict(model.nbayes, newdata=testdata[,-p])
  accuracy.nbayes <- mean(pred.nbayes == testdata[,p])
  return(accuracy.nbayes)
}
```


```{r,cache = TRUE}
cross_validation(data, "LDA"              , 10)
cross_validation(data, "QDA"              , 10)
cross_validation(data, "NAIVE_BAYES"      , 10)

```

Nous remarquons que la LDA permet d'obetnir une bonne performance. Il nous a semblé pertinent d’appliquer la methode RDA (regularized discriminant analysis), mentionnée en cours, qui est particulièrement utile dans le cas d’un grand nombre de variables explicatives et qui peut permettre d’obtenir de meilleurs résultats que les modèles vu précédemment. Elle combine la QDA et la LDA. Si gamma=0 et lambda=0, on retouve le modèle QDA et si gamma=0 et lambda=1, on retrouve le modèle LDA. Nous utilisons un gridsearch afin de trouver la meilleure valeur des paramètres lambda et gamma pour déterminer si la RDA est intéressante dans notre cas. Nous obtenons le meilleur résultat pour gamma=0,53 et lambda=0,92 sur l'ensemble de test. On applique donc la RDA avec ces paramètres.

![Texte alternatif](Z:/Documents/SY19/sy19/images/rda_dataset1.png "Screeplot Dataset1")


```{r, include=FALSE,cache = TRUE}
library(httpuv)
library(klaR)
###### RDA with gamma = 0.53, lampda = 0.92 ######
RDA_accuracy <- function(formula, traindata, testdata) {
  p <- ncol(traindata)
  model.rda <- rda(formula, data=traindata, gamma = 0.53, lambda = 0.92)
  pred.rda  <- predict(model.rda, newdata=testdata)
  accuracy.rda <- mean(pred.rda$class == testdata[,p])
  return(accuracy.rda)
}
```

```{r,cache = TRUE}
cross_validation(data, "RDA"               , 10)
```

La RDA nous permet d'améliorer légèrement le résultat obtenu précédemment avec la LDA en obtenant une précision proche de 0,93. Nous allons maintenant testé d'autre méthodes afin de voir s'il est possible d'améliorer encore ce résultat.

### KNN 

Nous utiliserons le package caret, qui teste automatiquement différentes valeurs possibles de k, puis choisit le k optimal qui minimise l'erreur de validation croisée ("cv") et correspond au meilleur modèle KNN final qui explique le mieux nos données

```{r}
KNN_accuracy <- function(formula, traindata, testdata) {
  traindata[["y"]] = factor(traindata[["y"]])
  set.seed(123)
  model.knn <- train(y ~., data = traindata, 
                     method = "knn",
                     preProcess = c("center","scale"),
                     tuneGrid = expand.grid(k = 1:20), 
                     trControl = trainControl("cv", number = 10) )
  pred.knn <- predict(model.knn, newdata = testdata)
  return(mean(pred.knn == testdata[,p]))
}

```

```{r}
cross_validation(data, "KNN", 10)
```


### Les SVM
Nous nous intéressons maintenant aux SVMs linéaire et radial. Les méthodes SVMs peuvent être appliqués dans le cas de classification multi-classes, elles decompose alors le problème en plusieurs problèmes de classification binaires. La méthode appliqué ici est la méthode en un contre un. La différence entre les deux méthodes présentés ci-dessous est que l'une trouve ses frontières linéaires dans l'espace de prédiction tandis que l'autre, plus flexible, permet d'élargir l'espace de prédiction afin de trouver une meilleur frontière linéaire dans un nouvel espace.


```{r, include=FALSE,cache = TRUE}
library(e1071)
######  SVM_RADIAL  ######

SVM_RADIAL_accuracy <- function(traindata, testdata) 
{
  x = subset(traindata, select = -y)
  p <- ncol(traindata)
  model.svm.radial     <- svm(x, data.frame(factor(traindata$y)), kernel="radial" , scale=F, type = "C-classification")
  pred.svm.radial      <- predict(model.svm.radial , newdata=testdata[,-p])
  accuracy.svm.radial  <- mean(pred.svm.radial == testdata[,p])
  return(  accuracy.svm.radial)
}

######  SVM_LINEAR  ######

SVM_LINEAR_accuracy <- function(traindata, testdata) 
{
  x = subset(traindata, select = -y)
  p <- ncol(traindata)
  model.svm.linear     <- svm(x, data.frame(factor(traindata$y)), kernel="linear" , scale=F, type = "C-classification")
  pred.svm.linear      <- predict(model.svm.linear , newdata=testdata[,-p])
  accuracy.svm.linear  <- mean(pred.svm.linear == testdata[,p])
  return(  accuracy.svm.linear)
}
```

```{r, cache = TRUE}
cross_validation(data, "SVM_RADIAL"       , 10)
cross_validation(data, "SVM_LINEAR"       , 10)
```


Nous obtenons un meilleur résultat avec le SVM radial mais qui reste toujours inférieur à celui obtenu avec la RDA. Cependant, ce modèle à permi d'obtenir un meilleur résultat lorsque nous l'avons déposé sur le site. Nous nous sommes ensuite intéressés à d'autres méthodes connues pour leurs performance sur des ensembles de données de grande dimension.

### Forêts aléatoires 
La méthode des forêts aléatoires est basée sur le système du bagging et est composé de plusieurs arbres de décision, travaillant de manière indépendante sur une vision d'un problème. Chacun produit une estimation, et c'est l'assemblage des arbres de décision et de leurs analyses, qui va donner une estimation globale. On choisit la catégorie de réponse la plus fréquente. Plutôt que d'utiliser tous les résultats obtenus, on procède à une sélection en recherchant la prévision qui revient le plus souvent. Cela permet d'obtenir de meilleurs résultats qu'avec un arbre de décision unique.

```{r, include=FALSE, cache = TRUE}
library(randomForest)
######  Random Forest  ######

RF_accuracy <- function(formula, traindata, testdata) 
{
  p <- ncol(traindata)
  model.RF             <- randomForest(as.factor(y)~.,data=traindata, ntree = 500, mtry = 5)
  pred.RF              <- predict( model.RF , newdata=testdata)
  perf                 <-table(testdata$y  , pred.RF)
  accuracy.RF = (sum(diag(perf))/sum(perf))
  return(accuracy.RF)
}
```

```{r, cache = TRUE}
cross_validation(data, "RF"               , 10)
```

Ce modèle semble donner de très bon résultats presque équivalents à ceux de la rda. Cependant, il faut faire attention car le random forest est un modèle d'apprentissage, dont l'efficacité dépend fortement de la qualité de l'échantillon de données de départ.


## Meilleurs Résultats :

Nous obtenons la meilleure précision avec le modèle SVM radial ou la RDA. Cela peut s'expliquer par le fait que ces deux méthodes fonctionnent très bien dans le cas d'espaces à grande dimension,ce qui est le cas içi.  

# Jeu de données 2 : Problème de classification

Le problème traité dans cette première partie est un problème de classification. La variable à expliquer, y, est qualitative nominale à 26 modalités. l'objectif est de prédire la lettre de l'alphabet à partir de 16 variables explicatives quantitatives dont les valeurs semblent être centrées et réduites. Nous disposons de 10000 relevés complets. 

## Prétraitement et Analyse Exploratoire :

Dans ce jeu de donnée, nous avons déplacé la colonne représentant la variable à expliquer à la fin du tableau pour s'adapter la fonction cross_validation que nous avons créée pour les problèmes de classification.Ici nous avons on notera n2, le nombre de relevés n2=1000 et le nombre de variables explicative p2=16.

```{r, include=FALSE, cache = TRUE}
#pré-traitement
#setwd("Z:/Documents/SY19/sy19/DonnéesTP10") #path philo
data2 = read.table("letters_train.txt")
data2 = data2[complete.cases(data2),]
n2<- nrow(data2)
p2<- ncol(data2)
row.names(data2) <- 1:nrow(data2)
lev = levels(as.factor(data2$y))

set.seed(42)#graine pour avoir les mêmes résultats sur différentes machines

#on range les données
rows <- sample(nrow(data2)) 
data2 <- data2[rows, ]
rownames(data2) <- 1:nrow(data2)


d = subset(data2, select= -Y)
y = subset(data2, select= Y)
d$y = y$Y

data2 = d

head(data2)
```

### ACP

Afin de rendre les résultats plus interprétables nous avons voulu réduire le nombre de variables du modèle. Nous avons d’abord regardé si celles-ci semblaient corrélées mais rien ne semblait significatif. Comme toutes les variables sont quantitatives, nous avons appliqué l’ACP. Ses résultats nous confirment qu’il est compliqué de réduire le nombre de variables avant de choisir notre modèle.

![Texte alternatif](Z:/Documents/SY19/sy19/images/ACP_dataset2.png "Screeplot Dataset1")

## Choix du modèle :

Pour ce problème, nous avons testé les mêmes modèles que ceux vu dans la première partie en partant de la même méthode car nous étions également dans un problème de classification. Cependant, le problème étant différent : p plus petit et n plus grand, le meilleur résultat est obtenu cependant obtenu avec un modèle semblable.


### KNN 

Nous utiliserons le package caret, qui teste automatiquement différentes valeurs possibles de k, puis choisit le k optimal qui minimise l'erreur de validation croisée ("cv") et correspond au meilleur modèle KNN final qui explique le mieux nos données

```{r}
KNN_accuracy <- function(formula, traindata, testdata) {
  traindata[["y"]] = factor(traindata[["y"]])
  set.seed(123)
  model.knn <- train(y ~., data = traindata, 
                     method = "knn",
                     preProcess = c("center","scale"),
                     tuneGrid = expand.grid(k = 1:20), 
                     trControl = trainControl("cv", number = 10) )
  pred.knn <- predict(model.knn, newdata = testdata)
  
  print(mean(pred.knn == testdata[,p2]))

  return(mean(pred.knn == testdata[,p2]))
}

```

```{r}
cross_validation(data2, "KNN", 10)
```

### Analyse discriminante
Pour ces modèles, nous avons obtenu les résultats suivants :

```{r, cache = TRUE}
cross_validation(data2, "LDA"              , 10)
cross_validation(data2, "QDA"              , 10)
cross_validation(data2, "NAIVE_BAYES"      , 10)

```
Nous remarquons qu'il sont beaucoup moins pertinent que pour le premier dataset, sauf en ce qui concerne la QDA. Et ici, la RDA n'était pas pertinente car nous obtenons le meilleur résultat pour gamma=0 et lambda=0, ce qui correspond à la QDA.

![Texte alternatif](Z:/Documents/SY19/sy19/images/rda.png "Screeplot Dataset1")

Ensuite, nous avons également testé la MDA (mixture discriminant analysis) qui nous a donné un bon résultat : une précision de 0,92. Dans cette méthode, chaque classe est supposée être un mélange gaussien de sous-classes.

### SVM
Comme dans la partie précédente, nous avons testé les SVM linéaire et radial.

```{r, cache = TRUE}
cross_validation(data, "SVM_RADIAL"       , 10)
cross_validation(data, "SVM_LINEAR"       , 10)
```

Le SVM radial nous permet donc d'obtenir le meilleur résultat des méthodes testées jusqu'à présent. Cependant, on peut encore améliorer sa précision. En effet, pour le SVM radial , on utilise la validation croisée à 10-folds pour trouver le meilleur parametre  de cout et ici, on obtient cost = 30

La fonction tune(), pour effectuer une validation croisée. Par défaut, tune() effectue une validation croisée décuplée sur un ensemble de modèles d'intérêt. Afin d'utiliser cette fonction, nous transmettons des informations pertinentes sur l'ensemble de modèles qui sont à l'étude. La commande suivante indique que nous voulons comparer les SVM avec un noyau radial, en utilisant une plage de valeurs du paramètre de coût.

```{r, include=FALSE, cache = TRUE}
library(e1071)

obj <- tune(svm, y~., data = data2, kernel = "radial",
            ranges =list(cost=c(29,30,31,32))
)

summary(obj)

```

```{r, include=FALSE, cache = TRUE}
######  SVM_RADIAL  ######

SVM_RADIAL_accuracy <- function(traindata, testdata) 
{
  x = subset(traindata, select = -y)
  p <- ncol(traindata)
  model.svm.radial     <- svm(x, data.frame(factor(traindata$y)), kernel="radial" , scale=F, type = "C-classification", cost = 32)
  pred.svm.radial      <- predict(model.svm.radial , newdata=testdata[,-p])
  accuracy.svm.radial  <- mean(pred.svm.radial == testdata[,p])
  return(  accuracy.svm.radial)
}

```

```{r, cache = TRUE}
cross_validation(data2, "SVM_RADIAL"       , 10)
```

### Forêt aléatoire
Enfin, nous avons également testé les forêts aléatoires qui nous permettent d'obtenir 
une très bonne précision de prediction de 0,95. C'est le meilleur modèle que nous avons obtenu apres le SVM radial avec cost=40.

```{r, cache = TRUE}
cross_validation(data2, "RF"               , 10)
```

### Réseau de neuronnes ?

### Conclusion 
Nous obtenons donc les meilleurs résultats de prédiction de classification avec les modèles des forêts aléatoires ou le SVM radial.


# Conclusion problèmes de classification :

Nous obtenons deux meilleurs modèles de classifications semblables pour les deux problèmes de classification bien que les caractéristiques des deux jeux de données soient différentes. Cela peut s'expliquer par le fait que les SVM radial peuvent être considérés comme de bons approximeurs universelles et sont efficaces en grandes dimension. Il en va de même pour la RDA et la méthode de forêts aléatoire, ce qui explique que l'on ai obtenu de bons résultats sur nos deux jeux de données avec ces méthodes.

# Jeu de données 3 : Problème de régression

Le dernier problème traité est un problème de régression. L'objectif est de prédire la variable quantitative "cnt", réprésentant le nombre de location de vélo, en fonction des paramètre environnementaux et saisonniers. Pour cela, nous disposons de 13 variables explicatives quantitatives et qualitatives et de 365 relevés complets.
  Pour ce problème, l'ensemble d'apprentissage est relativement petit, et peut être différent de l'ensemble de test disponible, ce qui peut expliquer que nous obtenons une MSE élevée lorsque nous testons nos modèles sur le site.
  Avant, d'essayer de prédire la variable count, nous avons essayé de déterminer les variables qui influent le plus sur le nombre de locations et d'analyser le sens de leur influence.

```{r, include=FALSE, cache = TRUE}
#prétraitement
data  <- read.csv('bike_train.csv',sep = ",", header = TRUE)
data = na.omit(data) #enlever les NA

set.seed(42)
rows <- sample(nrow(data)) 
data <- data[rows, ]
rownames(data) <- 1:nrow(data)
n<- nrow(data)
#on retire yr qui est tout le temps à 0, le jour de l'année et l'index
data = data[c('season','mnth','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed','cnt')] 
nfolds = 10
folds = cut(seq(1,nrow(data)),breaks=nfolds,labels=FALSE)

head(data)
```


## Prétraitement et Analyse Exploratoire :

```{r, include=FALSE, cache = TRUE}
#ensemble d'apprentissage et de test
set.seed(42) #pour avoir tjrs même répartition 
n<-nrow(data)
cnt<-data$cnt
ntrain <- round((2/3)*n)
train <- sample(n, ntrain)
data.train <- data[train,]
data.test <- data[-train,]

summary(data.train)


head(data.train)

```


Dans un premier temps, pour analyser l'influence des différentes variables sur le nombre de locations de vélo, nous avons décidé de nous intéresser aux corrélations et de réaliser une ACP.

### ACP

L'ACP nous a permis de déterminer les variables ayant le plus d'influence sur la varaible count ainsi que leur sens d'influence avec les graphiques.

![Texte alternatif](Z:/Documents/SY19/sy19/images/Var_infl_fleches.png "Screeplot Dataset1")

On s'intéresse aux fleches les plus grandes car plus elles sont grande mieux elles sont représentée par les axes. On voit ici que cnt,atemp,temp, season, month, weather, hum soemble être bien représentées.
comme on remarque que cnt suit la meme direction que temp et atemp, on peut en déduire qu'ils sont positivement corrélé.
le nombre de location se trouve de l'autre coté de l'axe que weather, on peut donc en déduire que cnt est inversement corrélée à weather. Le mauvais temps à donc une mauvaise influence sur le nombre de location.

### Corrélation

Ensuite, nous nous sommes intéressés aux corrélatios de ses varaibles qui nbous ont permis de confirmer les suppositions déduites de l'ACP.

![Texte alternatif](Z:/Documents/SY19/sy19/images/corr_princ.png "Screeplot Dataset1")

Les variables  atemp et temp sont très corrélées, et elles sont toutes les deux très corrélées à cnt. De plus, les variables season et month sont corrélés et légèrements corrélées à la variable cnt.


### Conclusion :

On en conclut que les variables qui influent le plus sur le nombre de locations sont temp et atemp, le sens de cette influence est positive car elles sont positivement corrélées à cnt. Ensuite, la variable saison et mois influent également de manière positive sur le nombre de locations tandis que la variable weather influent dans les sens négatif.

## Methodes utilisées :
Nous avons ensuite essayé de trouver le meilleur modèle pour de prédire la valeur de la variable cnt. Pour se faire nous avons testé un grand nombre de modèles.

On défini ensuite 3 fonctions qui va utiliser pour évaluer le model : 

```{r}
# Model performance metrics
RMSE = function(y_test, y_predict){
  sqrt(mean((y_test - y_predict)^2))
}
R2 = function(y_test,y_predict){
  cor(y_test,y_predict)^2
}
MSE = function(y_test,y_predict){
  mean((y_test-y_predict)^2)
}
```


### Méthode de selection du modèle

L'objectif dans cette section est d'identifier un sous-ensemble parmi les p variables et d'utiliser ce sous-ensemble pour construire notre modèle.
L'algorithme de sélection du meilleur modèle ne peut pas être appliqué car p est trop grand. En revanche, nous appliquons la sélection à pas ascendante et la sélection à pas descendante.

```{r, include=FALSE, cache = TRUE}

#bibliothèques
library('leaps')
library(pls)
library(rpart)

#selection du modèle
# methode ascendante
data.fwd <- regsubsets(cnt~., data=data.train, method='forward',nvmax=10)
data.fwd.sum <-summary(data.fwd)

get_model_formula <- function(k, object, outcome){
  models <- summary(object)$which[k,-1] 
  predictors <- names(which(models == TRUE))
  predictors <- paste(predictors, collapse = "+")
  as.formula(paste0(outcome, "~", predictors))
}


get_model_formula(data.fwd.r2.i, data.fwd, "cnt")
```


#### Critère de Sélection


*R² ajusté :* 

```{r, include=FALSE, cache = TRUE}
# R ajusté
data.fwd.r2.i <-which.max(data.fwd.sum$adjr2)
data.fwd.r2 <-lm(get_model_formula(data.fwd.r2.i, data.fwd, "cnt"), data= data.train)
data.fwd.pred.r2 <- predict(data.fwd.r2, newdata = data.test)
err.adjr<-mean((data.test$cnt - data.fwd.pred.r2)^2)
```

```{r, cache = TRUE}
mean((data.test$cnt - data.fwd.pred.r2)^2)
```


*BIC ;*

```{r, include=FALSE, cache = TRUE}
#BIC
data.fwd.bic.i <- which.min(data.fwd.sum$bic)
data.fwd.bic <- lm(get_model_formula(data.fwd.bic.i, data.fwd, "cnt"), data = data.train)
data.fwd.pred.bic <- predict(data.fwd.bic, newdata = data.test)
err.bic<-mean((data.test$cnt - data.fwd.pred.bic)^2)
```

```{r, cache = TRUE}
mean((data.test$cnt - data.fwd.pred.bic)^2)
```

#### Valisation d'ensemble :

```{r, include=FALSE, cache = TRUE}
#HO
Formulas.fwd <- c()
for (i in (1:10)){
  Formulas.fwd <- c(Formulas.fwd,get_model_formula(i,data.fwd,"cnt"))
}

err.fwd.holdout <- c()
for (i in (1:length(Formulas.fwd))){
  reg <- lm(Formulas.fwd[[i]], data = data.train)
  pred <- predict(reg, newdata = data.test)
  err.fwd.holdout[i] <- mean((data.test$cnt-pred)^2)
}
which.min(err.fwd.holdout)

data.fwd.holdout <- lm(get_model_formula(which.min(err.fwd.holdout), 
                                                data.fwd, "cnt"), data = data.train)
data.fwd.pred.holdout <- predict(data.fwd.holdout, newdata = data.test)
err.hO<-mean((data.test$cnt - data.fwd.pred.holdout)^2)
```

```{r, cache = TRUE}
mean((data.test$cnt - data.fwd.pred.holdout)^2)
```

*Validation Croisée :*

```{r, include=FALSE, cache = TRUE}
#CV
K<-5
err.CV.min<-10000000
CV.idx<-(-2)
CV.idx.result=0
CV.min.result=0
folds=sample(1:K,n,replace=TRUE)
CV<-rep(0,10)

for(i in (1:10)){
  for(k in (1:K)){
    reg<-lm(Formulas.fwd[[i]],data=data[folds!=k,])
    pred<-predict(reg,newdata=data[folds==k,])
    CV[i]<-CV[i]+ sum((data$cnt[folds==k]-pred)^2)
  }
  CV[i]<-CV[i]/n
  if (CV[i]<err.CV.min){
    err.CV.min<-CV[i]
    CV.idx<-i
  }
  CV.idx.result = CV.idx.result+CV.idx
  CV.min.result = CV.min.result+err.CV.min
}

```

```{r, cache = TRUE}
err.CV.min
```


#### méthode de régularisation 

Nous testons également les méthodes de régularisation qui souffrent moins de la variabilité des données. 

```{r, include=FALSE, cache = TRUE}
#Régularisation
install.packages("glmnet")
library(glmnet)

#mise en forme des données
set.seed(42)
x<-model.matrix(cnt~., data)
y<-data$cnt
train<-sample(1:n,ntrain)

xtrain<-x[train,]
ytrain<-y[train]
xtst<-x[-train,]
ytst<-y[-train]

```


*Méthode des crêtes :*

```{r, include=FALSE, cache = TRUE}
#Ridge Regression
cv.out<-cv.glmnet(xtrain,ytrain,alpha=0)
plot(cv.out)

fit<-glmnet(xtrain,ytrain,lambda=cv.out$lambda.min,alpha=0)
ridge.pred<-predict(fit,s=cv.out$lambda.min,newx=xtst)
err.RR<-mean((ytst-ridge.pred)^2)
```

```{r, cache = TRUE}

data.frame(
  MSE     = MSE(ridge.pred, data.test$cnt),
  RMSE    = RMSE(ridge.pred, data.test$cnt),
  Rsquare = R2(ridge.pred, data.test$cnt)
)

MSE(ridge.pred, data.test$cnt)
```

*Lasso :*

```{r, include=FALSE, cache = TRUE}
cv.out<-cv.glmnet(xtrain,ytrain,alpha=1)
plot(cv.out)

fit.lasso<-glmnet(xtrain,ytrain,lambda=cv.out$lambda.min,alpha=1)
lasso.pred<-predict(fit.lasso,s=cv.out$lambda.min,newx=xtst)
err.lasso<-mean((ytst-lasso.pred)^2)
```

```{r, cache = TRUE}

data.frame(
  MSE     = MSE(lasso.pred, data.test$cnt),
  RMSE    = RMSE(lasso.pred, data.test$cnt),
  Rsquare = R2(lasso.pred, data.test$cnt)
)

MSE(lasso.pred, data.test$cnt)
```


L'elastique net a aussi été testé (comme pour la rda mais avec methode="glmnet") mais nous obtenions alpha = 1, ce qui correspond à la méthode du lasso.

### Forêt aléatoire

Ensuite, nous avons testé les forêts aléatoire, plus précises que les arbres de décisions et nous avons utilisé ungridsearch avec double k-fold validation sur les paramètres : nombre d'arbres et nombre de paramètres à considérer sur chaque split pour essayer de déterminer leur meilleure valeur. On obtient ntree=340 et mtry=6. Ce qui nous donne pour résultat : 271 182. Ce qui est notre meilleur résultat pour le moment.


```{r}

library(caret)
library(e1071)
rf.tune <- train(
        cnt ~., data = data.train, method = "randomForest",
        trControl = trainControl("cv", number = 10),
        tuneGrid = expand.grid(alpha = c(100,200,300,400), lambda = c(5,6,7,8))
)

```


```{r}
library(randomForest)
set.seed(42)
rf <-randomForest(cnt~.,data=data.train, mtre = 340, mtry = 6) 
rf.pred = predict(rf,data.test)

data.frame(
  MSE     = MSE(rf.pred, data.test$cnt),
  RMSE    = RMSE(rf.pred, data.test$cnt),
  Rsquare = R2(rf.pred, data.test$cnt)
)

MSE(rf.pred, data.test$cnt)

```

### GAM

GAM paramétrique (recherche aléatoire sur les degrés des différentes splines) : 390507
Modèle linéaire : 505833

### Réseau de neuronnes



### SVR

Pour finir, nous avons testé les Support Vector Regression avec noyeau linéaire et radial. Nous avons utilisé, la validation croisée pour déterminé le paramètre C de la fonction. Au final, on obtient le meilleur résultat en général avec pour un SVM à noyeau radial prenant pour paramètres C=1000 et eps=0.001. Cependant, le résultat dépend énormément de l'ensemble d'apprentissage choisi.

```{r, include=FALSE, cache = TRUE}
library(kernlab)
library('MASS')
set.seed(42)
n<-nrow(data)
cnt<-data$cnt
ntrain <- round((2/3)*n)
train <- sample(n, ntrain)
data.train <- data[train,]
data.test <- data[-train,]
svmfit<-ksvm(cnt~.,data=data.train,scaled=FALSE,type="eps-svr",
             kernel="rbfdot",C=10e4, epsilon=0.001)
yhat<-predict(svmfit,newdata=data.test)
err.svm<-mean((data.test$cnt-yhat)^2)
```

```{r, cache = TRUE}
# TEST 1
mean((data.test$cnt-yhat)^2)
```

```{r,include=FALSE, cache = TRUE}
svmfit<-ksvm(cnt~.,data=data.train,scaled=FALSE,type="eps-svr",
             kernel="rbfdot",C=1000, epsilon=0.001)
yhat<-predict(svmfit,newdata=data.test)
err.svm<-mean((data.test$cnt-yhat)^2)
```

```{r, cache = TRUE}
# TEST 2
mean((data.test$cnt-yhat)^2)
```


## Conclusion

*A adapter avec forêts aléatoires*

Pour résoudre le problème de régression, nous avons choisi d'utiliser dans un premier temps une methode de sélection d'un sous ensemble de modèles, ici la méthode ascendante. Puis une méthode de sélection du meilleur modèle parmi le sous-ensemble obtenus. Pour la méthode de sélection du modèle, notre choix s'est porté sur le hold-out car on obtient la plus petite MSE parmi les modèles testés. Pourtant il était difficile d'estimer s'il allait aussi bien performer sur le test-set, car l'estimation de l'erreur est dépendante de la séparation des données,
