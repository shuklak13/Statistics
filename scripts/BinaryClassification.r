source("AuxiliaryFunctions.r")

# logisticRegression(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume, Smarket, Smarket, Smarket$Direction)
logisticRegression <- function(f, training, testing, y){
  library(mlogit)
  model <- glm(formula=f, data=training, family=binomial)
  print(summary(model))
  predictedProbs <- predict(model, testing, type="response")
  predictions <- predictedProbs>.5
  computeAccuracy(predictions)
}

# linearDiscriminantAnalysis(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume, Smarket, Smarket, Smarket$Direction)
linearDiscriminantAnalysis <- function(f, training, testing, y){
  library(MASS)
  model <- lda(formula=f, data=training)
  print(model)
  predictions <- predict(model, testing)$class
  computeAccuracy(predictions)
}

# quadraticDiscriminantAnalysis(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume, Smarket, Smarket, Smarket$Direction)
quadraticDiscriminantAnalysis <- function(f, training, testing, y){
  library(MASS)
  model <- qda(formula=f, data=training)
  print(model)
  predictions <- predict(model, testing)$class
  computeAccuracy(predictions)
}

# knearestNeighbors(cbind(Smarket$Lag1, Smarket$Lag2), cbind(Smarket$Lag1, Smarket$Lag2), Smarket$Direction, 3)
knearestNeighbors <- function(training, testing, y, k){
  library(class)
  predictions <- knn(training, testing, y, k=k)
  print(mean(predictions==y))
}
