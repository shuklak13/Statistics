library(AuxiliaryFunctions)

# linearDiscriminantAnalysis(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume, Smarket, Smarket, Smarket$Direction)
linearDiscriminantAnalysis <- function(f, training, testing, y){
  model <- lda(formula=f, data=training)
  print(model)
  predictions <- predict(model, testing)$class
  computeAccuracy(predictions)
}