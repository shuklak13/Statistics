library(AuxiliaryFunctions)

# logisticRegression(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume, Smarket, Smarket, Smarket$Direction)
logisticRegression <- function(f, training, testing, y){
  model <- glm(formula=f, data=training, family=binomial)
  print(summary(model))
  predictedProbs <- predict(model, testing, type="response")
  predictions <- predictedProbs>.5
  computeAccuracy(predictions)
}