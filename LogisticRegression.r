checkAssumptions <- function(d){
  
}

# data$y <- formatYtoNum(data$y, "Yes", "No")
formatYtoNum <- function(y, one, zero){
  levels(y)[levels(y)==one] <- TRUE
  levels(y)[levels(y)==zero] <- FALSE
  y
}

# logisticRegression(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume, Smarket2, Smarket2, Smarket2$Direction)
logisticRegression <- function(f, training, testing, y){
  model <- glm(formula=f, data=training, family=binomial)
  print(summary(model))
  
  predictedProbs <- predict(model, testing, type="response")
  predictions <- predictedProbs>.5
  accuracy <- mean(predictions == y)
  
  print(paste("Accuracy: ", accuracy))
  print(paste("Flipping a coin: ", 0.5))
  naiveAccuracy <- ifelse(mean(y=="TRUE")>.5, 
                          mean(y=="TRUE"), 
                          1-mean(y=="TRUE"))
  print(paste("Naively guessing most common category: ", naiveAccuracy))
}