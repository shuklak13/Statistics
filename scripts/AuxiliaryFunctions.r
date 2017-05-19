# data$y <- formatYtoBool(data$y, "Yes", "No")
formatYtoBool <- function(y, one, zero){
  levels(y)[levels(y)==one] <- TRUE
  levels(y)[levels(y)==zero] <- FALSE
  y
}

computeAccuracy <- function(predictions, y){
  accuracy <- mean(predictions == y)
  print(paste("Accuracy: ", accuracy))
  
  print(paste("Flipping a coin: ", 0.5))
  
  positiveClassPercentage <- mean(y==levels(y)[1])
  naiveAccuracy <- ifelse(positiveClassPercentage>.5,
                          positiveClassPercentage,
                          1-positiveClassPercentage)
  print(paste("Naively guessing most common category: ", naiveAccuracy))
}