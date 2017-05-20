# randomSubsampleRMSE(mpg~horsepower, Auto, Auto$mpg, 196)
randomSubsampleRMSE <- function(f, d, y, trainingSize){
  trainingSet=sample(nrow(d), trainingSize)
  model=lm(formula=f ,data=d, subset=trainingSet)
  errors = (y - predict(model, d))[-trainingSet]
  sqrt(mean(errors^2))
}

# crossValidationRMSE(mpg~horsepower, Auto, Auto$mpg, 196, 10)
crossValidationRMSE <- function(f, d, y, trainingSize, numRepetitions){
  meanRMSE = mean(replicate(n, randomSubsampleRMSE(f, d, y, trainingSize)))
  meanRMSE
}