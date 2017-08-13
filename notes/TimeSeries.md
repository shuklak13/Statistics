# Time Series

## Stationary Series
* a stochastic process whose districtuion is independent of time (verify this?)
* if a time series is not stationary, then you cannot build a time series model!

### Common Violations
* trends in the mean (what does this mean?)

### How to Bring Stationarity
#### Detrending
*

#### Differencing
*

## Random Walk
* random walks are NOT stationary series - their means stay constant, but their variances increase over time!
* Dicker Fuller Test of Stationarity???

## ARMA Time Series Modelling (??? ask dad if this is correct)
* Autoregressive Moving Average
* Autoregressive
    * AR1 - x(t) depends on x(t-1)
        * x(t) = alpha * x(t–1) + error(t)
    * ARn - x(t) depends on x(t-1) through x(t-n)
    * shock decreases slowly over time
* Moving Average
    * x(t) depends on the past error
        * x(t) = beta * error(t-1) + error (t)
    * shock decreases quickly, because error rebounds
