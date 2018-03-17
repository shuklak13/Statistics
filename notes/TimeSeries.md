# Common Workflow
1. Visualize the Time Series
    * to identify trends, seasonality, autoregression, etc.
2. Test if the Time Series is Stationary via the Dickey-Fuller Test (???) - if it isn't, Stationarize it
3. Plot ACF/PACF charts (???) to find optimal ARIMA parameters
4. Build an ARIMA model
5. Make Predictions

# Stationary Series AKA Covariance-Stationary
* Definition: a stochastic process whose distribution is independent of time
* 2 Criterion
    1. Constant Mean
    2. Homoscedasticity (constant variance)
    3. Constant Autocorrelation (covariance of x(i) and x(i+m) is constant with respect to i)
        * by extension, Homoscedasticity (when m=0, cov(x(i), x(i+0)) = var(x(i)), so Constant Autocorrelation means x's variance is constant)
        * you can check autocorrelation with R's `acf()` and `acf.plot()`
* Gaussian White Noise: simplest stationary series - normal distribution with mean 0 with variance independent of time
* Most time series models assume that series are stationary.
    * We can induce stationarity with transformations. We can then reverse these transformations to apply these time series models to our original series.
* Transformations
    * Logging - converts an exponential curve to a linear one
    * Detrending - subtracting a fitted trend line from the series prior to creating a model
        * "Trend-Stationary" - mean trend is deterministic; once removed, the residual series is stationary; shocks are temporary, the series will eventually return to the trend
            * can be represented as Y_t = mu_t + error_t
            * Trend is constant
    * Differencing - instead of modeling values-over-time, model (differences-of-values)-over-time
        * useful if the value series isn't stationary, but the difference-of-values series is
        * "Difference-Stationary" - mean trend is stochastic (random), not deterministic; shocks are permanent, the new values become the new "trend"
            * Differences are constant
    * Removing Seasonality (???)
* to test if a time series is stationary, we can use the Dicker Fuller Test of Stationarity???
    * `adf.test()` in R

# First Difference
* series of differences between i and i+1
* if stationary and completely random (no autocorrelation), use a Random Walk model
* If stationary but not completely random (some autocorrelation), use ARIMA (???)


# Random Walk
* x(t) = x(t-1) + error(t)
    * where the error is assumed to be independently and identically distributed (same for ARMA)
* random walks are NOT stationary series
    * means are constant
    * but variance increase over time
    * also has autocorrelation (Y(i) depends on Y(i-1))


# ARMA and ARIMA Time Series Modelling
* AKA "nonseasonal models"
* ARMA = Autoregressive Moving Average
    * Autoregressive (of Order n)
        * AR1 - next term depends on previous term - x(t) depends on x(t-1)
            * x(t) = alpha * x(t–1) + error(t)
        * ARn - next term depends on last n terms - x(t) depends on x(t-1) through x(t-n)
            * x(t) = Sum_i [alpha_i * x(t–i)] + error(t)
            * correlation between x(t) and x(t-m) decreases for larger m.
            * when m>n, the correlation between x(t) and x(t-m) becomes 0
    * Moving Average (of Order n)
        * x(t) depends on the past error
            * x(t) = beta * error(t-1) + error(t)
        * just like ARn, you can also account for multiple past errors with MAn
            * x(t) = Sum_i [beta_i * error(t–i)] + error(t)
        * Exponential Smoothing
            * modification on Moving Average to exponentially decrease the weight of older errors over time (rather than assigning each of the past n errors an individual weight beta_i, as standard MA does)
            * x(t) = beta * error(t-1) + (1-beta) * error(t)
                * here, beta is known as the "smoothing factor"
    * ARMA combines AR and MA
        * ARMA(1,1): x(t) = alpha * x(t–1) + beta * error(t-1) + error(t)
        * ARMA(1,0): Random Walk
        * ARMA(p, q): x(t) = Sum_i [alpha_i * x(t–i)] + Sum_i [beta_i * error(t–i)] + error(t)
* ARIMA = Autoregressive Integrated Moving Average = Box-Jenkins Modeling
    * Integrated
        * "integrate" Differencing by subtracting the previous value
        * Ex: ARIMA(p=0, q=0, d=1): x(t) = x(t) - x(t-1)
        * Ex: ARIMA(p=0, q=0, d=2): x(t) = (x(t) - x(t-1)) - (x(t-1) - x(t-2))
    * the p and q in ARIMA(p, q, d) are the orders of AR and MA (same as ARMA)
* R's `arima()` can be used to create a fitted model whos output can be used to predict the next n data points with `predict(arima_model, n.ahead)`. You can then plot this result with `ts.plot()`
* [Some examples of common ARIMA models](https://people.duke.edu/~rnau/411arim.htm)


# Exponential Smoothing
* a model that ___???