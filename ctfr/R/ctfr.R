library(FoReco)
library(forecast)
library(sarima)
values <- NULL
base <- NULL
residuals <- NULL
test <- NULL

bottom <- matrix(NA, nrow = 180, ncol = 5)
# Model definition
bts <- list()
#ARIMA(1,0,0)(0,0,0)[12]
bts[[1]] <- list(ar=0.31,
                 nseasons=12)
#ARIMA(0,0,1)(0,0,0)[12]
bts[[2]] <- list(ma=0.61,
                 nseasons=12)
#ARIMA(0,1,1)(0,1,1)[12]
bts[[3]] <- list(ma=-0.1,
                 sma=-0.12,
                 iorder=1,
                 siorder=1,
                 nseasons=12)
#ARIMA(2,1,0)(0,0,0)[12]
bts[[4]] <- list(ar=c(0.38,0.25),
                 iorder=1,
                 nseasons=12)
#ARIMA(2,0,0)(0,1,1)[12]
bts[[5]] <- list(ar=c(0.30,0.12),
                 sma=0.23,
                 siorder=1,
                 nseasons=12)
mm <- c(58.85, 60.68, 59.26, 35.47, 58.61)
set.seed(525)
for(i in 1:5){
  bottom[,i] <- mm[i] + sim_sarima(n=180, model = bts[[i]],
                                   n.start = 200)
}
colnames(bottom) <- c("AA", "AB", "BA", "BB", "C")
C <- matrix(c(rep(1,5),
              rep(1,2), rep(0,3),
              rep(0,2), rep(1,2), 0), byrow = TRUE, nrow = 3)

upper <- bottom%*%t(C)
colnames(upper) <- c("T", "A", "B")
values$k1 <- ts(cbind(upper, bottom), frequency = 12)
colnames(values$k1) <- c("T", "A", "B", "AA", "AB", "BA", "BB", "C")


# MONTHLY FORECASTS
base$k1 <- matrix(NA, nrow = 12, ncol = ncol(values$k1))
residuals$k1 <- matrix(NA, nrow = 168, ncol = ncol(values$k1))
for (i in 1:ncol(values$k1)) {
  train <- values$k1[1:168, i]
  forecast_arima <- forecast(auto.arima(train), h = 12)
  base$k1[, i] <- forecast_arima$mean
  residuals$k1[, i] <- forecast_arima$residuals
}
base$k1 <- ts(base$k1, frequency = 12, start = c(15, 1))
colnames(base$k1) <- c("T", "A", "B", "AA", "AB", "BA", "BB", "C")
residuals$k1 <- ts(residuals$k1, frequency = 12)
colnames(residuals$k1) <- c("T", "A", "B", "AA", "AB", "BA", "BB", "C")
test$k1 <- values$k1[-c(1:168), ]


# BI-MONTHLY SERIES
values$k2 <- ts(apply(values$k1, 2,
                      function(x) colSums(matrix(x, nrow = 2))),
                frequency = 6)

# QUARTERLY SERIES
values$k3 <- ts(apply(values$k1, 2,
                      function(x) colSums(matrix(x, nrow = 3))),
                frequency = 4)

# FOUR-MONTHLY SERIES
values$k4 <- ts(apply(values$k1, 2,
                      function(x) colSums(matrix(x, nrow = 4))),
                frequency = 3)

# SEMI-ANNUAL SERIES
values$k6 <- ts(apply(values$k1, 2,
                      function(x) colSums(matrix(x, nrow = 6))),
                frequency = 2)

# ANNUAL SERIES
values$k12 <- ts(apply(values$k1, 2,
                       function(x) colSums(matrix(x, nrow = 12))),
                 frequency = 1)



# BI-MONTHLY FORECASTS
base$k2 <- matrix(NA, nrow = 6, ncol = ncol(values$k2))
residuals$k2 <- matrix(NA, nrow = 84, ncol = ncol(values$k2))
for (i in 1:ncol(values$k2)) {
  train <- values$k2[1:84, i]
  forecast_arima <- forecast(auto.arima(train), h = 6)
  base$k2[, i] <- forecast_arima$mean
  residuals$k2[, i] <- forecast_arima$residuals
}
base$k2 <- ts(base$k2, frequency = 6, start = c(15, 1))
colnames(base$k2) <- c("T", "A", "B", "AA", "AB", "BA", "BB", "C")
residuals$k2 <- ts(residuals$k2, frequency = 6)
colnames(residuals$k2) <- c("T", "A", "B", "AA", "AB", "BA", "BB", "C")
test$k2 <- values$k2[-c(1:84), ]


# QUARTERLY FORECASTS
base$k3 <- matrix(NA, nrow = 4, ncol = ncol(values$k3))
residuals$k3 <- matrix(NA, nrow = 56, ncol = ncol(values$k3))
for (i in 1:ncol(values$k3)) {
  train <- values$k3[1:56, i]
  forecast_arima <- forecast(auto.arima(train), h = 4)
  base$k3[, i] <- forecast_arima$mean
  residuals$k3[, i] <- forecast_arima$residuals
}
base$k3 <- ts(base$k3, frequency = 4, start = c(15, 1))
colnames(base$k3) <- c("T", "A", "B", "AA", "AB", "BA", "BB", "C")
residuals$k3 <- ts(residuals$k3, frequency = 4)
colnames(residuals$k3) <- c("T", "A", "B", "AA", "AB", "BA", "BB", "C")
test$k3 <- values$k3[-c(1:56), ]


# FOUR-MONTHLY FORECASTS
base$k4 <- matrix(NA, nrow = 3, ncol = ncol(values$k4))
residuals$k4 <- matrix(NA, nrow = 42, ncol = ncol(values$k4))
for (i in 1:ncol(values$k4)) {
  train <- values$k4[1:42, i]
  forecast_arima <- forecast(auto.arima(train), h = 3)
  base$k4[, i] <- forecast_arima$mean
  residuals$k4[, i] <- forecast_arima$residuals
}
base$k4 <- ts(base$k4, frequency = 3, start = c(15, 1))
colnames(base$k4) <- c("T", "A", "B", "AA", "AB", "BA", "BB", "C")
residuals$k4 <- ts(residuals$k4, frequency = 3)
colnames(residuals$k4) <- c("T", "A", "B", "AA", "AB", "BA", "BB", "C")
test$k4 <- values$k4[-c(1:42), ]

# SEMI-ANNUAL FORECASTS
base$k6 <- matrix(NA, nrow = 2, ncol = ncol(values$k6))
residuals$k6 <- matrix(NA, nrow = 28, ncol = ncol(values$k6))
for (i in 1:ncol(values$k6)) {
  train <- values$k6[1:28, i]
  forecast_arima <- forecast(auto.arima(train), h = 2)
  base$k6[, i] <- forecast_arima$mean
  residuals$k6[, i] <- forecast_arima$residuals
}
base$k6 <- ts(base$k6, frequency = 2, start = c(15, 1))
colnames(base$k6) <- c("T", "A", "B", "AA", "AB", "BA", "BB", "C")
residuals$k6 <- ts(residuals$k6, frequency = 2)
colnames(residuals$k6) <- c("T", "A", "B", "AA", "AB", "BA", "BB", "C")
test$k6 <- values$k6[-c(1:28), ]

# ANNUAL FORECASTS
base$k12 <- matrix(NA, nrow = 1, ncol = ncol(values$k12))
residuals$k12 <- matrix(NA, nrow = 14, ncol = ncol(values$k12))
for (i in 1:ncol(values$k12)) {
  train <- values$k12[1:14, i]
  forecast_arima <- forecast(auto.arima(train), h = 1)
  base$k12[, i] <- forecast_arima$mean
  residuals$k12[, i] <- forecast_arima$residuals
}
base$k12 <- ts(base$k12, frequency = 1, start = c(15, 1))
colnames(base$k12) <- c("T", "A", "B", "AA", "AB", "BA", "BB", "C")
residuals$k12 <- ts(residuals$k12, frequency = 1)
colnames(residuals$k12) <- c("T", "A", "B", "AA", "AB", "BA", "BB", "C")
test$k12 <- values$k12[-c(1:14), ]

base <- t(do.call(rbind, rev(base)))
res <- t(do.call(rbind, rev(residuals)))
test <- t(do.call(rbind, rev(test)))

kset <- c(12, 6, 4, 3, 2, 1)
h <- 1
colnames(base) <- paste("k", rep(kset, h * rev(kset)), "_h",
                        do.call("c", as.list(sapply(
                          rev(kset) * h,
                          function(x) seq(1:x)))),
                        sep = "")


colnames(test) <- paste("k", rep(kset, h * rev(kset)), "_h",
                        do.call("c", as.list(sapply(
                          rev(kset) * h,
                          function(x) seq(1:x)))),
                        sep = "")

h <- 14
colnames(res) <- paste("k", rep(kset, h * rev(kset)), "_h",
                       do.call("c", as.list(sapply(
                         rev(kset) * h,
                         function(x) seq(1:x)))),
                       sep = "")

colnames(C) <- c("AA", "AB", "BA", "BB", "C")
rownames(C) <- c("Tot", "A", "B")
obs <- values
FoReco_data <- list(base = base,
                    test = test,
                    res = res,
                    C = C,
                    obs = obs)

debug(FoReco::octrec)
oct_recf <- octrec(FoReco_data$base, m = 12, C = FoReco_data$C,
                   comb = "ols", res = FoReco_data$res, keep = "recf")
