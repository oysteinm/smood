getwd()
setwd("C:/temp/smood")
dir()

library(tidyverse)
library(janitor)
library(lubridate)
library(KFAS)

# The Index of Consumer Sentiment 
# University of Michigan, Survey Research Center
#https://data.sca.isr.umich.edu/#

# Monthly data
ics_month <- read_csv("scaum-479.csv", col_types = cols(yyyymm = col_date(format = "%Y%m")))
ics_month

# Data for analysis
ics_data <- 
  ics_month %>% 
  select(yyyymm, ics_all) %>%
  rename(date = yyyymm, ics = ics_all)

# Individual response data
ics_month_id <- read_csv("AAk7MRJC.csv", col_types = cols(YYYYMM = col_date(format = "%Y%m")))

# Clean column names
ics_month_id <- ics_month_id %>% clean_names()
ics_month_id

ics_month_id %>% 
  group_by(method) %>%
  summarise(
    n = n())

# How many NA are there across all columns?
ics_month_id %>%
  summarise(across(everything(), ~ sum(is.na(.))))

# Select relevant columns and rename
ics_data_id <- 
  ics_month_id %>% 
  select(yyyymm, id, ics, wt) %>%
  rename(date = yyyymm)

ics_data_id

ics_data_id %>%
  group_by(date) %>%
  summarise(
    n = n())

ics_data_id %>%
  filter(is.finite(ics), is.finite(wt), wt > 0) %>%
  group_by(date) %>%
  summarise(
    n = n(),
    w_sum  = sum(wt),
    w2_sum = sum(wt^2),
    n_eff  = (w_sum^2) / w2_sum,
    deff   = n / n_eff   # design effect
  ) %>%
  summarise(
    n_med = median(n), neff_med = median(n_eff),
    deff_med = median(deff),
    deff_p90 = quantile(deff, 0.90)
  )


# weighted ics by month
ics_weighted <- 
  ics_data_id %>%
  group_by(date) %>%
  summarise(
    weighted_ics = sum(ics * wt) / sum(wt)
  )


# Join weighted ics with published ics
# published ics is missing for last month
test <- ics_weighted %>% left_join(ics_data, by = "date")

# Compare weighted ics with published ics
test %>%
  ggplot(aes(x = date)) +
  geom_line(aes(y = weighted_ics, color = "Weighted ICS")) +
  geom_line(aes(y = ics, color = "Published ICS")) +
  ggtitle("Comparison of Weighted ICS and Published ICS") +
  ylab("Index of Consumer Sentiment") +
  theme_minimal() +
  scale_color_manual(values = c("Weighted ICS" = "blue", "Published ICS" = "red")) +
  labs(color = "Legend")

# correlation between weighted ics and published ics
# They are very identical
cor(test$weighted_ics, test$ics, use = "complete.obs")
cor(round(test$weighted_ics,1), test$ics, use = "complete.obs")


# Plot the Index of Consumer Sentiment over time
ics_data %>%
  ggplot(aes(x = date, y = ics)) +
  geom_line() + 
  ggtitle("Index of Consumer Sentiment") +
  theme_minimal() 


#---------------------------------------------

# 1) Monthly weighted mean + effective N and measurement variance
ics_monthly <- 
  ics_data_id %>%
  filter(is.finite(ics), is.finite(wt), wt > 0) %>%
    group_by(date) %>%
    summarise(
    n        = n(),
    w_sum    = sum(wt),
    w2_sum   = sum(wt^2),
    n_eff    = (w_sum^2) / pmax(w2_sum, 1e-12),
    ics_hat  = sum(wt * ics) / w_sum,
    denom    = pmax(w_sum - (w2_sum / w_sum), 1e-8),           # weighted Bessel denom
    s2_w     = sum(wt * (ics - ics_hat)^2) / denom,            # weighted variance of units
    var_mean = s2_w / pmax(n_eff, 1e-8),                       # Var( monthly mean )
    se_mean  = sqrt(pmax(var_mean, 1e-12)),
    .groups = "drop"
  ) %>% arrange(date)


ics_monthly

# y, H_vec from your ics_monthly:
y     <- as.numeric(ics_monthly$ics_hat)
H_vec <- as.numeric(ics_monthly$var_mean)

# 1) basic guards
stopifnot(length(y) == length(H_vec))

# replace NA/Inf or nonpositive with a reasonable floor
good       <- is.finite(H_vec) & H_vec > 0
fallback   <- stats::median(H_vec[good], na.rm = TRUE)
if (!is.finite(fallback)) fallback <- stats::var(y, na.rm = TRUE) * 0.05
H_vec[!good] <- fallback
# gentle floor (avoid extremely tiny numbers)
floor_val <- stats::quantile(H_vec, 0.05, na.rm = TRUE) * 0.1
if (!is.finite(floor_val) || floor_val <= 0) floor_val <- fallback * 0.1
H_vec <- pmax(H_vec, floor_val)

# 2) ensure NO NA left
stopifnot(!any(is.na(H_vec)))

# 3) shape to 1x1xT array
H_arr <- array(H_vec, dim = c(1, 1, length(H_vec)))

# Build initial model with unknown Q
init_model <- SSModel(
  y ~ SSMtrend(degree = 1, Q = NA),
  H = H_arr
)

# Update function: parameterize Q = exp(theta) to keep it > 0
update_fn <- function(par, model) {
  model$Q[1, 1, 1] <- exp(par[1])
  model
}

# Initial value for theta (log Q)
theta0 <- log(stats::var(y, na.rm = TRUE) * 0.05)

fit <- fitSSM(
  model    = init_model,
  inits    = theta0,
  updatefn = update_fn,
  method   = "BFGS"
)

kfs <- KFS(fit$model, smoothing = c("state","disturbance"))

ics_sm    <- as.numeric(kfs$alphahat[, 1])
ics_sm_se <- sqrt(as.numeric(kfs$V[1, 1, ]))

ics_out <- ics_monthly %>%
  mutate(
    ics_sm    = ics_sm,
    ics_sm_lo = ics_sm - 1.96 * ics_sm_se,
    ics_sm_hi = ics_sm + 1.96 * ics_sm_se
  )

# Combine results
# 'ics_out' from previous step (date, ics_raw=ics_hat, ics_sm, ics_sm_lo, ics_sm_hi, n, n_eff, ...)

# Join to published index (if you loaded ics_data earlier)
ics_cmp <- ics_out %>% left_join(ics_data, by = "date")  # adds `ics` (published)

# Save for downstream use
#write.csv(ics_cmp, "ics_monthly_smoothed.csv", row.names = FALSE)

# need better color distinction
# Plot raw vs smoothed ICS
ics_out %>% 
ggplot(aes(date)) +
  geom_ribbon(aes(ymin = ics_sm_lo, ymax = ics_sm_hi), fill = "lightblue", alpha = 0.5) +
  geom_line(aes(y = ics_hat, color = "Raw ICS"), size = 1) +
  geom_line(aes(y = ics_sm, color = "Smoothed ICS"), size = 1) +
  ggtitle("Index of Consumer Sentiment: Raw vs Smoothed") +
  ylab("Index of Consumer Sentiment") +
  theme_minimal() +
  scale_color_manual(values = c("Raw ICS" = "red", "Smoothed ICS" = "blue")) +
  labs(color = "Legend")

# correlation between raw and smoothed ICS
cor(ics_out$ics_hat, ics_out$ics_sm, use = "complete.obs")
cor(round(ics_out$ics_hat,1), round(ics_out$ics_sm,1), use = "complete.obs")


# Simple LOESS
ics_data %>%
  mutate(
    ics_loess = predict(loess(ics ~ as.numeric(date), span = 0.1)),
    # span controls smoothness: 0.05-0.2 typical
  )

# With confidence bands (bootstrap or standard errors)
loess_fit <- loess(ics ~ as.numeric(date), data = ics_data, span = 0.1)
pred <- predict(loess_fit, se = TRUE)

ics_data %>%
  mutate(
    ics_loess = pred$fit,
    ics_loess_lo = pred$fit - 1.96 * pred$se.fit,
    ics_loess_hi = pred$fit + 1.96 * pred$se.fit
  )

# Plot LOESS smoothed ICS and ics
ics_data %>%
  mutate(
    ics_loess = pred$fit,
    ics_loess_lo = pred$fit - 1.96 * pred$se.fit,
    ics_loess_hi = pred$fit + 1.96 * pred$se.fit
  ) %>%
  ggplot(aes(x = date)) +
  geom_ribbon(aes(ymin = ics_loess_lo, ymax = ics_loess_hi), fill = "lightgreen", alpha = 0.5) +
  geom_line(aes(y = ics, color = "Raw ICS"), size = 1) +
  geom_line(aes(y = ics_loess, color = "LOESS Smoothed ICS"), size = 1) +
  ggtitle("Index of Consumer Sentiment: Raw vs LOESS Smoothed") +
  ylab("Index of Consumer Sentiment") +
  theme_minimal() +
  scale_color_manual(values = c("Raw ICS" = "red", "LOESS Smoothed ICS" = "green")) +
  labs(color = "Legend")


# ---------------------------------------------

# Now we don't know H either, so estimate both Q and H
y <- as.numeric(ics_data$ics)

init_model <- SSModel(
  y ~ SSMtrend(degree = 1, Q = NA),
  H = NA  # Unknown measurement variance
)

update_fn <- function(par, model) {
  model$Q[1, 1, 1] <- exp(par[1])
  model$H[1, 1, 1] <- exp(par[2])
  model
}

# Initial values
theta0 <- c(
  log(var(y, na.rm = TRUE) * 0.01),  # Q (process variance)
  log(var(y, na.rm = TRUE) * 0.05)   # H (measurement variance)
)

fit <- fitSSM(
  model = init_model,
  inits = theta0,
  updatefn = update_fn,
  method = "BFGS"
)

kfs <- KFS(fit$model, smoothing = c("state"))

ics_data %>%
  mutate(
    ics_kalman = as.numeric(kfs$alphahat[, 1]),
    ics_kalman_se = sqrt(as.numeric(kfs$V[1, 1, ])),
    ics_kalman_lo = ics_kalman - 1.96 * ics_kalman_se,
    ics_kalman_hi = ics_kalman + 1.96 * ics_kalman_se
  )

# Plot Kalman smoothed ICS
ics_data %>%
  mutate(
    ics_kalman = as.numeric(kfs$alphahat[, 1]),
    ics_kalman_se = sqrt(as.numeric(kfs$V[1, 1, ])),
    ics_kalman_lo = ics_kalman - 1.96 * ics_kalman_se,
    ics_kalman_hi = ics_kalman + 1.96 * ics_kalman_se
  ) %>%
  ggplot(aes(x = date)) +
  geom_ribbon(aes(ymin = ics_kalman_lo, ymax = ics_kalman_hi), fill = "lightcoral", alpha = 0.5) +
  geom_line(aes(y = ics, color = "Raw ICS"), size = 1) +
  geom_line(aes(y = ics_kalman, color = "Kalman Smoothed ICS"), size = 1) +
  ggtitle("Index of Consumer Sentiment: Raw vs Kalman Smoothed") +
  ylab("Index of Consumer Sentiment") +
  theme_minimal() +
  scale_color_manual(values = c("Raw ICS" = "red", "Kalman Smoothed ICS" = "purple")) +
  labs(color = "Legend")

# correlation between raw and kalman ICS
cor(ics_data$ics, as.numeric(kfs$alphahat[, 1]), use = "complete.obs")
cor(round(ics_data$ics,1), round(as.numeric(kfs$alphahat[, 1]),1), use = "complete.obs")

# After fitting, check the estimated parameters
cat("Estimated Q:", exp(fit$optim.out$par[1]), "\n")
cat("Estimated H:", exp(fit$optim.out$par[2]), "\n")
cat("Ratio Q/H:", exp(fit$optim.out$par[1]) / exp(fit$optim.out$par[2]), "\n")

library(mFilter)

hp_fit <- hpfilter(ics_data$ics, freq = 14400)  # Monthly: 14400, Quarterly: 1600

hp_fit

ics_data %>%
  mutate(
    ics_trend = hp_fit$trend,
    ics_cycle = hp_fit$cycle
  )

# Plot HP filter trend vs raw ICS
ics_data %>%
  mutate(
    ics_trend = hp_fit$trend,
    ics_cycle = hp_fit$cycle
  ) %>%
  ggplot(aes(x = date)) +
  geom_line(aes(y = ics, color = "Raw ICS"), size = 1) +
  geom_line(aes(y = ics_trend, color = "HP Filter Trend"), size = 1) +
  ggtitle("Index of Consumer Sentiment: Raw vs HP Filter Trend") +
  ylab("Index of Consumer Sentiment") +
  theme_minimal() +
  scale_color_manual(values = c("Raw ICS" = "red", "HP Filter Trend" = "orange")) +
  labs(color = "Legend")



library(mgcv)

# Generalized Additive Model with automatic smoothing
gam_fit <- gam(ics ~ s(as.numeric(date), bs = "ps"), data = ics_data)

pred <- predict(gam_fit, se.fit = TRUE)

ics_data %>%
  mutate(
    ics_spline = pred$fit,
    ics_spline_lo = pred$fit - 1.96 * pred$se.fit,
    ics_spline_hi = pred$fit + 1.96 * pred$se.fit
  )

# Plot GAM smoothed ICS
ics_data %>%
  mutate(
    ics_spline = pred$fit,
    ics_spline_lo = pred$fit - 1.96 * pred$se.fit,
    ics_spline_hi = pred$fit + 1.96 * pred$se.fit
  ) %>%
  ggplot(aes(x = date)) +
  geom_ribbon(aes(ymin = ics_spline_lo, ymax = ics_spline_hi), fill = "lightyellow", alpha = 0.5) +
  geom_line(aes(y = ics, color = "Raw ICS"), size = 1) +
  geom_line(aes(y = ics_spline, color = "GAM Smoothed ICS"), size = 1) +
  ggtitle("Index of Consumer Sentiment: Raw vs GAM Smoothed") +
  ylab("Index of Consumer Sentiment") +
  theme_minimal() +
  scale_color_manual(values = c("Raw ICS" = "red", "GAM Smoothed ICS" = "darkgreen")) +
  labs(color = "Legend")


# Auto-select LOESS span via cross-validation

library(fANCOVA)
optimal_span <- loess.as(x = as.numeric(ics_data$date), 
                         y = ics_data$ics,
                         criterion = "gcv")$pars$span

optimal_span
# Fit LOESS with optimal span
loess_fit_opt <- loess(ics ~ as.numeric(date), data = ics_data, span = optimal_span)
pred_opt <- predict(loess_fit_opt, se = TRUE)

# Plot LOESS with optimal span
ics_data %>%
  mutate(
    ics_loess_opt = pred_opt$fit,
    ics_loess_opt_lo = pred_opt$fit - 1.96 * pred_opt$se.fit,
    ics_loess_opt_hi = pred_opt$fit + 1.96 * pred_opt$se.fit
  ) %>%
  ggplot(aes(x = date)) +
  geom_ribbon(aes(ymin = ics_loess_opt_lo, ymax = ics_loess_opt_hi), fill = "lightpink", alpha = 0.5) +
  geom_line(aes(y = ics, color = "Raw ICS"), size = 1) +
  geom_line(aes(y = ics_loess_opt, color = "Optimal LOESS Smoothed ICS"), size = 1) +
  ggtitle(paste("Index of Consumer Sentiment: Raw vs LOESS Smoothed (Span =", round(optimal_span,3), ")")) +
  ylab("Index of Consumer Sentiment") +
  theme_minimal() +
  scale_color_manual(values = c("Raw ICS" = "red", "Optimal LOESS Smoothed ICS" = "darkblue")) +
  labs(color = "Legend")
# correlation between raw and optimal LOESS ICS
cor(ics_data$ics, pred_opt$fit, use = "complete.obs")
cor(round(ics_data$ics,1), round(pred_opt$fit,1), use = "complete.obs")
  
    
# ---------------------------------------------    
    
    
smooth_ts <- function(df_ts) {
  # 1. Sort and extract
  df_sorted <- df_ts %>% arrange(date)
  y <- df_sorted %>% pull(ics)
  
  # 2. Guards
  if (length(y) < 10) {
    warning("Too few observations for reliable smoothing")
    return(df_sorted %>% mutate(ics_sm = ics, ics_sm_lo = NA, ics_sm_hi = NA))
  }
  
  if (any(!is.finite(y))) {
    stop("Non-finite values in ics column")
  }
  
  # 3. Build and fit model
  init_model <- SSModel(y ~ SSMtrend(1, Q = NA), H = NA)
  
  update_fn <- function(par, model){
    model$Q[1,1,1] <- exp(par[1])
    model$H[1,1,1] <- exp(par[2])
    model
  }
  
  # Initial values: Q smaller than H (smoother trend)
  var_y <- var(y, na.rm = TRUE)
  theta0 <- log(c(var_y * 0.01, var_y * 0.1))
  
  fit <- tryCatch(
    fitSSM(model = init_model, inits = theta0, updatefn = update_fn, method = "BFGS"),
    error = function(e) {
      warning("Optimization failed: ", e$message)
      return(NULL)
    }
  )
  
  if (is.null(fit)) {
    return(df_sorted %>% mutate(ics_sm = ics, ics_sm_lo = NA, ics_sm_hi = NA))
  }
  
  # 4. Extract smoothed values
  kfs <- KFS(fit$model, smoothing = c("state"))
  
  ics_sm    <- as.numeric(kfs$alphahat[,1])
  ics_sm_se <- sqrt(pmax(as.numeric(kfs$V[1,1,]), 0))  # guard against tiny negatives
  
  # 5. Return results
  df_sorted %>%
    mutate(
      ics_sm = ics_sm,
      ics_sm_lo = ics_sm - 1.96 * ics_sm_se,
      ics_sm_hi = ics_sm + 1.96 * ics_sm_se,
      # Optional diagnostics
      Q_hat = exp(fit$optim.out$par[1]),
      H_hat = exp(fit$optim.out$par[2])
    )
}

# Usage
smoothed <- smooth_ts(ics_data)    
    
smoothed %>%
  ggplot(aes(x = date)) +
  geom_ribbon(aes(ymin = ics_sm_lo, ymax = ics_sm_hi), fill = "lightgray", alpha = 0.5) +
  geom_line(aes(y = ics, color = "Raw ICS"), size = 1) +
  geom_line(aes(y = ics_sm, color = "Smoothed ICS"), size = 1) +
  ggtitle("Index of Consumer Sentiment: Raw vs Smoothed (Function)") +
  ylab("Index of Consumer Sentiment") +
  theme_minimal() +
  scale_color_manual(values = c("Raw ICS" = "red", "Smoothed ICS" = "blue")) +
  labs(color = "Legend")
    


# Set smoothness level (lower = more smoothing)
snr <- 0.05  # 5% signal, 95% noise - try 0.01, 0.05, 0.1, 0.2

y <- as.numeric(ics_data$ics)

init_model <- SSModel(
  y ~ SSMtrend(degree = 1, Q = NA),
  H = NA
)

update_fn <- function(par, model) {
  sigma2 <- exp(par[1])  # overall variance scale
  model$Q[1, 1, 1] <- snr * sigma2       
  model$H[1, 1, 1] <- (1 - snr) * sigma2  
  model
}

theta0 <- log(var(y, na.rm = TRUE))

fit <- fitSSM(
  model = init_model,
  inits = theta0,
  updatefn = update_fn,
  method = "BFGS"
)

kfs <- KFS(fit$model, smoothing = c("state"))

# Check diagnostics
cat("SNR (fixed):", snr, "\n")
cat("Estimated sigma2:", exp(fit$optim.out$par[1]), "\n")
cat("Implied Q:", snr * exp(fit$optim.out$par[1]), "\n")
cat("Implied H:", (1-snr) * exp(fit$optim.out$par[1]), "\n")
cat("Ratio Q/H:", snr/(1-snr), "\n")

# Create smoothed data
ics_smoothed <- ics_data %>%
  mutate(
    ics_kalman = as.numeric(kfs$alphahat[, 1]),
    ics_kalman_se = sqrt(as.numeric(kfs$V[1, 1, ])),
    ics_kalman_lo = ics_kalman - 1.96 * ics_kalman_se,
    ics_kalman_hi = ics_kalman + 1.96 * ics_kalman_se
  )

# Now check correlation (should be lower)
cor(ics_smoothed$ics, ics_smoothed$ics_kalman)

# Plot
ics_smoothed %>%
  ggplot(aes(x = date)) +
  geom_ribbon(aes(ymin = ics_kalman_lo, ymax = ics_kalman_hi), 
              fill = "lightcoral", alpha = 0.5) +
  geom_line(aes(y = ics, color = "Raw ICS"), size = 1) +
  geom_line(aes(y = ics_kalman, color = "Kalman Smoothed"), size = 1) +
  ggtitle("Index of Consumer Sentiment: Raw vs Kalman Smoothed (SNR = 0.05)") +
  theme_minimal() +
  scale_color_manual(values = c("Raw ICS" = "red", "Kalman Smoothed" = "purple"))


library(kernlab)

# Fit Gaussian Process
gp_fit <- gausspr(
  x = as.numeric(ics_data$date), 
  y = ics_data$ics,
  kernel = "rbfdot",     # RBF kernel (smooth)
  kpar = "automatic",     # auto-tune length scale
  variance.model = TRUE   # get uncertainty bands
)

# Predict
pred <- predict(gp_fit, as.numeric(ics_data$date), type = "response")
pred_var <- predict(gp_fit, as.numeric(ics_data$date), type = "variance")

ics_gp <- ics_data %>%
  mutate(
    ics_gp = as.numeric(pred),
    ics_gp_se = sqrt(pred_var),
    ics_gp_lo = ics_gp - 1.96 * ics_gp_se,
    ics_gp_hi = ics_gp + 1.96 * ics_gp_se
  )

# Plot
ggplot(ics_gp, aes(x = date)) +
  geom_ribbon(aes(ymin = ics_gp_lo, ymax = ics_gp_hi), fill = "lightblue", alpha = 0.5) +
  geom_line(aes(y = ics, color = "Raw"), size = 0.5) +
  geom_line(aes(y = ics_gp, color = "GP Smoothed"), size = 1) +
  theme_minimal()
