# Previsão da Temperatura da Superfície do Oceano
# Localização 4N 23W

rm(list=ls())

# Bibliotecas -------------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(modeltime)
library(modeltime.resample)
library(modeltime.ensemble)
library(timetk)
library(lubridate)
library(zoo)
library(forecast)
library(tseries)


# Importando Dados 4n23w(10.05.2007 - 27.03.2019) ----------------------------
local = file.choose()
sst_4n23w = read.table(file = local, header=TRUE, dec=".")
sst_4n23w = sst_4n23w[,-1]
sst_4n23w$date = as.Date(sst_4n23w$date, format = "%Y-%m-%d")
attach(sst_4n23w)

teste_sst_4n23w <- ts(sst_4n23w[, 2], frequency = 365) # utilizado no teste de Dickey Fuller


# Transformando Dados -----------------------------------------------------

# Criando variável mês
sst_4n23w = sst_4n23w %>%
  mutate(month = month(date, label = TRUE))

# Transformando mês em fator
sst_4n23w$month = factor(sst_4n23w$month, ordered = FALSE)

# Criando dummies (dos meses) e uma série de Fourier (datas, com frequência trimestral e ordem 2)
rec_feat_eng = recipe(value ~ ., data = sst_4n23w) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_fourier(date, period = c(365/4, 365), K = 2)

sst_4n23w = rec_feat_eng %>%
  prep() %>%
  bake(new_data = NULL)
sst_4n23w

# Retirando o primeira dummy para evitar multicolineariedade
sst_4n23w = sst_4n23w %>%
  select(-month_Jan)

# Gráfico da Série Temporal
sst_4n23w %>%
  plot_time_series(date, value, .interactive = FALSE, 
                   .title = "(b) TSO Boia 4ºN 23ºW",
                   .y_lab = "TSO (ºC)",
                   .x_lab = "Tempo (diário)")


# Testes de Estacionariedade ----------------------------------------------

# Teste de Phillips-Perron (PP) 
adf.test(teste_sst_4n23w)


# Criando grupos de treinamento e teste (12 meses) ------------------------
set.seed(123)

splits = sst_4n23w %>%
  time_series_split(assess = "12 months", cumulative = TRUE)

# Gráfico grupos de treinamento e teste
splits %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(date, value,
                           .interactive = FALSE,
                           .title = NULL,
                           .y_lab = "TSO (ºC)",
                           .x_lab = "Tempo (diário)")

# Modelagem pelo modeltime ------------------------------------------------


# Criando fórmula para utilizar as va. exógenas (dummies e as séries de Fourier) nos modelos
fmla = as.formula(paste0("value ~ date + ", paste0(names(sst_4n23w)[-c(1,2)], collapse = " + ")))

# Modelo 1: PROPHET com dados de treinamento (valor e data)
model_prophet <- prophet_reg(
  changepoint_num = 50
) %>%
  set_engine("prophet") %>%
  fit(value ~ date, training(splits))

# Modelo 2: PROPHET com variáveis exógenas
model_prophetx <- prophet_reg(
  changepoint_num = 50
) %>%
  set_engine("prophet") %>%
  fit(fmla, training(splits))

# Modelo 3: PROPHET Boost com variáveis exógenas
model_prophet_boost <- prophet_boost(
  changepoint_num = 50,
  learn_rate = .005,
  tree_depth = 10
) %>%
  set_engine("prophet_xgboost") %>%
  fit(fmla, training(splits))

# Modelo 4: Suavização Exponencial
model_ets <- exp_smoothing()%>%
  set_engine("ets") %>%
  fit(value ~ date, training(splits))

# Modelo 5: ARIMA
model_arima <- arima_reg() %>%
  set_engine("auto_arima") %>%
  fit(value ~ date, training(splits))

# Modelo 6: ARIMAX
model_arimax <- arima_reg() %>%
  set_engine("auto_arima") %>%
  fit(fmla, training(splits))

# Modelo 7: ARIMA Boost
model_arima_boost <- arima_boost(
  # ARIMA args
  seasonal_period = 7,
  non_seasonal_ar = 3,
  non_seasonal_differences = 0,
  non_seasonal_ma = 0,
  seasonal_ar     = 0,
  seasonal_differences = 0,
  seasonal_ma     = 1,
  
  # XGBoost Args
  tree_depth = 10,
  learn_rate = 0.005
) %>%
  set_engine("arima_xgboost") %>%
  fit(fmla, training(splits))

# Modelo 8: Tendência Sazonal Usando LOES com SE
model_stlm_ets <- seasonal_reg() %>%
  set_engine("stlm_ets") %>%
  fit(value ~ date, training(splits))

# Modelo 9: Tendência Sazonal Usando LOES com ARIMA
model_stlm_arima <- seasonal_reg() %>%
  set_engine("stlm_arima") %>%
  fit(value ~ date, training(splits))

# Modelo 10: Tendência Sazonal Usando LOES com ARIMAX
model_stlm_arimax <- seasonal_reg() %>%
  set_engine("stlm_arima") %>%
  fit(fmla, training(splits))

# Modelo 11: TABTS
model_stlm_tbats <- seasonal_reg() %>%
  set_engine("tbats") %>%
  fit(value ~ date, training(splits))

# Modelo 12: Perceptron Multicamadas com 5 camadas ocultas (parâmetros retornados em model_arima)
model_nnetar5 <- nnetar_reg(
  seasonal_period = 7,
  non_seasonal_ar = 3,
  seasonal_ar = 0,
  hidden_units = 5,
  num_networks = 30
) %>%
  set_engine("nnetar") %>%
  fit(fmla, training(splits))

# Modelo 13: Perceptron Multicamadas com 10 camadas ocultas (parâmetros retornados em model_arima)
model_nnetar10 <- nnetar_reg(
  seasonal_period = 7,
  non_seasonal_ar = 3,
  seasonal_ar = 0,
  hidden_units = 10,
  num_networks = 30
) %>%
  set_engine("nnetar") %>%
  fit(fmla, training(splits))

# Modelo 14: Perceptron Multicamadas com 10 camadas ocultas (parâmetros retornados em model_arima)
model_nnetar15 <- nnetar_reg(
  seasonal_period = 7,
  non_seasonal_ar = 3,
  seasonal_ar = 0,
  hidden_units = 15,
  num_networks = 30
) %>%
  set_engine("nnetar") %>%
  fit(fmla, training(splits))

# Modelo 15: Perceptron Multicamadas com 20 camadas ocultas (parâmetros retornados em model_arima)
model_nnetar20 <- nnetar_reg(
  seasonal_period = 7,
  non_seasonal_ar = 3,
  seasonal_ar = 0,
  hidden_units = 20,
  num_networks = 30
) %>%
  set_engine("nnetar") %>%
  fit(fmla, training(splits))

# Modelo 16: Perceptron Multicamadas com 30 camadas ocultas (parâmetros retornados em model_arima)
model_nnetar30 <- nnetar_reg(
  seasonal_period = 7,
  non_seasonal_ar = 3,
  seasonal_ar = 0,
  hidden_units = 30,
  num_networks = 30
) %>%
  set_engine("nnetar") %>%
  fit(fmla, training(splits))

# Modelo 17: Floresta Aleatória
model_rf <- rand_forest() %>%
  set_mode("regression") %>%
  set_engine("ranger") %>%
  fit(fmla, training(splits))

# Modelo 18: SVM
model_svm <- svm_rbf() %>%
  set_mode("regression") %>%
  set_engine("kernlab") %>%
  fit(fmla, training(splits))

# Modelo 19: GBT
model_bt <- boost_tree(
  tree_depth = 10,
  learn_rate = .5
) %>%
  set_mode("regression") %>%
  set_engine("xgboost") %>%
  fit(fmla, training(splits))

# Modelo 20: Elastic Net
# (penalty = 1, mixture = 1)
model_glmnet1 <- linear_reg(
  penalty = .1,
  mixture = .1
) %>%
  set_mode("regression") %>%
  set_engine("glmnet") %>%
  fit(fmla, training(splits))

# Modelo 21: Elastic Net
# (penalty = 1, mixture = 3)
model_glmnet2 <- linear_reg(
  penalty = .1,
  mixture = .3
) %>%
  set_mode("regression") %>%
  set_engine("glmnet") %>%
  fit(fmla, training(splits))

# Modelo 22: Elastic Net
# (penalty = 1, mixture = 5)
model_glmnet3 <- linear_reg(
  penalty = .1,
  mixture = .5
) %>%
  set_mode("regression") %>%
  set_engine("glmnet") %>%
  fit(fmla, training(splits))

# Modelo 23: Elastic Net
# (penalty = 3, mixture = 1)
model_glmnet4 <- linear_reg(
  penalty = .3,
  mixture = .1
) %>%
  set_mode("regression") %>%
  set_engine("glmnet") %>%
  fit(fmla, training(splits))

# Modelo 24: Elastic Net
# (penalty = 3, mixture = 3)
model_glmnet5 <- linear_reg(
  penalty = .3,
  mixture = .3
) %>%
  set_mode("regression") %>%
  set_engine("glmnet") %>%
  fit(fmla, training(splits))

# Modelo 25: Elastic Net
# (penalty = 3, mixture = 5)
model_glmnet6 <- linear_reg(
  penalty = .3,
  mixture = .5
) %>%
  set_mode("regression") %>%
  set_engine("glmnet") %>%
  fit(fmla, training(splits))

# Modelo 26: Regressão ADAM (parâmetros retornados em model_arimax)
model_adam <- adam_reg(
  seasonal_period = 7,
  non_seasonal_ar = 3,
  non_seasonal_differences = 0,
  non_seasonal_ma = 0,
  seasonal_ar     = 0,
  seasonal_differences = 0,
  seasonal_ma     = 1
) %>%
  set_mode("regression") %>%
  set_engine("adam") %>%
  fit(fmla, training(splits))

# tabela geral de resultados
model_tbl <- modeltime_table(
  model_arima,
  model_arimax,
  model_arima_boost,
  model_ets,
  model_stlm_ets,
  model_stlm_arima,
  model_stlm_arimax,
  model_stlm_tbats,
  model_nnetar5,
  model_nnetar10,
  model_nnetar15,
  model_nnetar20,
  model_nnetar30,
  model_rf,
  model_svm,
  model_bt,
  model_glmnet1,
  model_glmnet2,
  model_glmnet3,
  model_glmnet4,
  model_glmnet5,
  model_glmnet6,
  model_adam,
  model_prophet,
  model_prophetx,
  model_prophet_boost
)%>% 
  update_modeltime_description(rep(1:26), c("ARIMA(3,0,0)(0,0,1)[7]",
                                            "ARIMAX(1,1,3)(0,0,1)[7]",
                                            "ARIMA Boost(3,0,0)(0,0,1)",
                                            "SE(A,N,N)",
                                            "STL SE(A,N,N)",
                                            "STL ARIMA(3,0,2)",
                                            "STL ARIMAX(3,1,3)",
                                            "TBATS",
                                            "Perceptron (5 camadas)",
                                            "Perceptron (10 camadas)",
                                            "Perceptron (15 camadas)",
                                            "Perceptron (20 camadas)",
                                            "Perceptron (30 camadas)",
                                            "Floresta Aleatória",
                                            "SVM",
                                            "Gradient Tree Boost",
                                            "Elastic Net (λ = 0,1, α = 0,1)",
                                            "Elastic Net (λ = 0,1, α = 0,3)",
                                            "Elastic Net (λ = 0,1, α = 0,5)",
                                            "Elastic Net (λ = 0,3, α = 0,1)",
                                            "Elastic Net (λ = 0,3, α = 0,3)",
                                            "Elastic Net (λ = 0,3, α = 0,5)",
                                            "ADAM",
                                            "Prophet",
                                            "Prophet (va. exógenas)",
                                            "Prophet Boost (va. exógenas)"))
  
# Calibrate ----
calib_tbl <- model_tbl %>%
  modeltime_calibrate(testing(splits))

# Accuracy ----
meas = calib_tbl %>% 
  modeltime_accuracy(metric_set = metric_set(mae,rmse))
                               
print(meas, n = 30)

# Melhores 10 modelos
meas_sorted = meas %>%
  arrange(mae)
meas_sorted

meas_sorted_rmse = meas %>%
  arrange(rmse)
meas_sorted_rmse

# Test Set Visualization ----
calib_tbl %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = sst_4n23w,
    conf_interval = 0
  ) %>%
  plot_modeltime_forecast(.interactive = FALSE,
                          .title = NULL,
                          .color_lab = "Legenda")

# Test Set Visualization ---- Zoom
calib_tbl %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = sst_4n23w,
    conf_interval = 0
  ) %>%
  plot_modeltime_forecast(.interactive = FALSE,
                          .title = NULL,
                          .color_lab = "Legenda") +
  scale_x_date(limits = as.Date(c("2018-03-20","2019-03-27"), format = "%Y-%m-%d")) +
  scale_y_continuous(limits = c(26,30))

# Tabela dos melhores modelos. Construir de acordo com o MAE da tabela geral
best_models_tbl <- modeltime_table(
  model_glmnet1,
  model_glmnet4,
  model_glmnet2,
  model_glmnet3,
  model_glmnet5,
  model_glmnet6,
  model_rf,
  model_prophet_boost,
  model_svm,
  model_nnetar5
)%>% 
  update_modeltime_description(rep(1:10), c("Elastic Net (λ = 0,1, α = 0,1)",
                                            "Elastic Net (λ = 0,3, α = 0,1)",
                                            "Elastic Net (λ = 0,1, α = 0,3)",
                                            "Elastic Net (λ = 0,1, α = 0,5)",
                                            "Elastic Net (λ = 0,3, α = 0,3)",
                                            "Elastic Net (λ = 0,3, α = 0,5)",
                                            "Floresta Aleatória",
                                            "Prophet Boost (va. exógenas)",
                                            "SVM",
                                            "Perceptron (5 camadas)"))

# Test Set Visualization ---- Zoom dos 10 melhores modelos
best_models_tbl %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = sst_4n23w,
    conf_interval = 0
  ) %>%
  plot_modeltime_forecast(.interactive = FALSE,
                          .title = NULL,
                          .color_lab = "Legenda") +
  scale_x_date(limits = as.Date(c("2018-03-20","2019-03-27"), format = "%Y-%m-%d")) +
  scale_y_continuous(limits = c(26,30)) 


# Combinação de Modelos ---------------------------------------------------

ensemble_fit_avg = best_models_tbl %>%
  ensemble_average(type = "mean")

ensemble_fit_avg

ensemble_fit_median = best_models_tbl %>%
  ensemble_average(type = "median")

ensemble_fit_median

# w = seq(from = 20, to = 2, length = 10)
w = 1/meas_sorted$mae[1:10]
ensemble_fit_weighted_avg = best_models_tbl %>%
  ensemble_weighted(loadings = w)

ensemble_fit_weighted_avg

# Tabela dos modelos combinados
ens_calib_tbl = modeltime_table(
  ensemble_fit_avg,
  ensemble_fit_median,
  ensemble_fit_weighted_avg
) %>% modeltime_calibrate(testing(splits))%>% 
  update_modeltime_description(rep(1:3),c("Hibridização por Média",
                                          "Hibridização por Mediana",
                                          "Hibridização por Pesos"))

# Accuracy ----                                               
meas_ens = ens_calib_tbl %>% modeltime_accuracy(metric_set = metric_set(mae,
                                                             rmse))
meas_ens

meas_ens_sorted = meas_ens %>% 
  arrange(mae)
meas_ens_sorted

# PLot dos modelos combinados
# Test Set Visualization ----
ens_calib_tbl %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = sst_4n23w,
    conf_interval = 0
  ) %>%
  plot_modeltime_forecast(.interactive = FALSE,
                          .title = NULL,
                          .color_lab = "Legenda")

# Zoom do plot dos modelos combinados 
# Test Set Visualization ----
ens_calib_tbl %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = sst_4n23w,
    conf_interval = 0
  ) %>%
  plot_modeltime_forecast(.interactive = FALSE,
                          .title = NULL,
                          .color_lab = "Legenda")+
  scale_x_date(limits = as.Date(c("2018-03-20","2019-03-27"), format = "%Y-%m-%d")) +
  scale_y_continuous(limits = c(26,30))
