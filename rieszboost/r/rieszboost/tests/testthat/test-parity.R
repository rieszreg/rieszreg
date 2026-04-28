simulate <- function(n, seed = 0L) {
  set.seed(seed)
  x <- runif(n)
  pi_x <- plogis(-0.02 * x - x^2 + 4 * log(x + 0.3) + 1.5)
  a <- rbinom(n, 1, pi_x)
  list(
    df = data.frame(a = a, x = x),
    pi = pi_x,
    alpha_true = a / pi_x - (1 - a) / (1 - pi_x)
  )
}


test_that("ATE fit produces reasonable Riesz representer", {
  s <- simulate(2000L, seed = 0L)
  n_tr <- 1600L
  fit <- fit_riesz(
    data = s$df[1:n_tr, ],
    m = ATE("a", "x"),
    feature_keys = c("a", "x"),
    valid_data = s$df[(n_tr + 1):2000, ],
    num_boost_round = 1000L,
    early_stopping_rounds = 20L,
    learning_rate = 0.05,
    max_depth = 3L,
    reg_lambda = 10
  )
  expect_s3_class(fit, "RieszBooster")
  alpha_hat <- predict(fit, s$df)
  expect_length(alpha_hat, 2000L)
  expect_true(cor(alpha_hat, s$alpha_true) > 0.85)
})


test_that("ATT factory traces and fits", {
  s <- simulate(2000L, seed = 1L)
  p_treated <- mean(s$df$a)
  n_tr <- 1600L
  fit <- fit_riesz(
    data = s$df[1:n_tr, ],
    m = ATT(p_treated = p_treated, treatment = "a", covariates = "x"),
    feature_keys = c("a", "x"),
    valid_data = s$df[(n_tr + 1):2000, ],
    num_boost_round = 1000L,
    early_stopping_rounds = 20L,
    learning_rate = 0.05,
    max_depth = 3L,
    reg_lambda = 10
  )
  alpha_hat <- predict(fit, s$df)
  alpha_true <- s$df$a / p_treated -
    (1 - s$df$a) * s$pi / ((1 - s$pi) * p_treated)
  expect_true(cor(alpha_hat, alpha_true) > 0.85)
})


test_that("crossfit returns OOF alpha for every row", {
  s <- simulate(800L, seed = 2L)
  res <- crossfit(
    data = s$df,
    m = ATE("a", "x"),
    feature_keys = c("a", "x"),
    n_folds = 4L,
    seed = 0L,
    early_stopping_inner_split = 0.2,
    num_boost_round = 500L,
    early_stopping_rounds = 10L,
    learning_rate = 0.05,
    max_depth = 3L,
    reg_lambda = 10
  )
  expect_length(res$alpha_hat, 800L)
  expect_length(res$boosters, 4L)
  expect_setequal(unique(res$fold_assignment), 0:3)
})


test_that("diagnose_alpha works with both alpha_hat and booster paths", {
  s <- simulate(500L, seed = 3L)
  fit <- fit_riesz(
    data = s$df, m = ATE("a", "x"), feature_keys = c("a", "x"),
    num_boost_round = 30L, learning_rate = 0.1, max_depth = 3L
  )
  d1 <- diagnose_alpha(booster = fit, data = s$df, m = ATE("a", "x"))
  expect_s3_class(d1, "RieszbDiagnostics")
  expect_equal(d1$n, 500L)
  expect_false(is.null(d1$riesz_loss))

  d2 <- diagnose_alpha(alpha_hat = predict(fit, s$df))
  expect_equal(d2$n, 500L)
  expect_null(d2$riesz_loss)
})


test_that("gradient_only=TRUE works end-to-end", {
  s <- simulate(800L, seed = 5L)
  fit <- fit_riesz(
    data = s$df, m = ATE("a", "x"), feature_keys = c("a", "x"),
    num_boost_round = 200L, learning_rate = 0.05,
    max_depth = 4L, gradient_only = TRUE, seed = 0L
  )
  alpha_hat <- predict(fit, s$df)
  expect_length(alpha_hat, 800L)
  expect_true(cor(alpha_hat, s$alpha_true) > 0.5)
})


test_that("StochasticIntervention works from R via list-column payload", {
  set.seed(7)
  n <- 400L
  x <- runif(n, 0, 2)
  a <- rnorm(n, x^2 - 1, sqrt(2))
  shift_samples <- lapply(seq_len(n), function(i) rnorm(20, a[i] + 1, 0.5))
  df <- data.frame(a = a, x = x)
  df$shift_samples <- shift_samples   # list-column

  fit <- fit_riesz(
    data = df,
    m = StochasticIntervention(samples_key = "shift_samples",
                               treatment = "a", covariates = "x"),
    feature_keys = c("a", "x"),
    num_boost_round = 30L,
    learning_rate = 0.05,
    max_depth = 3L
  )
  alpha_hat <- predict(fit, df)
  expect_length(alpha_hat, n)
  expect_true(all(is.finite(alpha_hat)))
})


test_that("R and Python produce identical predictions on the same data", {
  s <- simulate(400L, seed = 4L)
  fit <- fit_riesz(
    data = s$df, m = ATE("a", "x"), feature_keys = c("a", "x"),
    num_boost_round = 50L, learning_rate = 0.1, max_depth = 3L, seed = 0L
  )
  r_preds <- predict(fit, s$df)

  py <- reticulate::import("rieszboost", convert = TRUE)
  rows_py <- lapply(seq_len(nrow(s$df)), function(i) {
    list(a = as.integer(s$df$a[i]), x = as.numeric(s$df$x[i]))
  })
  py_preds <- as.numeric(fit$py$predict(reticulate::r_to_py(rows_py)))
  expect_equal(r_preds, py_preds, tolerance = 1e-10)
})
