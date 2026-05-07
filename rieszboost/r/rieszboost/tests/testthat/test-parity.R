simulate <- function(n, seed = 0L) {
  set.seed(seed)
  x <- runif(n)
  pi_x <- plogis(-0.02 * x - x^2 + 4 * log(x + 0.3) + 1.5)
  a <- rbinom(n, 1, pi_x)
  list(
    df = data.frame(a = as.numeric(a), x = x),
    pi = pi_x,
    alpha_true = a / pi_x - (1 - a) / (1 - pi_x)
  )
}


test_that("RieszBooster fits ATE on a data.frame", {
  s <- simulate(2000L, seed = 0L)
  n_tr <- 1600L
  booster <- RieszBooster$new(
    estimand = ATE("a", "x"),
    n_estimators = 1000L,
    early_stopping_rounds = 20L,
    learning_rate = 0.05,
    max_depth = 3L,
    reg_lambda = 10
  )
  booster$fit(s$df[1:n_tr, ], eval_set = s$df[(n_tr + 1):2000, ])
  alpha_hat <- booster$predict(s$df)
  expect_length(alpha_hat, 2000L)
  expect_true(cor(alpha_hat, s$alpha_true) > 0.85)
})


test_that("ATT (partial-parameter) factory traces and fits", {
  s <- simulate(2000L, seed = 1L)
  booster <- RieszBooster$new(
    estimand = ATT("a", "x"),
    n_estimators = 1000L,
    early_stopping_rounds = 20L,
    validation_fraction = 0.2,
    learning_rate = 0.05,
    max_depth = 3L,
    reg_lambda = 10
  )
  booster$fit(s$df)
  alpha_hat <- booster$predict(s$df)
  alpha_true <- s$df$a - (1 - s$df$a) * s$pi / (1 - s$pi)
  expect_true(cor(alpha_hat, alpha_true) > 0.85)
})


test_that("score returns negative riesz_loss", {
  s <- simulate(500L, seed = 2L)
  booster <- RieszBooster$new(estimand = ATE("a", "x"),
                              n_estimators = 30L, learning_rate = 0.1,
                              max_depth = 3L)
  booster$fit(s$df)
  expect_equal(booster$score(s$df), -booster$riesz_loss(s$df), tolerance = 1e-9)
})


test_that("diagnose returns a summary", {
  s <- simulate(500L, seed = 3L)
  booster <- RieszBooster$new(estimand = ATE("a", "x"),
                              n_estimators = 30L, learning_rate = 0.1,
                              max_depth = 3L)
  booster$fit(s$df)
  d <- booster$diagnose(s$df)
  expect_equal(d$n, 500L)
  expect_false(is.null(d$riesz_loss))
})


test_that("save / load round-trips a RieszBooster from R", {
  s <- simulate(400L, seed = 8L)
  booster <- RieszBooster$new(estimand = ATE("a", "x"),
                              n_estimators = 30L, learning_rate = 0.1,
                              max_depth = 3L)
  booster$fit(s$df)
  pre <- booster$predict(s$df)

  td <- tempfile()
  booster$save(td)
  loaded <- load_riesz_booster(td)
  post <- loaded$predict(s$df)
  expect_equal(pre, post, tolerance = 1e-12)
  unlink(td, recursive = TRUE)
})


test_that("Python-saved RieszBooster loads in R with bitwise-identical predictions", {
  s <- simulate(400L, seed = 9L)
  booster <- RieszBooster$new(estimand = ATE("a", "x"),
                              n_estimators = 30L, learning_rate = 0.1,
                              max_depth = 3L, random_state = 0L)
  booster$fit(s$df)

  # Save from R
  td <- tempfile()
  booster$save(td)

  # Load from Python directly via reticulate
  py_mod <- reticulate::import("rieszboost", convert = TRUE)
  py_loaded <- py_mod$RieszBooster$load(td)
  py_pd <- reticulate::import("pandas", convert = TRUE)
  py_df <- py_pd$DataFrame(list(a = s$df$a, x = s$df$x))
  py_preds <- as.numeric(py_loaded$predict(py_df))

  r_preds <- booster$predict(s$df)
  expect_equal(r_preds, py_preds, tolerance = 1e-12)
  unlink(td, recursive = TRUE)
})


test_that("R and Python predictions are bitwise-identical on the same data", {
  s <- simulate(400L, seed = 4L)
  booster <- RieszBooster$new(estimand = ATE("a", "x"),
                              n_estimators = 50L, learning_rate = 0.1,
                              max_depth = 3L, random_state = 0L)
  booster$fit(s$df)
  r_preds <- booster$predict(s$df)

  # Drive the Python booster directly via the same DataFrame
  py_pd <- reticulate::import("pandas", convert = TRUE)
  py_df <- py_pd$DataFrame(list(a = s$df$a, x = s$df$x))
  py_preds <- as.numeric(booster$py$predict(py_df))
  expect_equal(r_preds, py_preds, tolerance = 1e-10)
})
