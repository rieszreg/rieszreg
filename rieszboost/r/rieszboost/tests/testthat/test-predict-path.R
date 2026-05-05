simulate <- function(n, seed = 0L) {
  set.seed(seed)
  x <- runif(n)
  pi_x <- plogis(-0.02 * x - x^2 + 4 * log(x + 0.3) + 1.5)
  a <- rbinom(n, 1, pi_x)
  data.frame(a = as.numeric(a), x = x)
}


test_that("predict_path returns a matrix with grid-labelled columns", {
  df <- simulate(300L, seed = 1L)
  booster <- RieszBooster$new(estimand = ATE("a", "x"),
                              n_estimators = 40L, learning_rate = 0.05,
                              max_depth = 3L, random_state = 0L)
  booster$fit(df)
  m <- booster$predict_path(df, n_estimators_grid = c(5L, 17L, 40L))
  expect_true(is.matrix(m))
  expect_equal(dim(m), c(nrow(df), 3L))
  expect_equal(colnames(m), c("trees=5", "trees=17", "trees=40"))
})


test_that("predict_path final column matches predict on the same fit", {
  df <- simulate(300L, seed = 2L)
  booster <- RieszBooster$new(estimand = ATE("a", "x"),
                              n_estimators = 30L, learning_rate = 0.05,
                              max_depth = 3L, random_state = 0L)
  booster$fit(df)
  m <- booster$predict_path(df, n_estimators_grid = c(30L))
  expect_equal(as.numeric(m[, 1]), booster$predict(df), tolerance = 0)
})


test_that("predict_path columns equal independent shorter fits (bit-equal)", {
  df <- simulate(300L, seed = 3L)
  big <- RieszBooster$new(estimand = ATE("a", "x"),
                          n_estimators = 40L, learning_rate = 0.05,
                          max_depth = 3L, random_state = 0L)
  big$fit(df)
  grid <- c(5L, 17L, 40L)
  m <- big$predict_path(df, n_estimators_grid = grid)
  for (j in seq_along(grid)) {
    small <- RieszBooster$new(estimand = ATE("a", "x"),
                              n_estimators = grid[j], learning_rate = 0.05,
                              max_depth = 3L, random_state = 0L)
    small$fit(df)
    expect_equal(as.numeric(m[, j]), small$predict(df), tolerance = 0)
  }
})


test_that("predict_path output equals Python predict_path bit-for-bit", {
  df <- simulate(300L, seed = 4L)
  booster <- RieszBooster$new(estimand = ATE("a", "x"),
                              n_estimators = 40L, learning_rate = 0.05,
                              max_depth = 3L, random_state = 0L)
  booster$fit(df)
  grid <- c(5L, 17L, 40L)
  r_path <- booster$predict_path(df, n_estimators_grid = grid)

  py_pd <- reticulate::import("pandas", convert = TRUE)
  py_df <- py_pd$DataFrame(list(a = df$a, x = df$x))
  py_path <- as.matrix(booster$py$predict_path(py_df, as.integer(grid)))

  expect_equal(unname(r_path), py_path, tolerance = 0)
})


test_that("predict_path round-trips through save/load", {
  df <- simulate(300L, seed = 5L)
  booster <- RieszBooster$new(estimand = ATE("a", "x"),
                              n_estimators = 40L, learning_rate = 0.05,
                              max_depth = 3L, random_state = 0L)
  booster$fit(df)
  grid <- c(10L, 25L, 40L)
  pre <- booster$predict_path(df, n_estimators_grid = grid)

  td <- tempfile()
  booster$save(td)
  loaded <- load_riesz_booster(td)
  post <- loaded$predict_path(df, n_estimators_grid = grid)
  expect_equal(pre, post, tolerance = 0)
  unlink(td, recursive = TRUE)
})
