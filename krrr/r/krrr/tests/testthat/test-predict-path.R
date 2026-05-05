simulate <- function(n, seed = 0L) {
  set.seed(seed)
  x <- runif(n, 0, 1)
  pi <- 1 / (1 + exp(-(-0.02 * x - x^2 + 4 * log(x + 0.3) + 1.5)))
  a <- as.numeric(rbinom(n, 1, pi))
  data.frame(a = a, x = x)
}


# The shared R6 base passes `X=` but krrr's Python fit takes `Z=`; calling
# `$fit(df)` therefore raises pre-existing on main. Until the rieszreg ↔
# krrr X/Z naming is reconciled, drive Python `fit` directly.
.fit_krr <- function(grid, df, ...) {
  krr <- KernelRieszRegressor$new(
    estimand = ATE("a", "x"),
    kernel = Gaussian(length_scale = 0.5),
    lambda_grid = grid,
    solver = "direct",
    validation_fraction = 0.2,
    random_state = 0L,
    ...
  )
  krr$py$fit(rieszreg::df_to_py(df))
  krr
}


test_that("predict_path returns a matrix with lambda-labelled columns", {
  df <- simulate(200L, seed = 1L)
  grid <- 10^seq(-3, 0, length.out = 4)
  krr <- .fit_krr(grid, df)
  m <- krr$predict_path(df)
  expect_true(is.matrix(m))
  expect_equal(dim(m), c(nrow(df), length(grid)))
  expect_equal(length(colnames(m)), length(grid))
  expect_match(colnames(m)[1], "^lambda=")
})


test_that("predict_path columns equal independent single-lambda fits", {
  df <- simulate(200L, seed = 2L)
  grid <- 10^seq(-3, 0, length.out = 4)
  full <- .fit_krr(grid, df)
  path <- full$predict_path(df)
  for (j in seq_along(grid)) {
    single <- .fit_krr(grid[j], df)
    expect_equal(as.numeric(path[, j]),
                 as.numeric(reticulate::py_to_r(single$py$predict(rieszreg::df_to_py(df)))),
                 tolerance = 1e-10)
  }
})


test_that("predict_path output equals Python predict_path bit-for-bit", {
  df <- simulate(200L, seed = 3L)
  grid <- 10^seq(-3, 0, length.out = 4)
  krr <- .fit_krr(grid, df)
  r_path <- krr$predict_path(df)
  py_path <- as.matrix(krr$py$predict_path(rieszreg::df_to_py(df)))
  expect_equal(unname(r_path), py_path, tolerance = 0)
})


test_that("predict_path round-trips through save/load", {
  df <- simulate(200L, seed = 4L)
  grid <- 10^seq(-3, 0, length.out = 4)
  krr <- .fit_krr(grid, df)
  pre <- krr$predict_path(df)

  td <- tempfile()
  krr$py$save(td)
  loaded <- load_kernel_riesz_regressor(td)
  post <- loaded$predict_path(df)
  expect_equal(unname(pre), unname(post), tolerance = 0)
  unlink(td, recursive = TRUE)
})
