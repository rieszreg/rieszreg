simulate <- function(n, seed = 0L) {
  set.seed(seed)
  x <- rnorm(n)
  pi <- 1 / (1 + exp(-(0.5 * x)))
  a <- as.numeric(rbinom(n, 1, pi))
  data.frame(a = a, x = x)
}


.fit_net <- function(df, epochs = 10L, ticks = c(1L, 5L, 10L), ...) {
  rn <- RieszNet$new(
    estimand = ATE("a", "x"),
    hidden_sizes = c(8L, 8L),
    epochs = as.integer(epochs),
    learning_rate = 1e-2,
    snapshot_epochs = ticks,
    random_state = 0L,
    ...
  )
  # Drive python fit directly: full-batch is achieved by the default Python
  # batch_size=64 + small n; but RieszNet's R wrapper uses Python defaults
  # (batch_size=64). For a 50-row fixture that's full-batch already.
  rn$py$fit(rieszreg::df_to_py(df))
  rn
}


test_that("predict_path returns a matrix with epoch-labelled columns", {
  df <- simulate(50L, seed = 1L)
  rn <- .fit_net(df, epochs = 10L, ticks = c(1L, 5L, 10L))
  m <- rn$predict_path(df)
  expect_true(is.matrix(m))
  expect_equal(dim(m), c(nrow(df), 3L))
  expect_equal(colnames(m), c("epoch=1", "epoch=5", "epoch=10"))
})


test_that("predict_path final column matches predict on the same fit", {
  df <- simulate(50L, seed = 2L)
  rn <- .fit_net(df, epochs = 10L, ticks = c(1L, 5L, 10L))
  m <- rn$predict_path(df)
  py_alpha <- as.numeric(reticulate::py_to_r(rn$py$predict(rieszreg::df_to_py(df))))
  expect_equal(as.numeric(m[, ncol(m)]), py_alpha, tolerance = 1e-6)
})


test_that("predict_path output equals Python predict_path bit-for-bit", {
  df <- simulate(50L, seed = 3L)
  rn <- .fit_net(df, epochs = 10L, ticks = c(1L, 5L, 10L))
  r_path <- rn$predict_path(df)
  py_path <- as.matrix(rn$py$predict_path(rieszreg::df_to_py(df)))
  expect_equal(unname(r_path), py_path, tolerance = 0)
})


test_that("predict_path round-trips through save/load", {
  df <- simulate(50L, seed = 4L)
  rn <- .fit_net(df, epochs = 10L, ticks = c(1L, 5L, 10L))
  pre <- rn$predict_path(df)

  td <- tempfile()
  rn$py$save(td)
  loaded <- load_riesz_net(td)
  post <- loaded$predict_path(df)
  expect_equal(unname(pre), unname(post), tolerance = 1e-6)
  unlink(td, recursive = TRUE)
})
