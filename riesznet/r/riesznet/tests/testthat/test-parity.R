test_that("R-side and Python-side predictions match on a small TSM problem", {
  skip_if_not(reticulate::py_module_available("riesznet"))

  set.seed(0)
  n <- 200
  x <- runif(n, 0, 1)
  pi <- 1 / (1 + exp(-(0.5 * x - 0.3)))
  a <- as.numeric(rbinom(n, 1, pi))
  df <- data.frame(a = a, x = x)

  # R side: simple MLP on TSM.
  rn <- RieszNet$new(
    estimand = TSM(level = 1L, treatment = "a", covariates = "x"),
    hidden_sizes = c(8L, 8L),
    epochs = 30L,
    learning_rate = 5e-3,
    random_state = 0L
  )
  rn$fit(df)
  alpha_R <- rn$predict(df)

  # Python side directly.
  rn_py <- reticulate::import("riesznet", convert = FALSE)
  pd <- reticulate::import("pandas", convert = FALSE)
  py_df <- pd$DataFrame(reticulate::r_to_py(list(a = a, x = x)))

  py_rn <- rn_py$RieszNet(
    estimand = rn_py$TSM(level = 1L, treatment = "a", covariates = list("x")),
    hidden_sizes = reticulate::tuple(8L, 8L),
    epochs = 30L,
    learning_rate = 5e-3,
    random_state = 0L
  )
  py_rn$fit(py_df)
  alpha_py <- as.numeric(reticulate::py_to_r(py_rn$predict(py_df)))

  expect_equal(alpha_R, alpha_py, tolerance = 1e-6)
})
