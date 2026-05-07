test_that("R-side and Python-side predictions match on a small ATE problem", {
  skip_if_not(reticulate::py_module_available("krrr"))

  set.seed(0)
  n <- 200
  x <- runif(n, 0, 1)
  pi <- 1 / (1 + exp(-(-0.02 * x - x^2 + 4 * log(x + 0.3) + 1.5)))
  a <- as.numeric(rbinom(n, 1, pi))
  df <- data.frame(a = a, x = x)

  krr <- KernelRieszRegressor$new(
    estimand = ATE("a", "x"),
    kernel = Gaussian(length_scale = 0.5),
    lambda_grid = 10^seq(-3, 0, length.out = 6),
    solver = "direct",
    validation_fraction = 0.2,
    random_state = 0L
  )
  krr$fit(df)
  alpha_R <- krr$predict(df)

  # Python side directly
  krrr_py <- reticulate::import("krrr", convert = FALSE)
  pd <- reticulate::import("pandas", convert = FALSE)
  py_df <- pd$DataFrame(reticulate::r_to_py(list(a = a, x = x)))

  py_krr <- krrr_py$KernelRieszRegressor(
    estimand = krrr_py$ATE(treatment = "a", covariates = list("x")),
    kernel = krrr_py$Gaussian(length_scale = 0.5),
    lambda_grid = as.list(10^seq(-3, 0, length.out = 6)),
    solver = "direct",
    validation_fraction = 0.2,
    random_state = 0L
  )
  py_krr$fit(py_df)
  alpha_py <- as.numeric(reticulate::py_to_r(py_krr$predict(py_df)))

  expect_equal(alpha_R, alpha_py, tolerance = 1e-12)
})
