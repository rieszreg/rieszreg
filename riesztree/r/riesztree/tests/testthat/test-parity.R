test_that("R wrapper produces identical predictions to Python", {
  skip_on_cran()
  skip_if_not_installed("reticulate")

  venv <- normalizePath(file.path(getwd(), "../../../../../.venv"), mustWork = FALSE)
  if (!dir.exists(venv)) {
    skip("Shared rieszreg .venv not present at ../../../.venv")
  }
  reticulate::use_virtualenv(venv, required = TRUE)
  use_python_riesztree()

  set.seed(0)
  n  <- 600
  x  <- rnorm(n)
  pi <- 1 / (1 + exp(-(0.5 * x)))
  a  <- as.numeric(rbinom(n, 1, pi))
  df <- data.frame(a = a, x = x)

  rt <- RieszTreeRegressor$new(
    estimand = ATE(treatment = "a", covariates = "x"),
    max_depth = 4L,
    random_state = 0L
  )
  rt$fit(df)
  alpha_r <- as.numeric(rt$predict(df))

  # Same fit straight from Python.
  py <- reticulate::import("riesztree", convert = TRUE)
  pd <- reticulate::import("pandas", convert = FALSE)
  py_df <- pd$DataFrame(reticulate::r_to_py(list(a = a, x = x)))
  py_est <- py$RieszTreeRegressor(
    estimand = py$ATE(treatment = "a", covariates = list("x")),
    max_depth = 4L,
    random_state = 0L
  )
  py_est$fit(py_df)
  alpha_py <- as.numeric(reticulate::py_to_r(py_est$predict(py_df)))

  expect_equal(alpha_r, alpha_py, tolerance = 1e-12)
})
