test_that("max_features = 1 means all features, as 1.0 does in Python", {
  skip_if_not_installed("reticulate")
  est <- ATE(treatment = "a", covariates = c("x", "g"))
  py_max_features <- function(v) {
    reticulate::py_to_r(RieszTreeRegressor$new(estimand = est, max_features = v)$py$max_features)
  }
  expect_true(is.double(py_max_features(1)))
  expect_identical(py_max_features(1L), 1L)
  expect_identical(py_max_features(2), 2L)
  expect_true(is.double(py_max_features(0.5)))
})

test_that("categorical_features rejects positions below 1", {
  skip_if_not_installed("reticulate")
  est <- ATE(treatment = "a", covariates = c("x", "g"))
  expect_error(RieszTreeRegressor$new(estimand = est, categorical_features = 0L), "1-based")
})
