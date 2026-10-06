test_that("Tensor() takes 1-based positions and rejects 0", {
  expect_error(Tensor(Gaussian(), 0L, Gaussian(), 2L), "1-based")
  t <- Tensor(Gaussian(), 1L, Gaussian(), 2:3)
  expect_identical(unlist(reticulate::py_to_r(t$cols_a)), 0L)
  expect_identical(unlist(reticulate::py_to_r(t$cols_b)), c(1L, 2L))
})
