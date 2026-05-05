#' krrr: R wrapper for the Python krrr library
#'
#' Mirrors the Python sklearn-style API. Configure once with
#' `use_python_krrr()`, then construct a [KernelRieszRegressor] and call
#' `$fit(df)` / `$predict(df)`.
#'
#' Estimand and loss factories live in the shared `rieszreg` R package and
#' are re-exported from here for convenience.
#'
#' @keywords internal
"_PACKAGE"


.kr <- new.env(parent = emptyenv())


#' Configure the Python interpreter that holds the krrr module.
#'
#' Call this once per session before any other krrr function. Forwards to
#' `reticulate::use_python` / `reticulate::use_virtualenv` as appropriate.
#'
#' @param python Path to the Python interpreter or virtualenv directory.
#' @param required Whether reticulate should fail if the Python is unavailable.
#' @export
use_python_krrr <- function(python = NULL, required = TRUE) {
  if (!is.null(python)) {
    if (dir.exists(python)) {
      reticulate::use_virtualenv(python, required = required)
    } else {
      reticulate::use_python(python, required = required)
    }
  }
  .kr$mod <- reticulate::import("krrr", convert = FALSE)
  invisible(.kr$mod)
}


.module <- function() {
  if (is.null(.kr$mod)) {
    .kr$mod <- reticulate::import("krrr", convert = FALSE)
  }
  .kr$mod
}


# ---- Kernels (krrr-specific) -------------------------------------------

#' Gaussian / RBF kernel: k(x, y) = exp(-||x - y||^2 / (2 sigma^2)).
#' @param length_scale numeric, or one of "median", "scott", "silverman".
#' @export
Gaussian <- function(length_scale = "median") {
  .module()$Gaussian(length_scale = length_scale)
}

#' Matern kernel of half-integer smoothness nu in {0.5, 1.5, 2.5}.
#' @inheritParams Gaussian
#' @param nu One of 0.5, 1.5, 2.5.
#' @export
Matern <- function(nu = 2.5, length_scale = "median") {
  .module()$Matern(nu = nu, length_scale = length_scale)
}

#' Linear kernel: k(x, y) = c + x . y.
#' @param bias Additive offset c.
#' @export
Linear <- function(bias = 0.0) {
  .module()$Linear(bias = bias)
}

#' Polynomial kernel: k(x, y) = (gamma * x . y + coef0)^degree.
#' @param degree Polynomial degree (integer).
#' @param gamma,coef0 Polynomial-kernel hyperparameters.
#' @export
Polynomial <- function(degree = 3L, gamma = 1.0, coef0 = 1.0) {
  .module()$Polynomial(degree = as.integer(degree), gamma = gamma, coef0 = coef0)
}

#' Tensor-product kernel over disjoint feature subsets.
#' @param a Kernel applied on `cols_a`.
#' @param cols_a Integer vector of column indices.
#' @param b Kernel applied on `cols_b`.
#' @param cols_b Integer vector of column indices.
#' @export
Tensor <- function(a, cols_a, b, cols_b) {
  .module()$Tensor(a = a,
                   cols_a = as.list(as.integer(cols_a)),
                   b = b,
                   cols_b = as.list(as.integer(cols_b)))
}


# ---- Main estimator (R6 subclass) -----------------------------------------

#' KernelRieszRegressor — kernel ridge Riesz regression.
#'
#' Subclass of [rieszreg::RieszEstimatorR6] that defaults the backend to a
#' `KernelRidgeBackend` and surfaces kernel-specific hyperparameters (`kernel`,
#' `lambda_grid`, `solver`, `n_landmarks`, `n_features`, `cg_tol`, `cg_max_iter`)
#' on the constructor.
#'
#' @export
KernelRieszRegressor <- R6::R6Class(
  "KernelRieszRegressor",
  inherit = rieszreg::RieszEstimatorR6,
  public = list(
    initialize = function(estimand,
                          kernel = NULL, lambda_grid = NULL,
                          solver = "auto", loss = NULL,
                          n_landmarks = NULL, n_features = 1024L,
                          cg_tol = 1e-6, cg_max_iter = 200L,
                          init = NULL, validation_fraction = 0.2,
                          keep_path = TRUE,
                          random_state = 0L) {
      args <- list(
        estimand = estimand,
        solver = solver,
        n_features = as.integer(n_features),
        cg_tol = cg_tol,
        cg_max_iter = as.integer(cg_max_iter),
        validation_fraction = validation_fraction,
        keep_path = isTRUE(keep_path),
        random_state = as.integer(random_state)
      )
      if (!is.null(kernel)) args$kernel <- kernel
      if (!is.null(lambda_grid)) args$lambda_grid <- as.list(as.numeric(lambda_grid))
      if (!is.null(loss)) args$loss <- loss
      if (!is.null(n_landmarks)) args$n_landmarks <- as.integer(n_landmarks)
      if (!is.null(init)) args$init <- init
      py_object <- do.call(.module()$KernelRieszRegressor, args)
      super$initialize(py_object = py_object, estimand = estimand)
    },

    #' Predict α̂ at every λ in the (optionally subset) lambda_grid.
    #'
    #' Returns an n_rows × n_lambdas numeric matrix. Column ``j`` is the
    #' prediction at ``lambdas[j]`` (or ``self$py$lambda_grid[j]`` when
    #' `lambdas` is `NULL`). Each column is bit-equal to a fresh fit at a
    #' singleton `lambda_grid` containing that λ.
    #'
    #' Requires `keep_path = TRUE` (the default).
    #' @param Z Feature data.frame.
    #' @param lambdas Numeric vector of λ values (a subset of the stored
    #'   `lambda_grid`); defaults to the full grid.
    predict_path = function(Z, lambdas = NULL) {
      if (is.null(lambdas)) {
        py_lam <- NULL
      } else {
        py_lam <- reticulate::r_to_py(as.list(as.numeric(lambdas)))
      }
      out <- self$py$predict_path(rieszreg::df_to_py(Z), py_lam)
      m <- as.matrix(reticulate::py_to_r(out))
      lam_used <- if (is.null(lambdas)) {
        as.numeric(reticulate::py_to_r(self$py$predictor_$lambda_grid))
      } else {
        as.numeric(lambdas)
      }
      colnames(m) <- paste0("lambda=", format(lam_used, scientific = TRUE))
      m
    }
  )
)


#' Load a fitted KernelRieszRegressor from a directory written by `$save()`.
#'
#' For built-in estimands, fully reconstructs the estimand from the metadata.
#' For custom estimands (Python-only), pass `estimand=` explicitly.
#' @param path Directory path.
#' @param estimand Optional user-supplied `Estimand` (required for custom m).
#' @export
load_kernel_riesz_regressor <- function(path, estimand = NULL) {
  args <- list(path = path)
  if (!is.null(estimand)) args$estimand <- estimand
  py_obj <- do.call(.module()$KernelRieszRegressor$load, args)
  rb <- KernelRieszRegressor$new(estimand = py_obj$estimand)
  rb$py <- py_obj
  rb$estimand <- py_obj$estimand
  rb
}


# Estimand and loss factories are re-exported from rieszreg via NAMESPACE
# (importFrom + export), so `library(krrr); ATE(...)` keeps working.
