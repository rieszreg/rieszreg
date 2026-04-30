#' forestriesz: R wrapper for the Python forestriesz library
#'
#' Mirrors the Python sklearn-style API. Configure once with
#' `use_python_forestriesz()`, then construct a [ForestRieszRegressor] and call
#' `$fit(df)` / `$predict(df)`.
#'
#' Estimand and loss factories live in the shared `rieszreg` R package and are
#' re-exported from here for convenience.
#'
#' Scope note: the R wrapper exposes locally constant fits only. Locally
#' linear fits require Python-callable basis functions (`riesz_feature_fns`)
#' and are Python-only in this release. Difference-style estimands (ATE,
#' ATT, AdditiveShift, LocalShift) need a sieve, so call them from Python via
#' reticulate or wait for v2.
#'
#' @keywords internal
"_PACKAGE"


.fr <- new.env(parent = emptyenv())


#' Configure the Python interpreter that holds the forestriesz module.
#'
#' Call this once per session before any other forestriesz function. Forwards
#' to `reticulate::use_python` / `reticulate::use_virtualenv` as appropriate.
#'
#' @param python Path to the Python interpreter or virtualenv directory.
#' @param required Whether reticulate should fail if the Python is unavailable.
#' @export
use_python_forestriesz <- function(python = NULL, required = TRUE) {
  if (!is.null(python)) {
    if (dir.exists(python)) {
      reticulate::use_virtualenv(python, required = required)
    } else {
      reticulate::use_python(python, required = required)
    }
  }
  .fr$mod <- reticulate::import("forestriesz", convert = FALSE)
  invisible(.fr$mod)
}


.module <- function() {
  if (is.null(.fr$mod)) {
    .fr$mod <- reticulate::import("forestriesz", convert = FALSE)
  }
  .fr$mod
}


# ---- Main estimator (R6 subclass) -----------------------------------------

#' ForestRieszRegressor — random-forest Riesz regression.
#'
#' Subclass of [rieszreg::RieszEstimatorR6] that defaults the backend to a
#' GRF-based moment forest and surfaces forest hyperparameters
#' (`n_estimators`, `max_depth`, `min_samples_leaf`, `honest`, `inference`,
#' `l2`, ...) on the constructor.
#'
#' Locally constant only from R (single-level estimands like `TSM(level=1)`).
#' For ATE/ATT/sieve fits, call into Python via reticulate.
#'
#' @export
ForestRieszRegressor <- R6::R6Class(
  "ForestRieszRegressor",
  inherit = rieszreg::RieszEstimatorR6,
  public = list(
    initialize = function(estimand,
                          n_estimators = 100L,
                          max_depth = NULL,
                          min_samples_split = 10L,
                          min_samples_leaf = 5L,
                          max_features = "auto",
                          max_samples = 0.45,
                          min_balancedness_tol = 0.45,
                          honest = FALSE,
                          inference = FALSE,
                          subforest_size = 4L,
                          l2 = 0.01,
                          n_jobs = -1L,
                          loss = NULL,
                          init = NULL,
                          random_state = 0L) {
      args <- list(
        estimand = estimand,
        n_estimators = as.integer(n_estimators),
        min_samples_split = as.integer(min_samples_split),
        min_samples_leaf = as.integer(min_samples_leaf),
        max_features = max_features,
        max_samples = max_samples,
        min_balancedness_tol = min_balancedness_tol,
        honest = honest,
        inference = inference,
        subforest_size = as.integer(subforest_size),
        l2 = l2,
        n_jobs = as.integer(n_jobs),
        random_state = as.integer(random_state)
      )
      if (!is.null(max_depth)) args$max_depth <- as.integer(max_depth)
      if (!is.null(loss)) args$loss <- loss
      if (!is.null(init)) args$init <- init
      py_object <- do.call(.module()$ForestRieszRegressor, args)
      super$initialize(py_object = py_object, estimand = estimand)
    },

    #' Confidence interval for alpha(X) at confidence 1 - alpha.
    #' Requires `honest = TRUE` and `inference = TRUE` at fit. Locally
    #' constant only in v1.
    predict_interval = function(df, alpha = 0.05) {
      result <- self$py$predict_interval(df, alpha = alpha)
      list(lb = reticulate::py_to_r(result[[1]]),
           ub = reticulate::py_to_r(result[[2]]))
    }
  )
)


#' Load a fitted ForestRieszRegressor from a directory written by `$save()`.
#'
#' For built-in estimands, fully reconstructs the estimand from the metadata.
#' For custom estimands (Python-only), pass `estimand=` explicitly.
#' @param path Directory path.
#' @param estimand Optional user-supplied `Estimand` (required for custom m).
#' @export
load_forest_riesz_regressor <- function(path, estimand = NULL) {
  args <- list(path = path)
  if (!is.null(estimand)) args$estimand <- estimand
  py_obj <- do.call(.module()$ForestRieszRegressor$load, args)
  fr <- ForestRieszRegressor$new(estimand = py_obj$estimand)
  fr$py <- py_obj
  fr$estimand <- py_obj$estimand
  fr
}


# Estimand and loss factories are re-exported from rieszreg via NAMESPACE
# (importFrom + export), so `library(forestriesz); TSM(level=1)` works.
