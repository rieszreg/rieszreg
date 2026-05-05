#' rieszboost: R wrapper for the Python rieszboost library
#'
#' Mirrors the Python sklearn-style API. Configure once with
#' `use_python_rieszboost()`, then construct a [RieszBooster] and call
#' `$fit(df)` / `$predict(df)`.
#'
#' Estimand and loss factories live in the shared `rieszreg` R package and
#' are re-exported from here for convenience.
#'
#' @keywords internal
"_PACKAGE"


.rb <- new.env(parent = emptyenv())


#' Configure the Python interpreter that holds the rieszboost module.
#'
#' Call this once per session before any other rieszboost function. Forwards
#' to `reticulate::use_python` / `reticulate::use_virtualenv` as appropriate.
#'
#' @param python Path to the Python interpreter or virtualenv directory.
#' @param required Whether reticulate should fail if the Python is unavailable.
#' @export
use_python_rieszboost <- function(python = NULL, required = TRUE) {
  if (!is.null(python)) {
    if (dir.exists(python)) {
      reticulate::use_virtualenv(python, required = required)
    } else {
      reticulate::use_python(python, required = required)
    }
  }
  .rb$mod <- reticulate::import("rieszboost", convert = FALSE)
  invisible(.rb$mod)
}


.module <- function() {
  if (is.null(.rb$mod)) {
    .rb$mod <- reticulate::import("rieszboost", convert = FALSE)
  }
  .rb$mod
}


# ---- Backends (rieszboost-specific) ----

#' Default backend: data augmentation + xgboost custom objective.
#' @param hessian_floor Lower bound on per-row Hessian (default 2.0).
#' @param gradient_only If TRUE, disable second-order Newton step (Friedman 2001 mode).
#' @export
XGBoostBackend <- function(n_estimators = 200L,
                           learning_rate = 0.05,
                           early_stopping_rounds = NULL,
                           validation_fraction = 0.0,
                           hessian_floor = 2.0,
                           gradient_only = FALSE) {
  args <- list(
    n_estimators = as.integer(n_estimators),
    learning_rate = learning_rate,
    validation_fraction = validation_fraction,
    hessian_floor = hessian_floor,
    gradient_only = gradient_only
  )
  if (!is.null(early_stopping_rounds)) {
    args$early_stopping_rounds <- as.integer(early_stopping_rounds)
  }
  do.call(.module()$XGBoostBackend, args)
}

#' Slow general backend: Friedman gradient boosting with arbitrary
#' sklearn-compatible base learner.
#' @param base_learner_factory A zero-arg R function (or Python callable)
#'   returning a fresh sklearn estimator each round.
#' @param n_estimators,learning_rate,early_stopping_rounds,validation_fraction
#'   Boost-loop knobs. Live on the backend (rieszreg DESIGN §A.2 — the
#'   agnostic-orchestrator rule).
#' @export
SklearnBackend <- function(base_learner_factory,
                           n_estimators = 200L,
                           learning_rate = 0.05,
                           early_stopping_rounds = NULL,
                           validation_fraction = 0.0) {
  args <- list(
    base_learner_factory = base_learner_factory,
    n_estimators = as.integer(n_estimators),
    learning_rate = learning_rate,
    validation_fraction = validation_fraction
  )
  if (!is.null(early_stopping_rounds)) {
    args$early_stopping_rounds <- as.integer(early_stopping_rounds)
  }
  do.call(.module()$SklearnBackend, args)
}


# ---- Main estimator (R6 subclass) ----

#' RieszBooster — gradient-boosted estimator for the Riesz representer.
#'
#' Subclass of [rieszreg::RieszEstimatorR6] that defaults the backend to
#' `XGBoostBackend()` and surfaces xgboost-specific hyperparameters
#' (`max_depth`, `reg_lambda`, `subsample`) on the constructor.
#'
#' @export
RieszBooster <- R6::R6Class(
  "RieszBooster",
  inherit = rieszreg::RieszEstimatorR6,
  public = list(
    #' @param estimand An `Estimand` returned by [rieszreg::ATE()] etc.
    #' @param backend Backend object; default `XGBoostBackend()`.
    #' @param loss Loss spec; default `SquaredLoss()`.
    #' @param n_estimators,learning_rate,max_depth,reg_lambda,subsample
    #'   Hyperparameters.
    #' @param early_stopping_rounds,validation_fraction Early-stopping config.
    #' @param init Initial alpha (NULL → loss default; "m1" → mean of m(z,1); float → that value).
    #' @param random_state Random seed.
    initialize = function(estimand,
                          backend = NULL, loss = NULL,
                          n_estimators = 200L, learning_rate = 0.05,
                          max_depth = 4L, reg_lambda = 1.0, subsample = 1.0,
                          early_stopping_rounds = NULL,
                          validation_fraction = 0.0,
                          init = NULL,
                          random_state = 0L) {
      args <- list(
        estimand = estimand,
        n_estimators = as.integer(n_estimators),
        learning_rate = learning_rate,
        max_depth = as.integer(max_depth),
        reg_lambda = reg_lambda,
        subsample = subsample,
        validation_fraction = validation_fraction,
        random_state = as.integer(random_state)
      )
      if (!is.null(backend)) args$backend <- backend
      if (!is.null(loss)) args$loss <- loss
      if (!is.null(early_stopping_rounds))
        args$early_stopping_rounds <- as.integer(early_stopping_rounds)
      if (!is.null(init)) args$init <- init
      py_object <- do.call(.module()$RieszBooster, args)
      super$initialize(py_object = py_object, estimand = estimand)
    },

    #' Predict α̂ at every tree count in `n_estimators_grid` from one fit.
    #'
    #' Returns an n_rows × length(n_estimators_grid) numeric matrix whose
    #' column j is the α̂ obtained by truncating the booster to
    #' `n_estimators_grid[j]` trees. Columns are labelled `"trees=<k>"`.
    #'
    #' Each grid entry must be in `[1, booster.num_boosted_rounds()]`.
    #' @param Z Predictor data.frame (treatment + covariates in
    #'   `feature_keys` order).
    #' @param n_estimators_grid Integer vector of tree counts.
    predict_path = function(Z, n_estimators_grid) {
      grid <- as.integer(n_estimators_grid)
      out <- self$py$predict_path(rieszreg::df_to_py(Z),
                                  reticulate::r_to_py(as.list(grid)))
      m <- as.matrix(reticulate::py_to_r(out))
      colnames(m) <- paste0("trees=", grid)
      m
    }
  )
)


#' Load a RieszBooster from a directory written by `RieszBooster$save()`.
#'
#' For built-in estimands, fully reconstructs the estimand from the metadata.
#' For custom estimands (Python-only), pass `estimand=` explicitly.
#' @param path Directory path.
#' @param estimand Optional user-supplied `Estimand` (required for custom m).
#' @export
load_riesz_booster <- function(path, estimand = NULL) {
  args <- list(path = path)
  if (!is.null(estimand)) args$estimand <- estimand
  py_obj <- do.call(.module()$RieszBooster$load, args)
  rb <- RieszBooster$new(estimand = py_obj$estimand,
                         n_estimators = 1L)  # dummy, replaced below
  rb$py <- py_obj
  rb$estimand <- py_obj$estimand
  rb
}


# Estimand and loss factories are re-exported from rieszreg via NAMESPACE
# (importFrom + export), so `library(rieszboost); ATE(...)` keeps working.
