#' rieszboost: R wrapper for the Python rieszboost library
#'
#' Mirrors the Python sklearn-style API. Configure once with
#' `use_python_rieszboost()`, then construct a [RieszBooster] and call
#' `$fit(df)` / `$predict(df)`.
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


.df_to_py <- function(data, estimand) {
  # Convert R data.frame to a pandas DataFrame; preserves list-columns
  # (e.g. shift_samples) by element-wise conversion.
  cols <- colnames(data)
  py_dict <- list()
  for (k in cols) {
    v <- data[[k]]
    if (is.list(v)) {
      # list-column: each row is a numeric vector (e.g. shift_samples)
      py_dict[[k]] <- lapply(v, function(x) reticulate::r_to_py(as.numeric(x)))
    } else {
      py_dict[[k]] <- as.numeric(v)
    }
  }
  pd <- reticulate::import("pandas", convert = FALSE)
  pd$DataFrame(reticulate::r_to_py(py_dict))
}


# ---- Estimand factories (return opaque Python Estimand instances) ----

#' Average treatment effect estimand: m(z, alpha) = alpha(1, x) - alpha(0, x).
#' @param treatment Name of the treatment column.
#' @param covariates Character vector of covariate column names.
#' @return A Python `Estimand` object, suitable to pass to `RieszBooster$new(estimand=...)`.
#' @export
ATE <- function(treatment = "a", covariates = "x") {
  .module()$ATE(treatment = treatment, covariates = as.list(covariates))
}


#' ATT *partial parameter* estimand: m(z, alpha) = a*(alpha(1,x) - alpha(0,x)).
#' Full ATT divides by P(A=1) and is not a Riesz functional — combine
#' alpha_partial with a delta-method EIF (Hubbard 2011) downstream.
#' @inheritParams ATE
#' @export
ATT <- function(treatment = "a", covariates = "x") {
  .module()$ATT(treatment = treatment, covariates = as.list(covariates))
}


#' Treatment-specific mean: m(z, alpha) = alpha(level, x).
#' @param level Fixed treatment value.
#' @inheritParams ATE
#' @export
TSM <- function(level, treatment = "a", covariates = "x") {
  .module()$TSM(level = level, treatment = treatment, covariates = as.list(covariates))
}


#' Additive shift effect: m(z, alpha) = alpha(a + delta, x) - alpha(a, x).
#' @param delta Shift magnitude.
#' @inheritParams ATE
#' @export
AdditiveShift <- function(delta, treatment = "a", covariates = "x") {
  .module()$AdditiveShift(delta = delta, treatment = treatment,
                          covariates = as.list(covariates))
}


#' LASE *partial parameter* estimand. Full LASE divides by P(A < threshold)
#' and is not a Riesz functional.
#' @param delta Shift magnitude.
#' @param threshold Cutoff; only rows with `a < threshold` get shifted.
#' @inheritParams ATE
#' @export
LocalShift <- function(delta, threshold, treatment = "a", covariates = "x") {
  .module()$LocalShift(delta = delta, threshold = threshold,
                       treatment = treatment, covariates = as.list(covariates))
}


#' Stochastic intervention via pre-computed Monte Carlo samples per row.
#' Pass a data.frame with a list-column under `samples_key` containing
#' numeric vectors of shift samples.
#' @inheritParams ATE
#' @param samples_key Column holding the per-row sample vectors.
#' @export
StochasticIntervention <- function(samples_key = "shift_samples",
                                   treatment = "a", covariates = "x") {
  .module()$StochasticIntervention(samples_key = samples_key,
                                   treatment = treatment,
                                   covariates = as.list(covariates))
}


# ---- Loss specs ----

#' Squared Riesz loss (default — the standard Lee-Schuler / Chernozhukov objective).
#' @export
SquaredLoss <- function() {
  .module()$SquaredLoss()
}

#' KL-Bregman loss (phi = t log t with exp link). Suitable for density-ratio
#' targets like TSM / IPSI; requires non-negative m-coefficients.
#' @export
KLLoss <- function(max_eta = 50.0) {
  .module()$KLLoss(max_eta = max_eta)
}


# ---- Backends ----

#' Default backend: data augmentation + xgboost custom objective.
#' @param hessian_floor Lower bound on per-row Hessian (default 2.0).
#' @param gradient_only If TRUE, disable second-order Newton step (Friedman 2001 mode).
#' @export
XGBoostBackend <- function(hessian_floor = 2.0, gradient_only = FALSE) {
  .module()$XGBoostBackend(hessian_floor = hessian_floor,
                           gradient_only = gradient_only)
}

#' Slow general backend: Friedman gradient boosting with arbitrary
#' sklearn-compatible base learner.
#' @param base_learner_factory A zero-arg R function (or Python callable)
#'   returning a fresh sklearn estimator each round.
#' @export
SklearnBackend <- function(base_learner_factory) {
  .module()$SklearnBackend(base_learner_factory = base_learner_factory)
}


# ---- Main estimator (R6) ----

#' RieszBooster — gradient-boosted estimator for the Riesz representer.
#'
#' R6 wrapper around the Python `rieszboost.RieszBooster`. Construct with the
#' estimand baked in; standard `$fit(df)`, `$predict(df)`, `$score(df)`.
#'
#' @export
RieszBooster <- R6::R6Class(
  "RieszBooster",
  public = list(
    py = NULL,
    estimand = NULL,

    #' @param estimand An `Estimand` returned by [ATE()], [ATT()], etc.
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
      self$py <- do.call(.module()$RieszBooster, args)
      self$estimand <- estimand
      invisible(self)
    },

    fit = function(data, eval_set = NULL) {
      X <- .df_to_py(data, self$estimand)
      args <- list(X = X)
      if (!is.null(eval_set)) {
        args$eval_set <- .df_to_py(eval_set, self$estimand)
      }
      do.call(self$py$fit, args)
      invisible(self)
    },

    predict = function(data) {
      preds <- self$py$predict(.df_to_py(data, self$estimand))
      as.numeric(reticulate::py_to_r(preds))
    },

    score = function(data) {
      reticulate::py_to_r(self$py$score(.df_to_py(data, self$estimand)))
    },

    riesz_loss = function(data) {
      reticulate::py_to_r(self$py$riesz_loss(.df_to_py(data, self$estimand)))
    },

    diagnose = function(data, ...) {
      d <- self$py$diagnose(.df_to_py(data, self$estimand), ...)
      list(
        n = reticulate::py_to_r(d$n),
        rms = reticulate::py_to_r(d$rms),
        mean = reticulate::py_to_r(d$mean),
        min = reticulate::py_to_r(d$min),
        max = reticulate::py_to_r(d$max),
        abs_quantiles = reticulate::py_to_r(d$abs_quantiles),
        n_extreme = reticulate::py_to_r(d$n_extreme),
        extreme_fraction = reticulate::py_to_r(d$extreme_fraction),
        extreme_threshold = reticulate::py_to_r(d$extreme_threshold),
        riesz_loss = reticulate::py_to_r(d$riesz_loss),
        warnings = as.character(reticulate::py_to_r(d$warnings)),
        summary = reticulate::py_to_r(d$summary())
      )
    },

    print = function(...) {
      cat("<RieszBooster>\n")
      cat("  estimand   :", reticulate::py_to_r(self$estimand$name), "\n")
      best_iter <- tryCatch(reticulate::py_to_r(self$py$best_iteration_),
                            error = function(e) NULL)
      if (!is.null(best_iter)) {
        cat("  best_iter  :", best_iter, "\n")
        cat("  best_score :", reticulate::py_to_r(self$py$best_score_), "\n")
      } else {
        cat("  status     : unfitted\n")
      }
      invisible(self)
    }
  )
)
