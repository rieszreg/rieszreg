#' rieszboost: R wrapper for the Python rieszboost library
#'
#' All work is delegated to the Python package through reticulate. Set up the
#' Python interpreter once with `use_python_rieszboost()`, then use [fit_riesz()],
#' [crossfit()], [diagnose_alpha()], and the estimand factories [ATE()], [ATT()],
#' [TSM()], [AdditiveShift()].
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


.rows_from_df <- function(data, feature_keys) {
  feature_keys <- as.character(feature_keys)
  missing <- setdiff(feature_keys, colnames(data))
  if (length(missing) > 0) {
    stop("data is missing required columns: ", paste(missing, collapse = ", "))
  }
  cols <- colnames(data)
  rl <- vector("list", nrow(data))
  for (i in seq_len(nrow(data))) {
    row <- list()
    for (k in cols) {
      v <- data[[k]][i]
      # Unwrap list-columns: data[[k]][i] for a list-column returns a length-1
      # list whose element is the per-row vector. The Python side expects the
      # vector itself (e.g. for StochasticIntervention's shift_samples).
      if (is.list(v) && length(v) == 1L) v <- v[[1]]
      row[[k]] <- v
    }
    rl[[i]] <- reticulate::r_to_py(row)
  }
  reticulate::r_to_py(rl)
}


#' Average treatment effect estimand: m(z, alpha) = alpha(1, x) - alpha(0, x).
#' @param treatment Name of the treatment column.
#' @param covariates Character vector of covariate column names.
#' @return An opaque Python callable suitable to pass to `fit_riesz`.
#' @export
ATE <- function(treatment = "a", covariates = "x") {
  .module()$ATE(treatment = treatment, covariates = as.list(covariates))
}


#' Average treatment effect on the treated.
#' @param p_treated Marginal P(A=1); typically `mean(data$a)`.
#' @inheritParams ATE
#' @export
ATT <- function(p_treated, treatment = "a", covariates = "x") {
  .module()$ATT(
    p_treated = p_treated,
    treatment = treatment,
    covariates = as.list(covariates)
  )
}


#' Treatment-specific mean: m(z, alpha) = alpha(level, x).
#' @param level The fixed treatment value to evaluate alpha at.
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
  .module()$AdditiveShift(
    delta = delta,
    treatment = treatment,
    covariates = as.list(covariates)
  )
}


#' Stochastic intervention via pre-computed Monte Carlo samples.
#'
#' The functional is `theta = E[integral mu(a', X) g(a' | A, X) da']` for some
#' intervention density g. Pre-sample treatment values from g per row and
#' attach them under `samples_key`; the empirical m averages alpha over those
#' samples.
#' @param samples_key Per-row column holding a numeric vector of MC samples.
#' @inheritParams ATE
#' @export
StochasticIntervention <- function(samples_key = "shift_samples",
                                   treatment = "a", covariates = "x") {
  .module()$StochasticIntervention(
    samples_key = samples_key,
    treatment = treatment,
    covariates = as.list(covariates)
  )
}


#' Fit a Riesz representer to data.
#'
#' Uses the fast xgboost path. To use early stopping, supply `valid_data`.
#'
#' @param data Data frame with columns matching `feature_keys`.
#' @param m An estimand object from [ATE()] / [ATT()] / [TSM()] / [AdditiveShift()],
#'   or a Python callable with the same `(z, alpha)` opaque signature.
#' @param feature_keys Character vector of column names that index alpha.
#' @param valid_data Optional held-out data frame for early stopping.
#' @param num_boost_round,early_stopping_rounds,learning_rate,max_depth,reg_lambda,subsample,seed
#'   Hyperparameters forwarded to the Python `fit`.
#' @param init Initial value for alpha (in alpha space); one of `"m1"` or a
#'   numeric scalar. NULL takes the loss spec's default (0 for squared, 1 for KL).
#' @param gradient_only If TRUE, disable xgboost's second-order Newton step
#'   and use first-order gradient boosting (Friedman 2001) — Lee-Schuler's
#'   Algorithm 2 exactly. Default FALSE keeps the floored second-order step.
#' @return A `RieszBooster` object; call [predict()] on it.
#' @export
fit_riesz <- function(data, m, feature_keys,
                      valid_data = NULL,
                      num_boost_round = 100L,
                      early_stopping_rounds = NULL,
                      learning_rate = 0.1,
                      max_depth = 5L,
                      reg_lambda = 1.0,
                      subsample = 1.0,
                      seed = 0L,
                      init = NULL,
                      gradient_only = FALSE) {
  rows <- .rows_from_df(data, feature_keys)
  args <- list(
    rows = rows,
    m = m,
    feature_keys = as.list(feature_keys),
    num_boost_round = as.integer(num_boost_round),
    learning_rate = learning_rate,
    max_depth = as.integer(max_depth),
    reg_lambda = reg_lambda,
    subsample = subsample,
    seed = as.integer(seed),
    gradient_only = as.logical(gradient_only)
  )
  if (!is.null(init)) {
    args$init <- init
  }
  if (!is.null(valid_data)) {
    args$valid_rows <- .rows_from_df(valid_data, feature_keys)
  }
  if (!is.null(early_stopping_rounds)) {
    args$early_stopping_rounds <- as.integer(early_stopping_rounds)
  }
  py_booster <- do.call(.module()$fit, args)
  structure(
    list(py = py_booster, feature_keys = as.character(feature_keys), m = m),
    class = "RieszBooster"
  )
}


#' @export
predict.RieszBooster <- function(object, newdata, ...) {
  rows <- .rows_from_df(newdata, object$feature_keys)
  preds_py <- object$py$predict(rows)
  as.numeric(reticulate::py_to_r(preds_py))
}


#' @export
print.RieszBooster <- function(x, ...) {
  best <- reticulate::py_to_r(x$py$best_iteration)
  cat("<RieszBooster>\n")
  cat("  feature_keys :", paste(x$feature_keys, collapse = ", "), "\n")
  cat("  base_score   :", reticulate::py_to_r(x$py$base_score), "\n")
  if (!is.null(best)) {
    cat("  best_iter    :", best, "\n")
    cat("  best_score   :", reticulate::py_to_r(x$py$best_score), "\n")
  }
  invisible(x)
}


#' K-fold cross-fitting for downstream plug-in estimators.
#'
#' @inheritParams fit_riesz
#' @param n_folds Number of CV folds.
#' @param early_stopping_inner_split Fraction of each training fold held out
#'   for inner-fold early stopping (e.g. 0.2). NULL to disable.
#' @param ... Additional fit hyperparameters forwarded to the Python `fit`.
#' @return List with `alpha_hat` (numeric vector of OOF predictions),
#'   `fold_assignment` (integer vector), and `boosters` (list of fold models).
#' @export
crossfit <- function(data, m, feature_keys,
                     n_folds = 5L,
                     seed = 0L,
                     early_stopping_inner_split = NULL,
                     ...) {
  rows <- .rows_from_df(data, feature_keys)
  args <- list(
    rows = rows,
    m = m,
    feature_keys = as.list(feature_keys),
    n_folds = as.integer(n_folds),
    seed = as.integer(seed),
    ...
  )
  if (!is.null(early_stopping_inner_split)) {
    args$early_stopping_inner_split <- early_stopping_inner_split
  }
  res <- do.call(.module()$crossfit, args)
  list(
    alpha_hat = as.numeric(reticulate::py_to_r(res$alpha_hat)),
    fold_assignment = as.integer(reticulate::py_to_r(res$fold_assignment)),
    boosters = lapply(reticulate::py_to_r(res$boosters), function(b) {
      structure(list(py = b, feature_keys = as.character(feature_keys), m = m),
                class = "RieszBooster")
    })
  )
}


#' Diagnostics on a fitted Riesz representer.
#'
#' Provide either a fitted `RieszBooster` plus `data` (and optionally `m`),
#' or a precomputed numeric `alpha_hat`.
#'
#' @export
diagnose_alpha <- function(booster = NULL,
                           data = NULL,
                           m = NULL,
                           alpha_hat = NULL,
                           extreme_threshold = 30,
                           extreme_fraction_warn = 0.01) {
  args <- list(
    extreme_threshold = extreme_threshold,
    extreme_fraction_warn = extreme_fraction_warn
  )
  if (!is.null(alpha_hat)) {
    args$alpha_hat <- as.numeric(alpha_hat)
  } else {
    if (is.null(booster) || is.null(data)) {
      stop("Provide either alpha_hat, or both booster and data.")
    }
    args$booster <- booster$py
    args$rows <- .rows_from_df(data, booster$feature_keys)
    if (!is.null(m)) {
      args$m <- m
    }
  }
  d <- do.call(.module()$diagnose, args)
  out <- list(
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
  structure(out, class = "RieszbDiagnostics")
}


#' @export
print.RieszbDiagnostics <- function(x, ...) {
  cat(x$summary, "\n")
  invisible(x)
}
