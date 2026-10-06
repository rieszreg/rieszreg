#' rieszreg: shared abstractions for the Riesz regression family.
#'
#' Holds the estimand factories, Bregman-Riesz losses, and the base R6
#' estimator class used by every implementation package (rieszboost, krrr).
#' Implementation packages subclass [RieszEstimatorR6] and add their own
#' backend factories.
#'
#' @keywords internal
"_PACKAGE"


.rr <- new.env(parent = emptyenv())


#' Configure the Python interpreter that holds the rieszreg module.
#'
#' Call this once per session (or set `RETICULATE_PYTHON` before R starts).
#' Implementation packages have their own `use_python_<pkg>()` helpers; you
#' only need to call one of them per session.
#'
#' @param python Path to the Python interpreter or virtualenv directory.
#' @param required Whether reticulate should fail if the Python is unavailable.
#' @export
use_python_rieszreg <- function(python = NULL, required = TRUE) {
  if (!is.null(python)) {
    if (dir.exists(python)) {
      reticulate::use_virtualenv(python, required = required)
    } else {
      reticulate::use_python(python, required = required)
    }
  }
  .rr$mod <- reticulate::import("rieszreg", convert = FALSE)
  invisible(.rr$mod)
}


.module <- function() {
  if (is.null(.rr$mod)) {
    .rr$mod <- reticulate::import("rieszreg", convert = FALSE)
  }
  .rr$mod
}


# Numeric vector -> 1-d numpy array (a length-1 R vector stays an array).
.vec <- function(x) reticulate::np_array(as.numeric(x), dtype = "float64")

# Factor and character columns are converted through their labels, so
# factor(c("0", "1")) becomes 0/1 rather than the integer codes 1/2.
.numeric_column <- function(x, name) {
  if (!is.factor(x) && !is.character(x)) return(as.numeric(x))
  out <- suppressWarnings(as.numeric(as.character(x)))
  bad <- is.na(out) & !is.na(x)
  if (any(bad)) {
    stop(sprintf(
      "Column '%s' has non-numeric values (%s). Recode it as numbers, e.g. treated = 1, control = 0, or drop it from the data.",
      name, paste(utils::head(unique(as.character(x[bad])), 5), collapse = ", ")
    ), call. = FALSE)
  }
  out
}

#' Convert an R data.frame to a pandas DataFrame.
#'
#' Numeric and logical columns flow through as numbers. Factor columns are
#' converted through their labels, so `factor(c("0", "1"))` becomes 0/1.
#' Use this on the predictor data.frame `Z` of `fit(Z, y)` calls; the
#' outcome `y` is passed separately as a numeric vector.
#'
#' @param data An R data.frame.
#' @return A pandas DataFrame (Python object, `convert = FALSE`).
#' @export
df_to_py <- function(data) {
  py_dict <- list()
  for (k in colnames(data)) {
    py_dict[[k]] <- .vec(.numeric_column(data[[k]], k))
  }
  pd <- reticulate::import("pandas", convert = FALSE)
  pd$DataFrame(reticulate::r_to_py(py_dict))
}


# ---- Estimand factories (return opaque Python Estimand instances) ----

# NULL covariates -> Python None ("every non-treatment column").
.cov <- function(covariates) {
  if (is.null(covariates)) NULL else as.list(covariates)
}

#' Average treatment effect estimand: m(alpha)(z) = alpha(1, x) - alpha(0, x).
#' @param treatment Name of the treatment column.
#' @param covariates Character vector of covariate column names. `NULL`
#'   (default) uses every column of the data other than `treatment`.
#' @return A Python `Estimand` object, suitable to pass to any RieszReg
#'   estimator's constructor via `estimand=`.
#' @export
ATE <- function(treatment = "a", covariates = NULL) {
  .module()$ATE(treatment = treatment, covariates = .cov(covariates))
}


#' ATT *partial-estimand* surface: m(alpha)(z) = a*(alpha(1,x) - alpha(0,x)).
#'
#' Full ATT divides by P(A=1) and is not a Riesz functional — combine
#' alpha_partial with a delta-method EIF (Hubbard 2011) downstream.
#' @inheritParams ATE
#' @export
ATT <- function(treatment = "a", covariates = NULL) {
  .module()$ATT(treatment = treatment, covariates = .cov(covariates))
}


#' Treatment-specific mean: m(alpha)(z) = alpha(level, x).
#' @param level Fixed treatment value.
#' @inheritParams ATE
#' @export
TSM <- function(level, treatment = "a", covariates = NULL) {
  .module()$TSM(level = level, treatment = treatment,
                covariates = .cov(covariates))
}


#' Additive shift effect: m(alpha)(z) = alpha(a + delta, x) - alpha(a, x).
#' @param delta Shift magnitude.
#' @inheritParams ATE
#' @export
AdditiveShift <- function(delta, treatment = "a", covariates = NULL) {
  .module()$AdditiveShift(delta = delta, treatment = treatment,
                          covariates = .cov(covariates))
}


#' LASE *partial-estimand* surface. Full LASE divides by P(A < threshold)
#' and is not a Riesz functional.
#' @param delta Shift magnitude.
#' @param threshold Cutoff; only rows with `a < threshold` get shifted.
#' @inheritParams ATE
#' @export
LocalShift <- function(delta, threshold, treatment = "a", covariates = NULL) {
  .module()$LocalShift(delta = delta, threshold = threshold,
                       treatment = treatment, covariates = .cov(covariates))
}


#' Squared L2 norm of the outcome regression: theta = E[mu(X)^2], with
#' m(alpha)(z, y) = alpha(x) * y. Its Riesz representer is mu(x) = E[Y | X = x],
#' so fitting it is a regression of y on the covariates. Requires `y` at fit.
#' @param covariates Character vector of covariate column names. `NULL`
#'   (default) uses every column of the data.
#' @export
OutcomeRegNormSq <- function(covariates = NULL) {
  .module()$OutcomeRegNormSq(covariates = .cov(covariates))
}


# ---- Loss specs ----

#' Squared Riesz loss (default — the standard Lee-Schuler / Chernozhukov objective).
#' @export
SquaredLoss <- function() {
  .module()$SquaredLoss()
}

#' KL-Bregman loss (phi = t log t with exp link). Suitable for density-ratio
#' estimands like TSM / IPSI; requires non-negative m-coefficients.
#' @param max_eta Clip on η before applying the exponential link (numerical safety).
#' @export
KLLoss <- function(max_eta = 50.0) {
  .module()$KLLoss(max_eta = max_eta)
}

#' Bernoulli-Bregman loss (phi = t log t + (1-t) log(1-t), sigmoid link).
#'
#' Forces predictions into (0, 1) — useful when alpha_0 is known to lie
#' there by problem structure.
#' @param max_abs_eta Clip on |η| before applying the sigmoid link.
#' @export
BernoulliLoss <- function(max_abs_eta = 30.0) {
  .module()$BernoulliLoss(max_abs_eta = max_abs_eta)
}

#' Squared Riesz loss with predictions clipped into `(lo, hi)` via a
#' sigmoid-scaled link. Useful for representers with hard prior bounds
#' (e.g. trimmed propensity ratios). Pick bounds tightly around alpha_0;
#' very generous bounds saturate the link and slow boosting.
#' @param lo,hi Lower and upper bounds of the prediction range.
#' @param max_abs_eta Clip on |η| before the sigmoid (numerical safety).
#' @export
BoundedSquaredLoss <- function(lo, hi, max_abs_eta = 30.0) {
  .module()$BoundedSquaredLoss(lo = lo, hi = hi, max_abs_eta = max_abs_eta)
}


# ---- Base R6 class --------------------------------------------------

#' Base R6 class for Riesz regression estimators.
#'
#' Implementation packages (rieszboost, krrr) subclass this and override
#' `initialize` to construct their concrete Python estimator. The shared
#' methods (`fit`, `predict`, `score`, `riesz_loss`, `save`, `diagnose`,
#' `print`) operate on `self$py` (the Python object) and `self$estimand`.
#'
#' Subclasses typically look like:
#'
#' \preformatted{
#' RieszBooster <- R6::R6Class(
#'   "RieszBooster",
#'   inherit = rieszreg::RieszEstimatorR6,
#'   public = list(
#'     initialize = function(estimand, n_estimators = 200L, ...) {
#'       py_obj <- .module()$RieszBooster(estimand = estimand,
#'                                        n_estimators = as.integer(n_estimators), ...)
#'       super$initialize(py_object = py_obj, estimand = estimand)
#'     }
#'   )
#' )
#' }
#'
#' @export
RieszEstimatorR6 <- R6::R6Class(
  "RieszEstimatorR6",
  public = list(
    py = NULL,
    estimand = NULL,

    #' @param py_object The constructed Python estimator (subclasses build this).
    #' @param estimand The Python `Estimand` object that was passed to the constructor.
    initialize = function(py_object, estimand) {
      self$py <- py_object
      self$estimand <- estimand
      invisible(self)
    },

    #' Fit the estimator on a predictor data.frame and an outcome vector.
    #' @param Z Training predictor data (R data.frame; converted to pandas):
    #'   the treatment column plus covariates, matched by name. Leave the
    #'   outcome out of `Z`.
    #' @param y Optional outcome vector (numeric). The built-in treatment
    #'   estimands ignore it; estimands whose functional reads the outcome
    #'   need it.
    #' @param eval_set Optional held-out predictor data.frame for early
    #'   stopping / λ selection.
    #' @param eval_y Optional outcome vector aligned with `eval_set`.
    fit = function(Z, y = NULL, eval_set = NULL, eval_y = NULL) {
      Z_py <- df_to_py(Z)
      args <- list(Z = Z_py)
      if (!is.null(y)) args$y <- .vec(y)
      if (!is.null(eval_set)) {
        args$eval_set <- df_to_py(eval_set)
        if (!is.null(eval_y)) args$eval_y <- .vec(eval_y)
      }
      do.call(self$py$fit, args)
      invisible(self)
    },

    #' Predict α̂ on a predictor data.frame. Returns a numeric vector.
    predict = function(Z) {
      preds <- self$py$predict(df_to_py(Z))
      as.numeric(reticulate::py_to_r(preds))
    },

    #' Negative held-out Riesz loss (sklearn higher-is-better).
    #' @param Z Held-out predictor data.frame.
    #' @param y Optional held-out outcome vector for Y-dependent estimands.
    score = function(Z, y = NULL) {
      args <- list(df_to_py(Z))
      if (!is.null(y)) args$y <- .vec(y)
      reticulate::py_to_r(do.call(self$py$score, args))
    },

    #' Held-out per-row Riesz loss.
    #' @param Z Held-out predictor data.frame.
    #' @param y Optional held-out outcome vector for Y-dependent estimands.
    riesz_loss = function(Z, y = NULL) {
      args <- list(df_to_py(Z))
      if (!is.null(y)) args$y <- .vec(y)
      reticulate::py_to_r(do.call(self$py$riesz_loss, args))
    },

    #' Save the fitted estimator to a directory.
    save = function(path) {
      self$py$save(path)
      invisible(self)
    },

    #' Diagnostics. Returns a list mirroring the Python `Diagnostics` dataclass.
    #' @param Z Held-out predictor data.frame.
    #' @param y Optional held-out outcome vector for Y-dependent estimands.
    diagnose = function(Z, y = NULL, ...) {
      args <- list(df_to_py(Z), ...)
      if (!is.null(y)) args$y <- .vec(y)
      d <- do.call(self$py$diagnose, args)
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
      cat("<", class(self)[[1]], ">\n", sep = "")
      cat("  estimand   :", reticulate::py_to_r(self$estimand$name), "\n")
      if (reticulate::py_has_attr(self$py, "predictor_")) {
        cat("  status     : fitted\n")
        best_iter <- reticulate::py_to_r(self$py$best_iteration_)
        if (!is.null(best_iter)) cat("  best_iter  :", best_iter, "\n")
        bs <- reticulate::py_to_r(self$py$best_score_)
        if (!is.null(bs)) cat("  best_score :", bs, "\n")
      } else {
        cat("  status     : unfitted\n")
      }
      invisible(self)
    }
  )
)
