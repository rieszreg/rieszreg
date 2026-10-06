#' riesztree: R wrapper for the Python riesztree library
#'
#' Mirrors the Python sklearn-style API. Configure once with
#' `use_python_riesztree()`, then construct a [RieszTreeRegressor] and call
#' `$fit(df)` / `$predict(df)`.
#'
#' Estimand and loss factories live in the shared `rieszreg` R package and are
#' re-exported from here for convenience.
#'
#' @keywords internal
"_PACKAGE"


.rt <- new.env(parent = emptyenv())


#' Configure the Python interpreter that holds the riesztree module.
#'
#' Call this once per session before any other riesztree function. Forwards
#' to `reticulate::use_python` / `reticulate::use_virtualenv` as appropriate.
#'
#' @param python Path to the Python interpreter or virtualenv directory.
#' @param required Whether reticulate should fail if the Python is unavailable.
#' @export
use_python_riesztree <- function(python = NULL, required = TRUE) {
  if (!is.null(python)) {
    if (dir.exists(python)) {
      reticulate::use_virtualenv(python, required = required)
    } else {
      reticulate::use_python(python, required = required)
    }
  }
  .rt$mod <- reticulate::import("riesztree", convert = FALSE)
  invisible(.rt$mod)
}


.module <- function() {
  if (is.null(.rt$mod)) {
    .rt$mod <- reticulate::import("riesztree", convert = FALSE)
  }
  .rt$mod
}


# ---- Main estimator (R6 subclass) -----------------------------------------

#' RieszTreeRegressor — single-tree Riesz regression.
#'
#' Subclass of [rieszreg::RieszEstimatorR6] that defaults the backend to a
#' single tree fit on the augmented Bregman-Riesz loss. Hyperparameters
#' mirror `sklearn.tree.DecisionTreeRegressor` where the augmented
#' Bregman-Riesz setting allows: `max_depth`, `min_samples_split`,
#' `min_samples_leaf`, `min_weight_fraction_leaf`, `max_leaf_nodes`,
#' `max_features`, `growth_policy`, `min_impurity_decrease`, `ccp_alpha`,
#' `early_stopping_rounds`, `validation_fraction`, `categorical_features`,
#' `splitter`, `max_bins`. `max_depth = NULL` and `max_leaf_nodes = NULL`
#' mean no limit. `max_features` follows Python: an integer (`3L`, or a
#' whole number above 1) is a feature count, while a fraction in (0, 1] or a
#' string such as `"sqrt"` is passed as is, so `max_features = 1` means all
#' features. `categorical_features` takes 1-based positions in the
#' estimand's input columns (treatment first, then covariates).
#'
#' @export
RieszTreeRegressor <- R6::R6Class(
  "RieszTreeRegressor",
  inherit = rieszreg::RieszEstimatorR6,
  public = list(
    initialize = function(estimand,
                          loss = NULL,
                          max_depth = 8L,
                          min_samples_split = 20L,
                          min_samples_leaf = 10L,
                          min_weight_fraction_leaf = 0.0,
                          max_leaf_nodes = 31L,
                          max_features = NULL,
                          growth_policy = "depthwise",
                          min_impurity_decrease = 0.0,
                          ccp_alpha = 0.0,
                          early_stopping_rounds = NULL,
                          validation_fraction = 0.1,
                          categorical_features = NULL,
                          init = NULL,
                          random_state = 0L,
                          splitter = "exact",
                          max_bins = 255L) {
      args <- list(
        estimand = estimand,
        min_samples_split = as.integer(min_samples_split),
        min_samples_leaf = as.integer(min_samples_leaf),
        min_weight_fraction_leaf = min_weight_fraction_leaf,
        growth_policy = growth_policy,
        min_impurity_decrease = min_impurity_decrease,
        ccp_alpha = ccp_alpha,
        validation_fraction = validation_fraction,
        random_state = as.integer(random_state),
        splitter = splitter,
        max_bins = as.integer(max_bins)
      )
      if (!is.null(loss)) args$loss <- loss
      if (!is.null(init)) args$init <- init
      args["max_depth"] <- list(if (is.null(max_depth)) NULL else as.integer(max_depth))
      args["max_leaf_nodes"] <- list(if (is.null(max_leaf_nodes)) NULL else as.integer(max_leaf_nodes))
      if (!is.null(max_features)) {
        # An R integer or a whole number above 1 is a feature count (Python
        # int). 1 stays a float: in Python, 1.0 means all features.
        count <- is.integer(max_features) ||
          (is.numeric(max_features) && max_features > 1 &&
             max_features == round(max_features))
        args$max_features <- if (count) as.integer(max_features) else max_features
      }
      if (!is.null(early_stopping_rounds)) {
        args$early_stopping_rounds <- as.integer(early_stopping_rounds)
      }
      if (!is.null(categorical_features)) {
        pos <- as.integer(categorical_features)
        if (any(is.na(pos) | pos < 1L)) {
          stop("`categorical_features` takes 1-based column positions (>= 1).",
               call. = FALSE)
        }
        args$categorical_features <- as.list(pos - 1L)
      }
      py_object <- do.call(.module()$RieszTreeRegressor, args)
      super$initialize(py_object = py_object, estimand = estimand)
    }
  )
)


#' Load a fitted RieszTreeRegressor from a directory written by `$save()`.
#'
#' For built-in estimands, fully reconstructs the estimand from the metadata.
#' For custom estimands (Python-only), pass `estimand=` explicitly.
#' @param path Directory path.
#' @param estimand Optional user-supplied `Estimand` (required for custom m).
#' @export
load_riesz_tree_regressor <- function(path, estimand = NULL) {
  args <- list(path = path)
  if (!is.null(estimand)) args$estimand <- estimand
  py_obj <- do.call(.module()$RieszTreeRegressor$load, args)
  rt <- RieszTreeRegressor$new(estimand = py_obj$estimand)
  rt$py <- py_obj
  rt$estimand <- py_obj$estimand
  rt
}


# Estimand and loss factories are re-exported from rieszreg via NAMESPACE.
