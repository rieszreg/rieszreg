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
#' `splitter`, `max_bins`. The v0.0.1 names `max_leaves` / `pruning_alpha`
#' remain as deprecated aliases (emit `FutureWarning` from Python).
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
                          max_bins = 255L,
                          # Deprecated aliases for v0.0.1 compatibility.
                          max_leaves = NULL,
                          pruning_alpha = NULL) {
      args <- list(
        estimand = estimand,
        max_depth = as.integer(max_depth),
        min_samples_split = as.integer(min_samples_split),
        min_samples_leaf = as.integer(min_samples_leaf),
        min_weight_fraction_leaf = min_weight_fraction_leaf,
        max_leaf_nodes = as.integer(max_leaf_nodes),
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
      if (!is.null(max_features)) args$max_features <- max_features
      if (!is.null(early_stopping_rounds)) {
        args$early_stopping_rounds <- as.integer(early_stopping_rounds)
      }
      if (!is.null(categorical_features)) {
        args$categorical_features <- as.integer(categorical_features)
      }
      # Deprecated aliases — pass through if explicitly set; Python emits
      # the FutureWarning.
      if (!is.null(max_leaves)) args$max_leaves <- as.integer(max_leaves)
      if (!is.null(pruning_alpha)) args$pruning_alpha <- pruning_alpha
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
