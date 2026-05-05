#' riesznet: R wrapper for the Python riesznet library
#'
#' Mirrors the Python sklearn-style API. Configure once with
#' `use_python_riesznet()`, then construct a [RieszNet] and call
#' `$fit(df)` / `$predict(df)`.
#'
#' Estimand and loss factories live in the shared `rieszreg` R package and are
#' re-exported from here for convenience.
#'
#' Scope note: the R wrapper exposes the simple-MLP knobs (`hidden_sizes`,
#' `activation`, `dropout`, `learning_rate`, `weight_decay`, `epochs`,
#' `device`). Custom torch architectures are Python-only — write the
#' `nn.Module` factory in Python and call into Python via reticulate.
#'
#' @keywords internal
"_PACKAGE"


.rn <- new.env(parent = emptyenv())


#' Configure the Python interpreter that holds the riesznet module.
#'
#' Call this once per session before any other riesznet function. Forwards
#' to `reticulate::use_python` / `reticulate::use_virtualenv` as appropriate.
#'
#' @param python Path to the Python interpreter or virtualenv directory.
#' @param required Whether reticulate should fail if the Python is unavailable.
#' @export
use_python_riesznet <- function(python = NULL, required = TRUE) {
  if (!is.null(python)) {
    if (dir.exists(python)) {
      reticulate::use_virtualenv(python, required = required)
    } else {
      reticulate::use_python(python, required = required)
    }
  }
  .rn$mod <- reticulate::import("riesznet", convert = FALSE)
  invisible(.rn$mod)
}


.module <- function() {
  if (is.null(.rn$mod)) {
    .rn$mod <- reticulate::import("riesznet", convert = FALSE)
  }
  .rn$mod
}


# ---- Main estimator (R6 subclass) -----------------------------------------

#' RieszNet — neural-network Riesz regression.
#'
#' Subclass of [rieszreg::RieszEstimatorR6] that defaults the backend to a
#' simple MLP trained with Adam and surfaces the simple-MLP knobs
#' (`hidden_sizes`, `activation`, `dropout`, `learning_rate`, `weight_decay`,
#' `epochs`, `device`) on the constructor.
#'
#' Custom torch architectures are Python-only. R users who need a custom
#' `nn.Module` write the factory in Python and call into Python via reticulate.
#'
#' @export
RieszNet <- R6::R6Class(
  "RieszNet",
  inherit = rieszreg::RieszEstimatorR6,
  public = list(
    initialize = function(estimand,
                          hidden_sizes = c(64L, 64L),
                          activation = "relu",
                          dropout = 0.0,
                          learning_rate = 1e-3,
                          weight_decay = 0.0,
                          epochs = 200L,
                          device = "cpu",
                          dtype = "float32",
                          grad_clip_norm = NULL,
                          loss = NULL,
                          init = NULL,
                          validation_fraction = 0.0,
                          early_stopping_rounds = NULL,
                          snapshot_epochs = NULL,
                          random_state = 0L) {
      # Build the hidden_sizes Python tuple from the R integer vector.
      hs <- reticulate::tuple(lapply(as.integer(hidden_sizes), as.integer))
      args <- list(
        estimand = estimand,
        hidden_sizes = hs,
        activation = activation,
        dropout = dropout,
        learning_rate = learning_rate,
        weight_decay = weight_decay,
        epochs = as.integer(epochs),
        device = device,
        dtype = dtype,
        validation_fraction = validation_fraction,
        random_state = as.integer(random_state)
      )
      if (!is.null(grad_clip_norm)) args$grad_clip_norm <- grad_clip_norm
      if (!is.null(loss)) args$loss <- loss
      if (!is.null(init)) args$init <- init
      if (!is.null(early_stopping_rounds)) {
        args$early_stopping_rounds <- as.integer(early_stopping_rounds)
      }
      if (!is.null(snapshot_epochs)) {
        args$snapshot_epochs <- reticulate::r_to_py(
          as.list(as.integer(snapshot_epochs))
        )
      }
      py_object <- do.call(.module()$RieszNet, args)
      super$initialize(py_object = py_object, estimand = estimand)
    },

    #' Predict α̂ at every snapshot epoch from one training run.
    #'
    #' Returns an n_rows × n_epochs numeric matrix. Column ``j`` is the
    #' prediction obtained from the ``state_dict`` at snapshot epoch
    #' ``epochs[j]`` (or every retained snapshot when `epochs` is `NULL`).
    #' Each column equals a fresh fit at ``epochs[j]`` to within Adam's
    #' deterministic-trajectory tolerance.
    #'
    #' Requires `snapshot_epochs` (or the auto-grid default) to have been
    #' enabled at construction.
    #' @param X Feature data.frame.
    #' @param epochs Integer vector of epoch ticks (subset of stored grid);
    #'   defaults to the full stored grid.
    predict_path = function(X, epochs = NULL) {
      if (is.null(epochs)) {
        py_ep <- NULL
      } else {
        py_ep <- reticulate::r_to_py(as.list(as.integer(epochs)))
      }
      out <- self$py$predict_path(rieszreg::df_to_py(X), py_ep)
      m <- as.matrix(reticulate::py_to_r(out))
      ep_used <- if (is.null(epochs)) {
        as.integer(reticulate::py_to_r(self$py$predictor_$snapshot_epochs))
      } else {
        as.integer(epochs)
      }
      colnames(m) <- paste0("epoch=", ep_used)
      m
    }
  )
)


#' Load a fitted RieszNet from a directory written by `$save()`.
#'
#' For built-in estimands, fully reconstructs the estimand from the metadata.
#' For custom estimands (Python-only), pass `estimand=` explicitly.
#' @param path Directory path.
#' @param estimand Optional user-supplied `Estimand` (required for custom m).
#' @export
load_riesz_net <- function(path, estimand = NULL) {
  args <- list(path = path)
  if (!is.null(estimand)) args$estimand <- estimand
  py_obj <- do.call(.module()$RieszNet$load, args)
  rn <- RieszNet$new(estimand = py_obj$estimand)
  rn$py <- py_obj
  rn$estimand <- py_obj$estimand
  rn
}


# Estimand and loss factories are re-exported from rieszreg via NAMESPACE
# (importFrom + export), so `library(riesznet); TSM(level=1)` works.
