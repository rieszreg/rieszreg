# Reference papers

LaTeX sources fetched from arXiv. Not committed (see `.gitignore`); refetch with the IDs below.

| Directory | arXiv ID | Citation |
|---|---|---|
| `lee-schuler-rieszboost/` | [2501.04871](https://arxiv.org/abs/2501.04871) | Lee & Schuler, *RieszBoost: Gradient Boosting for Riesz Regression*. The core method this library implements. |
| `chernozhukov-riesz-regression/` | [2104.14737](https://arxiv.org/abs/2104.14737) | Chernozhukov et al., *Automatic Debiased Machine Learning via Riesz Regression*. Origin of the squared Riesz loss. |
| `chernozhukov-grf-nn/` | [2110.03031](https://arxiv.org/abs/2110.03031) | Chernozhukov et al., *RieszNet and ForestRiesz*. NN/RF implementations of Riesz regression. |
| `singh-kernel-riesz/` | [2102.11076](https://arxiv.org/abs/2102.11076) | Singh, *Kernel Ridge Riesz Representers*. Closed-form RKHS estimator. |
| `hines-miles-bregman-riesz/` | [2510.16127](https://arxiv.org/abs/2510.16127) | Hines & Miles, *Learning density ratios in causal inference using Bregman-Riesz regression*. Bregman generalization. |
| `kato-bregman-unified/` | [2601.07752](https://arxiv.org/abs/2601.07752) | Kato, *Riesz Representer Fitting under Bregman Divergence: A Unified Framework for Debiased Machine Learning*. The other Bregman generalization. |
| `vdl-autodml-smooth-functionals/` | [2501.11868](https://arxiv.org/abs/2501.11868) | van der Laan et al., *AutoDML for Smooth Functionals of Nonparametric M-Estimands*. Beyond linear functionals. |

To refetch:

```sh
for entry in \
  "lee-schuler-rieszboost:2501.04871" \
  "chernozhukov-riesz-regression:2104.14737" \
  "chernozhukov-grf-nn:2110.03031" \
  "singh-kernel-riesz:2102.11076" \
  "hines-miles-bregman-riesz:2510.16127" \
  "kato-bregman-unified:2601.07752" \
  "vdl-autodml-smooth-functionals:2501.11868"; do
    name="${entry%%:*}"; id="${entry##*:}"
    mkdir -p "$name"
    curl -sL -o "$name/source.tar.gz" "https://arxiv.org/e-print/$id"
    (cd "$name" && tar -xzf source.tar.gz)
done
```
