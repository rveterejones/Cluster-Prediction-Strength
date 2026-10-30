# Evaluating Cluster Stability Using Prediction Strength

**R · Unsupervised learning · K-means · Simulation · Cluster validation**

## Overview

How many clusters should we choose when the true groups are unknown? Unlike supervised learning, clustering has no observed outcome labels against which to score predictions. This project investigates why minimizing cross-validated held-out within-cluster sum of squares (WSS) can favor too many clusters, then implements **prediction strength** as a stability-based alternative.

The analysis uses simulated two-dimensional data with a known four-cluster structure, allowing the selection methods to be compared against a known answer.

## Research questions

1. Does selecting the number of clusters by minimum five-fold held-out WSS recover the true number of clusters?
2. Can prediction strength identify the correct number of clusters?
3. Does the result persist when cluster sizes are unequal?

## Simulation design

The initial simulation contains **400 observations**, equally divided among **four clusters**, with two independent normally distributed features (standard deviation 1 within each cluster). The cluster centers are `(0, 0)`, `(5, 0)`, `(2.5, 5)`, and `(6, 8.5)`. Two features make the generated structure easy to visualize. All K-means fits use `nstart = 25`.

The sensitivity analysis retains the same cluster centers and spread but draws the number of observations in each cluster from a discrete uniform distribution between 20 and 500.

## Method 1: Five-fold held-out WSS

For each candidate number of clusters, **K = 1, …, 10**:

1. Split observations into five folds.
2. Fit K-means to four folds.
3. For each held-out observation, calculate its squared Euclidean distance to the nearest training centroid.
4. Sum the held-out distances and average across folds.

The resulting held-out WSS decreased as K increased, and **minimizing it selected K = 10**, the largest candidate considered, rather than the true K = 4. The curve nevertheless showed an elbow around K = 4.

**Why this happens:** Adding centroids gives observations more opportunities to lie near a centroid. Unlike classification error, this distance-based criterion does not directly measure whether clusters reproduce stable, meaningful groupings.

## Method 2: Prediction strength

Prediction strength assesses whether observations grouped together by K-means in one sample are also grouped together when assigned to centroids learned from another sample.

For each **K = 2, …, 10**:

1. Randomly split the data into equal training and test subsets.
2. Fit K-means separately to both subsets.
3. Assign test observations to the nearest *training* centroids using `clue::cl_predict()`.
4. Within each cluster identified by the independent test K-means fit, calculate the proportion of distinct observation pairs also assigned to the same training-based cluster.
5. Define prediction strength as the **minimum** of these proportions across test clusters.

For test cluster \(C_j\), the within-cluster agreement is

\[
PS_j(K) = \frac{1}{\binom{|C_j|}{2}}
\sum_{a<b;\,a,b\in C_j} I(\hat c_a^{\mathrm{train}}=\hat c_b^{\mathrm{train}}).
\]

Overall prediction strength is \(PS(K)=\min_j PS_j(K)\). This pairwise approach does not require cluster labels to match numerically across independently fitted models.

In this analysis, the selection rule was to choose the **largest K with prediction strength greater than 0.90**. This correctly selected **K = 4** for the balanced simulation.

## Sensitivity analysis: Unequal cluster sizes

I repeated the prediction-strength procedure after varying cluster sizes while holding the cluster centers and within-cluster standard deviations constant. The procedure **again selected K = 4** using the same 0.90 threshold. In this run, the prediction strength for K = 3 was substantially higher than in the balanced simulation, illustrating that stability scores depend on the underlying data configuration.

## Main findings

| Evaluation approach | Selected K | True K | Interpretation |
|---|---:|---:|---|
| Minimum five-fold held-out WSS | 10 | 4 | Favored the largest number of centroids tested |
| Prediction strength, balanced clusters | 4 | 4 | Recovered the simulated cluster count |
| Prediction strength, unequal cluster sizes | 4 | 4 | Also recovered the simulated cluster count |

These findings describe **one simulated dataset and split per scenario**, not a general guarantee that prediction strength will recover the true K. The 0.90 threshold is a chosen decision rule; results can change with the random split, cluster separation, cluster sizes, and threshold. Test clusters containing fewer than two observations require explicit handling because pairwise agreement is undefined.

## Reproduce the analysis

The companion script, [`analysis.R`](analysis.R), generates the simulated data, computes held-out WSS and prediction strength, prints the selected K values, and creates plots. No external dataset is required.

Required R packages: `dplyr`, `purrr`, `tibble`, `ggplot2`, `clue`, and `sharp`.

```r
source("analysis.R")
```

## Skills demonstrated

- Designing and interpreting simulation studies
- Implementing a cluster-validation algorithm from its statistical definition
- K-means clustering, cross-validation, and resampling
- Pairwise co-membership calculations and model evaluation
- Communicating limitations of unsupervised model selection

## Reference

Tibshirani, R., & Walther, G. (2005). Cluster validation by prediction strength. *Journal of Computational and Graphical Statistics, 14*(3), 511–528.
