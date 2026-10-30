# Cluster validation simulation: held-out WSS versus prediction strength
# Adapted from statistical learning assignment.
library(dplyr)
library(purrr)
library(tibble)
library(ggplot2)
library(clue)
library(sharp)

centers <- matrix(c(0, 0, 5, 0, 2.5, 5, 6, 8.5), ncol = 2, byrow = TRUE)
K_true <- nrow(centers)

simulate_clusters <- function(sizes, seed = 42) {
  set.seed(seed)
  map_dfr(seq_along(sizes), function(k) {
    tibble(x1 = rnorm(sizes[k], centers[k, 1], 1),
           x2 = rnorm(sizes[k], centers[k, 2], 1),
           true_cluster = factor(k))
  })
}

heldout_wss <- function(test, centers) {
  test <- as.matrix(test)
  centers <- as.matrix(centers)
  distances <- sapply(seq_len(nrow(centers)), function(k) {
    rowSums(sweep(test, 2, centers[k, ], "-")^2)
  })
  sum(apply(matrix(distances, nrow = nrow(test)), 1, min))
}

cv_wss <- function(X, max_k = 10, seed = 42) {
  set.seed(seed)
  folds <- sample(rep(1:5, length.out = nrow(X)))
  map_dfr(1:max_k, function(k) {
    fold_wss <- map_dbl(1:5, function(fold) {
      km <- kmeans(X[folds != fold, , drop = FALSE], centers = k, nstart = 25)
      heldout_wss(X[folds == fold, , drop = FALSE], km$centers)
    })
    tibble(K = k, mean_wss = mean(fold_wss))
  })
}

# Within each test cluster, check whether pairs remain together under
# assignments to training centroids. Take the minimum agreement.
prediction_strength <- function(test_clusters, pred_clusters) {
  strengths <- map_dbl(unique(test_clusters), function(k) {
    pred <- pred_clusters[test_clusters == k]
    if (length(pred) < 2) return(NA_real_)
    cm <- sharp::CoMembership(pred)
    mean(cm[upper.tri(cm)])
  })
  if (anyNA(strengths)) return(NA_real_)
  min(strengths)
}

calculate_ps <- function(X, max_k = 10, seed = 42) {
  set.seed(seed)
  train_ind <- sample(nrow(X), size = floor(0.5 * nrow(X)))
  train <- X[train_ind, , drop = FALSE]
  test <- X[-train_ind, , drop = FALSE]
  map_dfr(2:max_k, function(k) {
    train_fit <- kmeans(train, centers = k, nstart = 25)
    test_fit <- kmeans(test, centers = k, nstart = 25)
    predicted <- as.integer(clue::cl_predict(train_fit, newdata = test))
    tibble(K = k, prediction_strength = prediction_strength(test_fit$cluster, predicted))
  })
}

select_ps <- function(results, threshold = 0.90) {
  eligible <- results$K[!is.na(results$prediction_strength) &
                        results$prediction_strength > threshold]
  if (!length(eligible)) return(NA_integer_)
  max(eligible)
}

# Scenario 1: four equally sized clusters
balanced <- simulate_clusters(rep(100, K_true))
X <- select(balanced, x1, x2)
wss_results <- cv_wss(X)
ps_balanced <- calculate_ps(X)
cat("WSS selected K:", wss_results$K[which.min(wss_results$mean_wss)], "\n")
cat("Prediction strength selected K (balanced):", select_ps(ps_balanced), "\n")

p_data <- ggplot(balanced, aes(x1, x2, color = true_cluster)) +
  geom_point(alpha = 0.6) + labs(title = "Four simulated clusters", color = "True cluster") + theme_bw()
p_wss <- ggplot(wss_results, aes(K, mean_wss)) + geom_line() + geom_point() +
  scale_x_continuous(breaks = 1:10) + labs(title = "Five-fold held-out WSS", y = "Mean held-out WSS") + theme_bw()
p_ps <- ggplot(ps_balanced, aes(K, prediction_strength)) + geom_line() + geom_point() +
  geom_hline(yintercept = 0.90, linetype = "dashed") +
  scale_x_continuous(breaks = 2:10) + labs(title = "Prediction strength: balanced clusters") + theme_bw()

# Scenario 2: unequal cluster sizes
set.seed(42)
cluster_sizes <- sample(20:500, K_true, replace = TRUE)
unequal <- simulate_clusters(cluster_sizes)
ps_unequal <- calculate_ps(select(unequal, x1, x2))
cat("Unequal cluster sizes:", cluster_sizes, "\n")
cat("Prediction strength selected K (unequal):", select_ps(ps_unequal), "\n")
p_unequal <- ggplot(ps_unequal, aes(K, prediction_strength)) + geom_line() + geom_point() +
  geom_hline(yintercept = 0.90, linetype = "dashed") +
  scale_x_continuous(breaks = 2:10) + labs(title = "Prediction strength: unequal clusters") + theme_bw()

# Plots are available as p_data, p_wss, p_ps, and p_unequal.
print(p_data)
print(p_wss)
print(p_ps)
print(p_unequal)
