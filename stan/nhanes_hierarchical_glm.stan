
// Hierarchical Bernoulli GLM — NHANES diabetes classification
// Model structure follows UBC STAT 405 w08 topic03 style:
// parameters -> transformed parameters (logistic mean) -> model (likelihood)

data {
  int<lower=0> N;                              // training observations
  int<lower=1> K;                              // number of features
  int<lower=1> J;                              // number of race groups
  array[N] int<lower=0, upper=1> y;            // diabetes label
  matrix[N, K] X;                              // feature matrix
  array[N] int<lower=1, upper=J> race;         // race group index (1-indexed)

  int<lower=0> N_test;                         // test observations
  matrix[N_test, K] X_test;                    // test feature matrix
  array[N_test] int<lower=1, upper=J> race_test;
}

parameters {
  // Hyperparameters for hierarchical intercepts
  real mu_alpha;
  real<lower=0> sigma_alpha;

  // Non-centered parameterization (same trick used in UBC site's examples)
  vector[J] alpha_offset;

  // Shared regression coefficients
  vector[K] beta;
}

transformed parameters {
  // Group intercepts — assembled here like the site assembles mu from parameters
  vector[J] alpha = mu_alpha + sigma_alpha * alpha_offset;

  // Predicted probability for each training observation
  // inv_logit is Stan's sigmoid — same role as inv_logit in the UBC Beta model
  vector[N] mu = inv_logit(alpha[race] + X * beta);
}

model {
  // Weakly informative priors — same philosophy as the UBC site's priors
  mu_alpha    ~ normal(0, 1);
  sigma_alpha ~ exponential(1);
  alpha_offset ~ normal(0, 1);
  beta        ~ normal(0, 1);

  // Bernoulli likelihood with logit parameterization
  y ~ bernoulli(mu);
}

generated quantities {
  // Posterior predictive probabilities for test set
  // Mirrors the UBC site's generated quantities pattern
  vector[N_test] mu_test =
    inv_logit(alpha[race_test] + X_test * beta);

  array[N_test] int y_test_rep;
  for (n in 1:N_test) {
    y_test_rep[n] = bernoulli_rng(mu_test[n]);
  }
}
