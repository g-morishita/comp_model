data {
  int<lower=1> N; // Number of participants
  int<lower=1> S; // Number of sessions per participant
  int<lower=1> T; // Number of trials per session
  int<lower=1> NC; // Number of unique choices/actions
  array[N, S, T] int<lower=1, upper=NC> C;   // Participant's own choices
  array[N, S, T] int<lower=0, upper=1> R;    // Participant's own rewards
  array[N, S] int<lower=1, upper=2> condition; // Condition indicator (1 = A, 2 = B)
}
parameters {
  // Group-level parameters for condition 2 (baseline)
  real mu_alpha_own_nd_2;
  real<lower=0> sigma_alpha_own_nd;

  real mu_beta_nd_2;
  real<lower=0> sigma_beta_nd;

  // Group-level means of individual-level deltas
  real delta_mu_alpha_own_nd;
  real delta_mu_beta_nd;

  // Individual-level parameters for condition 2
  vector[N] alpha_own_nd_2;
  vector[N] beta_nd_2;

  // Individual-level parameters for condition 2
  vector[N] alpha_own_nd_1;
  vector[N] beta_nd_1;
}
transformed parameters {
  // Transform to appropriate scales
  array[N, 2] real<lower=0.0, upper=1.0> alpha_own;
  array[N, 2] real<lower=0.0> beta;

  for (i in 1:N) {
    // Condition 1
    alpha_own[i, 1] = inv_logit(mu_alpha_own_nd_2 + delta_mu_alpha_own_nd + sigma_alpha_own_nd * alpha_own_nd_1[i]);
    beta[i, 1] = exp(mu_beta_nd_2 + delta_mu_beta_nd + sigma_beta_nd * beta_nd_1[i]);

    // Condition 2
    alpha_own[i, 2] = inv_logit(mu_alpha_own_nd_2 + sigma_alpha_own_nd * alpha_own_nd_2[i]);
    beta[i, 2] = exp(mu_beta_nd_2 + sigma_beta_nd * beta_nd_2[i]);
  }
}
model {
  // Priors for group-level parameters (Condition 2)
  mu_alpha_own_nd_2 ~ normal(0, 1);
  sigma_alpha_own_nd ~ cauchy(0, 0.25);
  mu_beta_nd_2 ~ normal(0, 1);
  sigma_beta_nd ~ cauchy(0, 0.25);

  // Priors for group-level means of individual-level deltas
  delta_mu_alpha_own_nd ~ normal(0, 1);
  delta_mu_beta_nd ~ normal(0, 1);

  // Priors for individual-level parameters (Condition 2)
  alpha_own_nd_2 ~ normal(0, 1);
  beta_nd_2 ~ normal(0, 1);

  // Priors for individual-level parameters (Condition 2)
  alpha_own_nd_1 ~ normal(0, 1);
  beta_nd_1 ~ normal(0, 1);

  // Likelihood
  for (i in 1:N) { // Participant loop
    for (j in 1:S) { // Session loop
      vector[NC] Q; // Q-values for each action

      // Initialize Q-values and action values
      Q = rep_vector(0.5, NC);

      int c = condition[i, j]; // Condition for participant i, session j

      for (t in 1:T) { // Trial loop
        vector[NC] combined_values;
        vector[NC] log_probs;
        int a = C[i, j, t]; // Participant's choice (1-based)
        real r = R[i, j, t]; // Participant's reward

        // Compute log probabilities using softmax
        log_probs = beta[i, c] * Q - log_sum_exp(beta[i, c] * Q);

        // Update the target log likelihood with own choice
        target += log_probs[a];

        // Update Q-values based on own choice and reward
        Q[a] += alpha_own[i, c] * (r - Q[a]);
      }
    }
  }
}
generated quantities {
  // Transformed group-level parameters for interpretation
  real<lower=0, upper=1> mu_alpha_own_2 = inv_logit(mu_alpha_own_nd_2);
  real<lower=0> mu_beta_2 = exp(mu_beta_nd_2);

  // Transformed group-level parameters for interpretation
  real<lower=0, upper=1> mu_alpha_own_1 = inv_logit(mu_alpha_own_nd_2 + delta_mu_alpha_own_nd);
  real<lower=0> mu_beta_1 = exp(mu_beta_nd_2 + delta_mu_beta_nd);

  // Compute log-likelihood for each trial
  vector[N * S * T] log_lik;
  int trial_count = 0;

  for (i in 1:N) {
    for (j in 1:S) {
      vector[NC] Q; // Q-values for each action

      // Initialize Q-values and action values
      Q = rep_vector(0.5, NC);

      int c = condition[i, j]; // Condition for participant i, session j

      for (t in 1:T) {
        vector[NC] combined_values;
        vector[NC] log_probs;
        int a = C[i, j, t]; // Participant's choice (1-based)
        real r = R[i, j, t]; // Participant's reward

        // Compute log probabilities using softmax
        log_probs = beta[i, c] * Q - log_sum_exp(beta[i, c] * Q);

        // Store log-likelihood for each trial
        trial_count += 1;
        log_lik[trial_count] = log_probs[a];

        // Update Q-values based on own choice and reward
        Q[a] += alpha_own[i, c] * (r - Q[a]);
      }
    }
  }
}
