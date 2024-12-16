data {
  int<lower=1> N; // Number of participants
  int<lower=1> S; // Number of sessions per participant
  int<lower=1> T; // Number of trials per session
  int<lower=1> NC; // Number of unique choices/actions
  array[N, S, T] int<lower=1, upper=NC> C;   // Participant's own choices
  array[N, S, T] int<lower=0, upper=1> R;    // Participant's own rewards
  array[N, S, T] int<lower=1, upper=NC> PC;  // Partner's choices
  array[N, S, T] int<lower=0, upper=1> PR;   // Partner's rewards
  array[N, S] int<lower=1, upper=2> condition; // Condition indicator (1 = A, 2 = B)
}
parameters {
  // Group-level parameters for condition 2 (baseline)
  real mu_alpha_partner_action_nd_2;
  real<lower=0> sigma_alpha_partner_action_nd;

  real mu_beta_nd_2;
  real<lower=0> sigma_beta_nd;

  // Group-level means of individual-level deltas
  real delta_mu_alpha_partner_action_nd;
  real delta_mu_beta_nd;

  // Individual-level parameters for condition 2
  vector[N] alpha_partner_action_nd_2;
  vector[N] beta_nd_2;

  // Individual-level parameters for condition 2
  vector[N] alpha_partner_action_nd_1;
  vector[N] beta_nd_1;
}
transformed parameters {
  // Transform to appropriate scales
  array[N, 2] real<lower=0.0, upper=1.0> alpha_partner_action;
  array[N, 2] real<lower=0.0> beta;

  for (i in 1:N) {
    // Condition 1
    alpha_partner_action[i, 1] = inv_logit(mu_alpha_partner_action_nd_2 + delta_mu_alpha_partner_action_nd + sigma_alpha_partner_action_nd * alpha_partner_action_nd_1[i]);
    beta[i, 1] = exp(mu_beta_nd_2 + delta_mu_beta_nd + sigma_beta_nd * beta_nd_1[i]);

    // Condition 2
    alpha_partner_action[i, 2] = inv_logit(mu_alpha_partner_action_nd_2 + sigma_alpha_partner_action_nd * alpha_partner_action_nd_2[i]);
    beta[i, 2] = exp(mu_beta_nd_2 + sigma_beta_nd * beta_nd_2[i]);
  }
}
model {
  // Priors for group-level parameters (Condition 2)
  mu_alpha_partner_action_nd_2 ~ normal(0, 1);
  sigma_alpha_partner_action_nd ~ cauchy(0, 2.5);
  mu_beta_nd_2 ~ normal(0, 1);
  sigma_beta_nd ~ cauchy(0, 2.5);

  // Priors for group-level means of individual-level deltas
  delta_mu_alpha_partner_action_nd ~ normal(0, 1);
  delta_mu_beta_nd ~ normal(0, 1);

  // Priors for individual-level parameters (Condition 2)
  alpha_partner_action_nd_2 ~ normal(0, 1);
  beta_nd_2 ~ normal(0, 1);

  // Priors for individual-level parameters (Condition 2)
  alpha_partner_action_nd_1 ~ normal(0, 1);
  beta_nd_1 ~ normal(0, 1);

  // Likelihood
  for (i in 1:N) { // Participant loop
    for (j in 1:S) { // Session loop
      vector[NC] action_values; // Action values

      // Initialize Q-values and action values
      action_values = rep_vector(1.0 / NC, NC);

      int c = condition[i, j]; // Condition for participant i, session j

      for (t in 1:T) { // Trial loop
        vector[NC] log_probs;
        int a = C[i, j, t]; // Participant's choice (1-based)
        int pa = PC[i, j, t]; // Partner's choice (1-based)

        // Compute log probabilities using softmax
        log_probs = beta[i, c] * action_values - log_sum_exp(beta[i, c] * action_values);

        // Update the target log likelihood with own choice
        target += log_probs[a];

        // Update action values based on partner's action
        action_values[pa] += alpha_partner_action[i, c] * (1 - action_values[pa]);
        for (k in 1:NC) {
          if (k != pa) {
            action_values[k] += alpha_partner_action[i, c] * (0 - action_values[k]);
          }
        }
      }
    }
  }
}
generated quantities {
  // Transformed group-level parameters for interpretation
  real<lower=0, upper=1> mu_alpha_partner_action_2 = inv_logit(mu_alpha_partner_action_nd_2);
  real<lower=0> mu_beta_2 = exp(mu_beta_nd_2);

  // Transformed group-level parameters for interpretation
  real<lower=0, upper=1> mu_alpha_partner_action_1 = inv_logit(mu_alpha_partner_action_nd_2 + delta_mu_alpha_partner_action_nd);
  real<lower=0> mu_beta_1 = exp(mu_beta_nd_2 + delta_mu_beta_nd);

  // Compute log-likelihood for each trial
  vector[N * S * T] log_lik;
  int trial_count = 0;

  for (i in 1:N) {
    for (j in 1:S) {
      vector[NC] action_values; // Action values

      // Initialize action values
      action_values = rep_vector(1.0 / NC, NC);

      int c = condition[i, j]; // Condition for participant i, session j

      for (t in 1:T) {
        vector[NC] log_probs;
        int a = C[i, j, t]; // Participant's choice (1-based)
        int pa = PC[i, j, t]; // Partner's choice (1-based)

        // Compute log probabilities using softmax
        log_probs = beta[i, c] * action_values - log_sum_exp(beta[i, c] * action_values);

        // Store log-likelihood for each trial
        trial_count += 1;
        log_lik[trial_count] = log_probs[a];

        // Update action values based on partner's action
        action_values[pa] += alpha_partner_action[i, c] * (1 - action_values[pa]);
        for (k in 1:NC) {
          if (k != pa) {
            action_values[k] += alpha_partner_action[i, c] * (0 - action_values[k]);
          }
        }
      }
    }
  }
}