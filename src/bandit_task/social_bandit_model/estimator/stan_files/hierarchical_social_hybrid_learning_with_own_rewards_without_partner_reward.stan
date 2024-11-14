data {
  int<lower=1> N; // Number of participants
  int<lower=1> S; // Number of sessions per participant
  int<lower=1> T; // Number of trials per session
  int<lower=1> NC; // Number of unique choices/actions
  array[N, S, T] int<lower=1, upper=NC> C;   // Participant's own choices
  array[N, S, T] int<lower=0, upper=1> R;    // Participant's own rewards
  array[N, S, T] int<lower=1, upper=NC> PC;  // Partner's choices
  array[N, S, T] int<lower=0, upper=1> PR;   // Partner's rewards
}
parameters {
  // Individual-level latent parameters (standard normal)
  vector[N] alpha_own_nd;
  vector[N] alpha_partner_reward_nd;
  vector[N] alpha_partner_action_nd;
  vector[N] beta_nd;data {
  int<lower=1> N; // Number of participants
  int<lower=1> S; // Number of sessions per participant
  int<lower=1> T; // Number of trials per session
  int<lower=1> NC; // Number of unique choices/actions
  int<lower=1, upper=NC> C[N, S, T];   // Participant's own choices
  real<lower=0, upper=1> R[N, S, T];    // Participant's own rewards
  int<lower=1, upper=NC> PC[N, S, T];  // Partner's choices
}

parameters {
  // Individual-level latent parameters (standard normal)
  vector[N] lr_own_nd;
  vector[N] lr_partner_action_nd;
  vector[N] beta_nd;
  vector[N] omega_nd;

  // Group-level parameters (means and standard deviations)
  real mu_lr_own_nd;
  real<lower=0> sigma_lr_own_nd;
  real mu_lr_partner_action_nd;
  real<lower=0> sigma_lr_partner_action_nd;
  real mu_beta_nd;
  real<lower=0> sigma_beta_nd;
  real mu_omega_nd;
  real<lower=0> sigma_omega_nd;
}

transformed parameters {
  vector<lower=0, upper=1>[N] lr_own;
  vector<lower=0, upper=1>[N] lr_partner_action;
  vector<lower=0>[N] beta;
  vector<lower=0, upper=1>[N] omega;

  // Transform individual-level parameters to appropriate scales
  lr_own = inv_logit(mu_lr_own_nd + sigma_lr_own_nd * lr_own_nd);
  lr_partner_action = inv_logit(mu_lr_partner_action_nd + sigma_lr_partner_action_nd * lr_partner_action_nd);
  beta = exp(mu_beta_nd + sigma_beta_nd * beta_nd);
  omega = inv_logit(mu_omega_nd + sigma_omega_nd * omega_nd);
}

model {
  // Priors for individual-level latent parameters
  lr_own_nd ~ normal(0, 1);
  lr_partner_action_nd ~ normal(0, 1);
  beta_nd ~ normal(0, 1);
  omega_nd ~ normal(0, 1);

  // Priors for group-level parameters
  mu_lr_own_nd ~ normal(0, 1);
  sigma_lr_own_nd ~ cauchy(0, 2.5);
  mu_lr_partner_action_nd ~ normal(0, 1);
  sigma_lr_partner_action_nd ~ cauchy(0, 2.5);
  mu_beta_nd ~ normal(0, 1);
  sigma_beta_nd ~ cauchy(0, 2.5);
  mu_omega_nd ~ normal(0, 1);
  sigma_omega_nd ~ cauchy(0, 2.5);

  // Likelihood
  for (i in 1:N) { // Participant loop
    for (j in 1:S) { // Session loop
      vector[NC] Q; // Q-values for each action
      vector[NC] action_values; // Action values

      // Initialize Q-values and action values
      Q = rep_vector(0.5, NC);
      action_values = rep_vector(1.0 / NC, NC);

      for (t in 1:T) { // Trial loop
        vector[NC] combined_values;
        vector[NC] log_probs;
        int a = C[i, j, t]; // Participant's choice (1-based indexing)
        int pa = PC[i, j, t]; // Partner's choice (1-based indexing)
        real r = R[i, j, t]; // Participant's reward

        // Compute combined values
        combined_values = omega[i] * Q + (1 - omega[i]) * action_values;

        // Compute log probabilities using softmax
        log_probs = beta[i] * combined_values - log_sum_exp(beta[i] * combined_values);

        // Update the target log likelihood with own choice
        target += log_probs[a];

        // Update Q-values based on own choice and reward
        Q[a] += lr_own[i] * (r - Q[a]);

        // Update action values based on partner's action
        for (k in 1:NC) {
          if (k == pa) {
            action_values[k] += lr_partner_action[i] * (1 - action_values[k]);
          } else {
            action_values[k] += lr_partner_action[i] * (0 - action_values[k]);
          }
        }
      }
    }
  }
}

generated quantities {
  // Transformed group-level parameters for interpretation
  real<lower=0, upper=1> mu_lr_own = inv_logit(mu_lr_own_nd);
  real<lower=0, upper=1> mu_lr_partner_action = inv_logit(mu_lr_partner_action_nd);
  real<lower=0> mu_beta = exp(mu_beta_nd);
  real<lower=0, upper=1> mu_omega = inv_logit(mu_omega_nd);

  vector[N * S * T] log_lik; // Log-likelihood for each trial
  int trial_count = 0;

  // Compute log-likelihood for each trial
  for (i in 1:N) {
    for (j in 1:S) {
      vector[NC] Q; // Q-values for each action
      vector[NC] action_values; // Action values

      // Initialize Q-values and action values
      Q = rep_vector(0.5, NC);
      action_values = rep_vector(1.0 / NC, NC);

      for (t in 1:T) {
        vector[NC] combined_values;
        vector[NC] log_probs;
        int a = C[i, j, t]; // Participant's choice (1-based indexing)
        int pa = PC[i, j, t]; // Partner's choice (1-based indexing)
        real r = R[i, j, t]; // Participant's reward

        // Compute combined values
        combined_values = omega[i] * Q + (1 - omega[i]) * action_values;

        // Compute log probabilities using softmax
        log_probs = beta[i] * combined_values - log_sum_exp(beta[i] * combined_values);

        // Store log-likelihood for each trial
        trial_count += 1;
        log_lik[trial_count] = log_probs[a];

        // Update Q-values based on own choice and reward
        Q[a] += lr_own[i] * (r - Q[a]);

        // Update action values based on partner's action
        for (k in 1:NC) {
          if (k == pa) {
            action_values[k] += lr_partner_action[i] * (1 - action_values[k]);
          } else {
            action_values[k] += lr_partner_action[i] * (0 - action_values[k]);
          }
        }
      }
    }
  }
}
