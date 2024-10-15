data {
  int<lower=1> N; // Number of participants
  int<lower=1> S; // Number of sessions per participant
  int<lower=1> T; // Number of trials per session
  int<lower=1> NC; // Number of unique choices
  array[N, S, T] int<lower=1, upper=NC> C; // Choices (your own choices)
  array[N, S, T] int R; // Rewards received
  array[N, S, T] int<lower=1, upper=NC> PC; // Partner's choices
  array[N, S, T] int PR; // Partner's rewards
}

parameters {
  // Individual-level latent parameters (standard normal)
  vector[N] alpha_own_nd;
  vector[N] alpha_partner_nd;
  vector[N] beta_nd;
  vector[N] s_own_nd;
  vector[N] s_partner_nd;

  // Group-level parameters (means and standard deviations)
  real mu_alpha_own_nd;
  real<lower=0> sigma_alpha_own_nd;
  real mu_alpha_partner_nd;
  real<lower=0> sigma_alpha_partner_nd;
  real mu_beta_nd;
  real<lower=0> sigma_beta_nd;
  real mu_s_own;
  real<lower=0> sigma_s_own;
  real mu_s_partner;
  real<lower=0> sigma_s_partner;
}

transformed parameters {
  vector<lower=0, upper=1>[N] alpha_own;
  vector<lower=0, upper=1>[N] alpha_partner;
  vector<lower=0>[N] beta;
  vector[N] s_own;
  vector[N] s_partner;

  // Transform individual-level parameters to appropriate scales
  alpha_own = inv_logit(mu_alpha_own_nd + sigma_alpha_own_nd * alpha_own_nd);
  alpha_partner = inv_logit(mu_alpha_partner_nd + sigma_alpha_partner_nd * alpha_partner_nd);
  beta = exp(mu_beta_nd + sigma_beta_nd * beta_nd);
  s_own = mu_s_own + sigma_s_own * s_own_nd;
  s_partner = mu_s_partner + sigma_s_partner * s_partner_nd;
}

model {
  // Priors for individual-level latent parameters
  alpha_own_nd ~ normal(0, 1);
  alpha_partner_nd ~ normal(0, 1);
  beta_nd ~ normal(0, 1);
  s_own_nd ~ normal(0, 1);
  s_partner_nd ~ normal(0, 1);

  // Priors for group-level parameters
  mu_alpha_own_nd ~ normal(0, 1);
  sigma_alpha_own_nd ~ cauchy(0, 2.5);
  mu_alpha_partner_nd ~ normal(0, 1);
  sigma_alpha_partner_nd ~ cauchy(0, 2.5);
  mu_beta_nd ~ normal(0, 1);
  sigma_beta_nd ~ cauchy(0, 2.5);
  mu_s_own ~ normal(0, 1);
  sigma_s_own ~ cauchy(0, 2.5);
  mu_s_partner ~ normal(0, 1);
  sigma_s_partner ~ cauchy(0, 2.5);

  // Likelihood
  for (i in 1:N) { // Participant loop
    for (j in 1:S) { // Session loop
      vector[NC] Q; // Q-values for each action
      int prev_own_choice = 0; // Initialize previous own choice
      int prev_partner_choice = 0; // Initialize previous partner choice

      // Initialize Q-values to 0.5
      Q = rep_vector(0.5, NC);

      for (t in 1:T) { // Trial loop
        vector[NC] action_values = Q;

        // Add stickiness effects if previous choices are available
        if (prev_own_choice > 0) {
          action_values[prev_own_choice] += s_own[i];
        }
        if (prev_partner_choice > 0) {
          action_values[prev_partner_choice] += s_partner[i];
        }

        // Compute log probabilities using softmax
        vector[NC] log_probs = log_softmax(beta[i] * action_values);

        // Update the target log likelihood with own choice
        target += log_probs[C[i, j, t]];

        // Update Q-values based on own choice and reward
        Q[C[i, j, t]] += alpha_own[i] * (R[i, j, t] - Q[C[i, j, t]]);

        // Update Q-values based on partner's choice and reward
        Q[PC[i, j, t]] += alpha_partner[i] * (PR[i, j, t] - Q[PC[i, j, t]]);

        // Update previous choices
        prev_own_choice = C[i, j, t];
        prev_partner_choice = PC[i, j, t];
      }
    }
  }
}

generated quantities {
  // Transformed group-level parameters for interpretation
  real<lower=0, upper=1> mu_alpha_own = inv_logit(mu_alpha_own_nd);
  real<lower=0, upper=1> mu_alpha_partner = inv_logit(mu_alpha_partner_nd);
  real<lower=0> mu_beta = exp(mu_beta_nd);

  vector[N * S * T] log_lik; // Log-likelihood for each trial
  int trial_count = 0;

  // Compute log-likelihood for each trial
  for (i in 1:N) {
    for (j in 1:S) {
      vector[NC] Q; // Q-values for each action
      int prev_own_choice = 0;
      int prev_partner_choice = 0;

      // Initialize Q-values to 0.5
      Q = rep_vector(0.5, NC);

      for (t in 1:T) {
        vector[NC] action_values = Q;

        // Add stickiness effects if previous choices are available
        if (prev_own_choice > 0) {
          action_values[prev_own_choice] += s_own[i];
        }
        if (prev_partner_choice > 0) {
          action_values[prev_partner_choice] += s_partner[i];
        }

        // Compute log probabilities using softmax
        vector[NC] log_probs = log_softmax(beta[i] * action_values);

        // Store log-likelihood for each trial
        trial_count += 1;
        log_lik[trial_count] = log_probs[C[i, j, t]];

        // Update Q-values based on own choice and reward
        Q[C[i, j, t]] += alpha_own[i] * (R[i, j, t] - Q[C[i, j, t]]);

        // Update Q-values based on partner's choice and reward
        Q[PC[i, j, t]] += alpha_partner[i] * (PR[i, j, t] - Q[PC[i, j, t]]);

        // Update previous choices
        prev_own_choice = C[i, j, t];
        prev_partner_choice = PC[i, j, t];
      }
    }
  }
}
