data {
  int<lower=1> N; // Number of participants
  int<lower=1> S; // Number of sessions per participant
  int<lower=1> T; // Number of trials per session
  int<lower=1> NC; // Number of unique choices/actions
  array[N, S, T] int<lower=1, upper=NC> C; // Participant's own choices
  array[N, S, T] int R; // Participant's own rewards
  array[N, S, T] int<lower=1, upper=NC> PC; // Partner's choices
  array[N, S, T] int PR; // Partner's rewards
}
parameters {
  // Individual-level latent parameters (standard normal)
  vector[N] lr_own_nd;
  vector[N] lr_partner_nd;
  vector[N] beta_nd;
  vector[N] coef_info_bonus_nd;
  vector[N] f_own_nd;
  vector[N] f_partner_nd;
  vector[N] s_own_nd;
  vector[N] s_partner_nd;

  // Group-level parameters (means and standard deviations)
  real mu_lr_own_nd;
  real<lower=0> sigma_lr_own_nd;
  real mu_lr_partner_nd;
  real<lower=0> sigma_lr_partner_nd;
  real mu_beta_nd;
  real<lower=0> sigma_beta_nd;
  real mu_coef_info_bonus_nd;
  real<lower=0> sigma_coef_info_bonus_nd;
  real mu_f_own_nd;
  real<lower=0> sigma_f_own_nd;
  real mu_f_partner_nd;
  real<lower=0> sigma_f_partner_nd;
  real mu_s_own_nd;
  real<lower=0> sigma_s_own_nd;
  real mu_s_partner_nd;
  real<lower=0> sigma_s_partner_nd;
}
transformed parameters {
  vector<lower=0, upper=1>[N] lr_own;
  vector<lower=0, upper=1>[N] lr_partner;
  vector<lower=0>[N] beta;
  vector<lower=0, upper=1>[N] coef_info_bonus;
  vector<lower=0, upper=1>[N] f_own;
  vector<lower=0, upper=1>[N] f_partner;
  vector<lower=0, upper=1>[N] s_own;
  vector<lower=0, upper=1>[N] s_partner;

  // Transform individual-level parameters to appropriate scales
  lr_own = inv_logit(mu_lr_own_nd + sigma_lr_own_nd * lr_own_nd);
  lr_partner = inv_logit(mu_lr_partner_nd + sigma_lr_partner_nd * lr_partner_nd);
  beta = exp(mu_beta_nd + sigma_beta_nd * beta_nd);
  coef_info_bonus = inv_logit(mu_coef_info_bonus_nd + sigma_coef_info_bonus_nd * coef_info_bonus_nd);
  f_own = inv_logit(mu_f_own_nd + sigma_f_own_nd * f_own_nd);
  f_partner = inv_logit(mu_f_partner_nd + sigma_f_partner_nd * f_partner_nd);
  s_own = inv_logit(mu_s_own_nd + sigma_s_own_nd * s_own_nd);
  s_partner = inv_logit(mu_s_partner_nd + sigma_s_partner_nd * s_partner_nd);
}
model {
  // Priors for individual-level latent parameters
  lr_own_nd ~ normal(0, 1);
  lr_partner_nd ~ normal(0, 1);
  beta_nd ~ normal(0, 1);
  coef_info_bonus_nd ~ normal(0, 1);
  f_own_nd ~ normal(0, 1);
  f_partner_nd ~ normal(0, 1);
  s_own_nd ~ normal(0, 1);
  s_partner_nd ~ normal(0, 1);

  // Priors for group-level parameters
  mu_lr_own_nd ~ normal(0, 1);
  sigma_lr_own_nd ~ cauchy(0, 2.5);
  mu_lr_partner_nd ~ normal(0, 1);
  sigma_lr_partner_nd ~ cauchy(0, 2.5);
  mu_beta_nd ~ normal(0, 1);
  sigma_beta_nd ~ cauchy(0, 2.5);
  mu_coef_info_bonus_nd ~ normal(0, 1);
  sigma_coef_info_bonus_nd ~ cauchy(0, 2.5);
  mu_f_own_nd ~ normal(0, 1);
  sigma_f_own_nd ~ cauchy(0, 2.5);
  mu_f_partner_nd ~ normal(0, 1);
  sigma_f_partner_nd ~ cauchy(0, 2.5);
  mu_s_own_nd ~ normal(0, 1);
  sigma_s_own_nd ~ cauchy(0, 2.5);
  mu_s_partner_nd ~ normal(0, 1);
  sigma_s_partner_nd ~ cauchy(0, 2.5);

  // Likelihood
  for (i in 1:N) { // Participant loop
    for (j in 1:S) { // Session loop
      vector[NC] Q; // Q-values for each action
      vector[NC] initial_values; // Initial Q-values
      vector[NC] n_chosen; // Number of times each action has been chosen
      int prev_own_choice = 0; // Initialize previous own choice
      int prev_partner_choice = 0; // Initialize previous partner choice

      // Initialize Q-values and counts
      Q = rep_vector(0.5, NC);
      initial_values = rep_vector(0.5, NC);
      n_chosen = rep_vector(1e-3, NC); // Small value to prevent division by zero

      for (t in 1:T) { // Trial loop
        vector[NC] action_values;
        vector[NC] info_bonus;
        vector[NC] log_probs;

        // Compute the information bonus
        info_bonus = coef_info_bonus[i] ./ sqrt(n_chosen);

        // Calculate action values
        action_values = Q + info_bonus;

        // Add stickiness effects
        if (prev_own_choice > 0) {
          action_values[prev_own_choice] += s_own[i];
        }
        if (prev_partner_choice > 0) {
          action_values[prev_partner_choice] += s_partner[i];
        }

        // Compute log probabilities using softmax
        log_probs = log_softmax(beta[i] * action_values);

        // Update the target log likelihood with own choice
        target += log_probs[C[i, j, t]];

        // Update counts
        n_chosen[C[i, j, t]] += 1;
        n_chosen[PC[i, j, t]] += 1;

        // Update previous choices
        prev_own_choice = C[i, j, t];
        prev_partner_choice = PC[i, j, t];

        // Copy Q-values for updates
        vector[NC] Q_next = Q;

        // Update Q-values for own choice
        Q_next[C[i, j, t]] += lr_own[i] * (R[i, j, t] - Q[C[i, j, t]]);

        // Apply forgetting to other actions after own choice
        for (k in 1:NC) {
          if (k != C[i, j, t]) {
            Q_next[k] = f_own[i] * initial_values[k] + (1 - f_own[i]) * Q_next[k];
          }
        }

        // Update Q-values for partner's choice
        Q_next[PC[i, j, t]] += lr_partner[i] * (PR[i, j, t] - Q_next[PC[i, j, t]]);

        // Apply forgetting to other actions after partner's choice
        for (k in 1:NC) {
          if (k != PC[i, j, t]) {
            Q_next[k] = f_partner[i] * initial_values[k] + (1 - f_partner[i]) * Q_next[k];
          }
        }

        // Update Q-values for the next trial
        Q = Q_next;
      }
    }
  }
}
generated quantities {
  // Transformed group-level parameters for interpretation
  real<lower=0, upper=1> mu_lr_own = inv_logit(mu_lr_own_nd);
  real<lower=0, upper=1> mu_lr_partner = inv_logit(mu_lr_partner_nd);
  real<lower=0> mu_beta = exp(mu_beta_nd);
  real<lower=0, upper=1> mu_coef_info_bonus = inv_logit(mu_coef_info_bonus_nd);
  real<lower=0, upper=1> mu_f_own = inv_logit(mu_f_own_nd);
  real<lower=0, upper=1> mu_f_partner = inv_logit(mu_f_partner_nd);
  real<lower=0, upper=1> mu_s_own = inv_logit(mu_s_own_nd);
  real<lower=0, upper=1> mu_s_partner = inv_logit(mu_s_partner_nd);

  vector[N * S * T] log_lik; // Log-likelihood for each trial
  int trial_count = 0;

  // Compute log-likelihood for each trial
  for (i in 1:N) {
    for (j in 1:S) {
      vector[NC] Q; // Q-values for each action
      vector[NC] initial_values;
      vector[NC] n_chosen;
      int prev_own_choice = 0;
      int prev_partner_choice = 0;

      // Initialize Q-values and counts
      Q = rep_vector(0.5, NC);
      initial_values = rep_vector(0.5, NC);
      n_chosen = rep_vector(1e-3, NC);

      for (t in 1:T) {
        vector[NC] action_values;
        vector[NC] info_bonus;
        vector[NC] log_probs;

        // Compute the information bonus
        info_bonus = coef_info_bonus[i] ./ sqrt(n_chosen);

        // Calculate action values
        action_values = Q + info_bonus;

        // Add stickiness effects
        if (prev_own_choice > 0) {
          action_values[prev_own_choice] += s_own[i];
        }
        if (prev_partner_choice > 0) {
          action_values[prev_partner_choice] += s_partner[i];
        }

        // Compute log probabilities using softmax
        log_probs = log_softmax(beta[i] * action_values);

        // Store log-likelihood for each trial
        trial_count += 1;
        log_lik[trial_count] = log_probs[C[i, j, t]];

        // Update counts
        n_chosen[C[i, j, t]] += 1;
        n_chosen[PC[i, j, t]] += 1;

        // Update previous choices
        prev_own_choice = C[i, j, t];
        prev_partner_choice = PC[i, j, t];

        // Copy Q-values for updates
        vector[NC] Q_next = Q;

        // Update Q-values for own choice
        Q_next[C[i, j, t]] += lr_own[i] * (R[i, j, t] - Q[C[i, j, t]]);

        // Apply forgetting to other actions after own choice
        for (k in 1:NC) {
          if (k != C[i, j, t]) {
            Q_next[k] = f_own[i] * initial_values[k] + (1 - f_own[i]) * Q_next[k];
          }
        }

        // Update Q-values for partner's choice
        Q_next[PC[i, j, t]] += lr_partner[i] * (PR[i, j, t] - Q_next[PC[i, j, t]]);

        // Apply forgetting to other actions after partner's choice
        for (k in 1:NC) {
          if (k != PC[i, j, t]) {
            Q_next[k] = f_partner[i] * initial_values[k] + (1 - f_partner[i]) * Q_next[k];
          }
        }

        // Update Q-values for the next trial
        Q = Q_next;
      }
    }
  }
}
