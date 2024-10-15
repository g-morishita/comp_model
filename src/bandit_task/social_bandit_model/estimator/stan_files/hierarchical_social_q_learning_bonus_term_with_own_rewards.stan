data {
  int<lower=1> N; // The number of participants
  int<lower=1> S; // The number of sessions
  int<lower=1> T; // The number of trials
  int<lower=1> NC; // The number of unique choices
  array[N, S, T] int<lower=1, upper=NC> C; // Choices
  array[N, S, T] int R; // Rewards
  array[N, S, T] int<lower=1, upper=NC> PC; // Partner's choices
  array[N, S, T] int PR; // Partner's rewards
}

parameters {
  vector[N] alpha_own_nd;
  vector[N] alpha_partner_nd;
  vector[N] bonus_coef_nd;
  vector[N] beta_nd;

  // Population level parameters
  real mu_alpha_own_nd;
  real<lower=0> sigma_alpha_own_nd;
  real mu_alpha_partner_nd;
  real<lower=0> sigma_alpha_partner_nd;
  real mu_bonus_coef_nd;
  real<lower=0> sigma_bonus_coef_nd;
  real mu_beta_nd;
  real<lower=0> sigma_beta_nd;
}

transformed parameters {
  vector<lower=0,upper=1>[N] alpha_own;
  vector<lower=0,upper=1>[N] alpha_partner;
  vector<lower=0>[N] bonus_coef;
  vector<lower=0>[N] beta;

  // Transform the parameters
  alpha_own = inv_logit(mu_alpha_own_nd + sigma_alpha_own_nd * alpha_own_nd);
  alpha_partner = inv_logit(mu_alpha_partner_nd + sigma_alpha_partner_nd * alpha_partner_nd);
  bonus_coef = exp(mu_bonus_coef_nd + sigma_bonus_coef_nd * bonus_coef_nd);
  beta = exp(mu_beta_nd + sigma_beta_nd * beta_nd);
}

model {
  vector[NC] Q; // Q values
  vector[NC] n_choices; // the number of chosen choices

  // Priors for individual-level parameters
  alpha_own_nd ~ normal(0, 1);
  alpha_partner_nd ~ normal(0, 1);
  bonus_coef_nd ~ normal(0, 1);
  beta_nd ~ normal(0, 1);

  // Priors for group-level parameters
  mu_alpha_own_nd ~ normal(0, 1);
  sigma_alpha_own_nd ~ cauchy(0, 2.5);
  mu_alpha_partner_nd ~ normal(0, 1);
  sigma_alpha_partner_nd ~ cauchy(0, 2.5);
  mu_bonus_coef_nd ~ normal(0, 1);
  sigma_bonus_coef_nd ~ cauchy(0, 2.5);
  mu_beta_nd ~ normal(0, 1);
  sigma_beta_nd ~ cauchy(0, 2.5);

  for (i in 1:N) { // participant
    for (j in 1:S) { // session

      // Initialize Q values and n_choices
      Q = rep_vector(0.5, NC);
      n_choices = rep_vector(1e-3, NC); // Small positive value to prevent division by zero

      for (t in 1:T) { // trial
        // Compute action values with information bonus
        vector[NC] action_values = Q + bonus_coef[i] ./ sqrt(n_choices);

        // Compute log probabilities using softmax
        vector[NC] log_probs = log_softmax(beta[i] * action_values);

        // Update the target log likelihood
        target += log_probs[C[i, j, t]];

        // Update Q value for own choice
        Q[C[i, j, t]] += alpha_own[i] * (R[i, j, t] - Q[C[i, j, t]]);
        n_choices[C[i, j, t]] += 1;

        // Update Q value for partner's choice
        Q[PC[i, j, t]] += alpha_partner[i] * (PR[i, j, t] - Q[PC[i, j, t]]);
        n_choices[PC[i, j, t]] += 1;
      }
    }
  }
}

generated quantities {
  real<lower=0,upper=1> mu_alpha_own;
  real<lower=0,upper=1> mu_alpha_partner;
  real<lower=0> mu_bonus_coef;
  real<lower=0> mu_beta;

  vector[N * S * T] log_lik;
  int trial_count = 0;
  vector[NC] Q; // Q values
  vector[NC] n_choices; // the number of chosen choices

  // Transform group-level means to original scales
  mu_alpha_own = inv_logit(mu_alpha_own_nd);
  mu_alpha_partner = inv_logit(mu_alpha_partner_nd);
  mu_bonus_coef = exp(mu_bonus_coef_nd);
  mu_beta = exp(mu_beta_nd);

  for (i in 1:N) { // participant
    for (j in 1:S) { // session
      // Initialize Q values and n_choices
      Q = rep_vector(0.5, NC);
      n_choices = rep_vector(1e-3, NC);

      for (t in 1:T) { // trials
        // Compute action values with information bonus
        vector[NC] action_values = Q + bonus_coef[i] ./ sqrt(n_choices);

        // Compute log probabilities using softmax
        vector[NC] log_probs = log_softmax(beta[i] * action_values);

        // Store the log likelihood for each trial
        trial_count += 1;
        log_lik[trial_count] = log_probs[C[i, j, t]];

        // Update Q value for own choice
        Q[C[i, j, t]] += alpha_own[i] * (R[i, j, t] - Q[C[i, j, t]]);
        n_choices[C[i, j, t]] += 1;

        // Update Q value for partner's choice
        Q[PC[i, j, t]] += alpha_partner[i] * (PR[i, j, t] - Q[PC[i, j, t]]);
        n_choices[PC[i, j, t]] += 1;
      }
    }
  }
}
