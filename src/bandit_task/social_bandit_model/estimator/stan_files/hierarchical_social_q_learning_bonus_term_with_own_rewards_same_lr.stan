data {
  int<lower=1> N; // The number of participants
  int<lower=1> S; // The number of sessions
  int<lower=1> T; // The number of trials
  int<lower=1> NC; // The number of unique choices
  array[N, S, T] int C; // Choices
  array[N, S, T] int R; // Rewards
  array[N, S, T] int PC; // Partner's choice
  array[N, S, T] int PR; // Partner's rewards
}

parameters {
  vector[N] alpha_nd;
  vector[N] bonus_coef_nd;
  vector[N] beta_nd;

  // Population level parameters of mean and variance of
  // learning rate and inverse temperature
  real mu_alpha_nd;
  real sigma_alpha_nd;
  real mu_bonus_coef_nd;
  real sigma_bonus_coef_nd;
  real mu_beta_nd;
  real sigma_beta_nd;
}

transformed parameters {
  vector<lower=0, upper=1.0>[N] alpha;
  vector<lower=0>[N] bonus_coef;
  vector<lower=0>[N] beta;

  // Transform the parameters
  alpha = inv_logit(mu_alpha_nd + exp(sigma_alpha_nd) * alpha_nd);
  bonus_coef = exp(mu_bonus_coef_nd + exp(sigma_bonus_coef_nd) * bonus_coef_nd);
  beta = exp(mu_beta_nd + exp(sigma_beta_nd) * beta_nd);
}

model {
  vector[NC] Q; // Q values
  vector[NC] n_choices; // the number of chosen choices

  alpha_nd ~ normal(0, 1); // learning rate for your own experience (before transformation)
  bonus_coef_nd ~ normal(0, 1); // the bonus coefficient (before transformation)
  beta_nd ~ normal(0, 1); // inverse temperature (before transformation)
  mu_alpha_nd ~ normal(0, 1);
  sigma_alpha_nd ~ normal(0, 1);
  mu_bonus_coef_nd ~ normal(0, 1);
  sigma_bonus_coef_nd ~ normal(0, 1);
  mu_beta_nd ~ normal(0, 1);
  sigma_beta_nd ~ normal(0, 1);

  for (i in 1:N) { // participant
    for (j in 1:S) { // session

      // Initialize Q values
      for (k in 1:NC) {
        Q[k] = 0.5;
        n_choices[k] = 1;
      }

      for (t in 1:T) { // trial
        // Add the likelihood according to your own choice
        target += log_softmax(beta[i] * (Q + bonus_coef[i] / sqrt(n_choices)))[C[i, j, t]];

        // Update Q value according to your own choice and reward.
        Q[C[i, j, t]] = Q[C[i, j, t]] + alpha[i] * (R[i, j, t] - Q[C[i, j, t]]);
        n_choices[C[i, j, t]] = 1 + n_choices[C[i, j, t]];

        // Update Q value according to partner's choice and reward.
        Q[PC[i, j, t]] = Q[PC[i, j, t]] + alpha[i] * (PR[i, j, t] - Q[PC[i, j, t]]);
        n_choices[PC[i, j, t]] = 1 + n_choices[PC[i, j, t]];
      }
    }
  }
}

generated quantities {
  real<lower=0, upper=1.0> mu_alpha;
  real<lower=0> mu_bonus_coef;
  real<lower=0> mu_beta;

  vector[N * S * T] log_lik;
  int trial_count;
  vector[NC] Q; // Q values
  vector[NC] n_choices; // the number of chosen choices

  real eps;
  eps = machine_precision();

  mu_alpha = inv_logit(mu_alpha_nd);
  mu_bonus_coef = exp(mu_bonus_coef_nd);
  mu_beta = exp(mu_beta_nd);

  trial_count = 0;
  for (i in 1:N) { // participant
    for (j in 1:S) { // session
      // Initialize Q values
      for (k in 1:NC) {
        Q[k] = 0.5;
        n_choices[k] = 1;
      }

      for (t in 1:T) { // trials
        // Add the likelihood according to your own choice
        trial_count = trial_count + 1;
        log_lik[trial_count] = log_softmax(beta[i] * (Q + bonus_coef[i] / sqrt(n_choices)))[C[i, j, t]];

        // Update Q value according to your own choice and reward.
        Q[C[i, j, t]] = Q[C[i, j, t]] + alpha[i] * (R[i, j, t] - Q[C[i, j, t]]);
        n_choices[C[i, j, t]] = 1 + n_choices[C[i, j, t]];

        // Update Q value according to the partner's choice and reward.
        Q[PC[i, j, t]] = Q[PC[i, j, t]] + alpha[i] * (PR[i, j, t] - Q[PC[i, j, t]]);
        n_choices[PC[i, j, t]] = 1 + n_choices[PC[i, j, t]];
      }
    }
  }
}
