data {
  int<lower=1> N; // The number of participants
  int<lower=1> S; // The number of sessions
  int<lower=1> T; // The number of trials
  int<lower=1> NC; // The number of unique choices
  array[N,S,T] int C; // Choices
  array[N,S,T] int R; // Rewards
  array[N,S,T] int PC; // Partner's choice
  array[N,S,T] int PR; // Partner's rewards
}


parameters {
  vector[N] alpha_own_nd; //
  vector[N] alpha_partner_nd; //
  vector[N] beta_nd; //

  // Population level parameters of mean and variance of
  // learning rate and inverse temperature
  real mu_alpha_own_nd;
  real sigma_alpha_own_nd;
   real mu_alpha_partner_nd;
  real sigma_alpha_partner_nd;
  real mu_beta_nd;
  real sigma_beta_nd;
}

transformed parameters {
  vector<lower=0,upper=1.0>[N] alpha_own; // transformed learning rate for your own experience [0, 1]
  vector<lower=0,upper=1.0>[N] alpha_partner; // transformed learning rate for partner's experience [0, 1]
  vector<lower=0>[N] beta; // transformed inverse temperature [0, inf)

  // Transform the parameters
  alpha_own = inv_logit(mu_alpha_own_nd + exp(sigma_alpha_own_nd) * alpha_own_nd);
  alpha_partner = inv_logit(mu_alpha_partner_nd + exp(sigma_alpha_partner_nd) * alpha_partner_nd);
  beta = exp(mu_beta_nd + exp(sigma_beta_nd) * beta_nd);
}

model {
  vector[NC] Q; // Q values

  alpha_own_nd ~ normal(0,1); // learning rate for your own experience (before transformation)
  alpha_partner_nd ~ normal(0, 1) // learning rate for partner's experience (before transformation)
  beta_nd ~ normal(0,1); // inverse temperature (before transformation)
  mu_alpha_own_nd ~ normal(0, 1);
  sigma_alpha_own_nd ~ normal(0, 1);
   mu_alpha_partner_nd ~ normal(0, 1);
  sigma_alpha_partner_nd ~ normal(0, 1);
  mu_beta_nd ~ normal(0, 1);
  sigma_beta_nd ~ normal(0, 1);

  for ( i in 1:N ) { // participant
    for (j in 1:S) { // session

      // Initialize Q values
      for (k in 1:NC) {
        Q[k] = 0.5;
      }

      for ( t in 1:T ) { // trial
        // Update Q value according to partner's choice and reward.
        Q[PC[i, j, t]] = Q[PC[i, j, t]] + alpha_partner[i] * (PR[i, j, t] - Q[PC[i, j, t]]);

        // Add the likelihood according to your own choice
        target += log_softmax(beta[i] * Q)[C[i, j, t]];

        // Update Q value according to your own choice and reward.
        Q[C[i, j, t]] = Q[C[i, j, t]] + alpha_own[i] * (R[i, j, t] - Q[C[i, j, t]]);
      }
    }
  }
}

generated quantities {
  real<lower=0,upper=1.0> mu_alpha_own;
  real<lower=0,upper=1.0> mu_alpha_partner;
  real<lower=0> mu_beta;

  vector[N * S * T] log_lik;
  int trial_count;
  vector[NC] Q; // Q values

  real eps;
  eps = machine_precision();

  mu_alpha_own = inv_logit(mu_alpha_own_nd);
  mu_alpha_partner = inv_logit(mu_alpha_partner_nd);
  mu_beta = exp(mu_beta_nd);

  trial_count = 0;
  for ( i in 1:N ) { // participant
    for (j in 1:S) { // session
      // Initialize Q values
      for (k in 1:NC) {
        Q[k] = 0.5;
      }

      for ( t in 1:T ) { // trials
        // Update Q value according to the partner's choice and reward.
        Q[PC[i, j, t]] = Q[PC[i, j, t]] + alpha_partner[i] * (PR[i, j, t] - Q[PC[i, j, t]]);

        // Add the likelihood according to your own choice
        trial_count = trial_count + 1;
        log_lik[trial_count] = log_softmax(beta[i] * Q)[C[i, j, t]];

        // Update Q value according to your own choice and reward.
        Q[C[i, j, t]] = Q[C[i, j, t]] + alpha_own[i] * (R[i, j, t] - Q[C[i, j, t]]);
      }
    }
  }
}