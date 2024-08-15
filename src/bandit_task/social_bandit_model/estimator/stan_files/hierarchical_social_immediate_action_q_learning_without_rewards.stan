data {
  int<lower=1> N; // The number of participants
  int<lower=1> S; // The number of sessions
  int<lower=1> T; // The number of trials
  int<lower=1> NC; // The number of unique choices
  array[N,S,T] int C; // Choices
  array[N,S,T] int PC; // Partner's choice
  array[N,S,T] int PR; // Partner's rewards
}


parameters {
  vector[N] alpha_nd; //
  vector[N] beta_nd; //
  vector[N] s_nd; // stickiness

  // Population level parameters of mean and variance of
  // learning rate and inverse temperature
  real mu_alpha_nd;
  real sigma_alpha_nd;
  real mu_beta_nd;
  real sigma_beta_nd;
  real mu_s_nd;
  real sigma_s_nd;
}

transformed parameters {
  vector<lower=0,upper=1.0>[N] alpha; // transformed learning rate [0, 1]
  vector<lower=0>[N] beta; // transformed inverse temperature [0, inf)
  vector<lower=0,upper=1.0>[N] s; // transformed stickiness parameter [0, 1]

  // Transform the parameters
  alpha = inv_logit(mu_alpha_nd + exp(sigma_alpha_nd) * alpha_nd);
  beta = exp(mu_beta_nd + exp(sigma_beta_nd) * beta_nd);
  s = inv_logit(mu_s_nd + exp(sigma_s_nd) * s_nd);
}

model {
  vector[NC] Q; // Q values

  alpha_nd ~ normal(0,1); // learning rate (before transformation)
  beta_nd ~ normal(0,1); // inverse temperature (before transformation)
  s_nd ~ normal(0,1); // forgetfulness parameter (before transformation)
  mu_alpha_nd ~ normal(0, 1);
  sigma_alpha_nd ~ normal(0, 1);
  mu_beta_nd ~ normal(0, 1);
  sigma_beta_nd ~ normal(0, 1);
  mu_s_nd ~ normal(0, 1);
  sigma_s_nd ~ normal(0, 1);

  for ( i in 1:N ) { // participant
    for (j in 1:S) { // session

      // Initialize Q values
      for (k in 1:NC) {
        Q[k] = 0.5;
      }

      for ( t in 1:T ) { // trial
        // Update Q value according to partner's choice and reward.
        Q[PC[i, j, t]] = Q[PC[i, j, t]] + alpha[i] * (PR[i, j, t] - Q[PC[i, j, t]]) + s;

        // Add the likelihood according to your own choice
        target += log_softmax(beta[i] * Q)[C[i, j, t]];
      }
    }
  }
}

generated quantities {
  real<lower=0,upper=1.0> mu_alpha;
  real<lower=0> mu_beta;
  real<lower=0, upper=1.0> mu_s;

  vector[N * S * T] log_lik;
  int trial_count;
  vector[NC] Q; // Q values

  real eps;
  eps = machine_precision();

  mu_alpha = inv_logit(mu_alpha_nd);
  mu_beta = exp(mu_beta_nd);
  mu_s = inv_logit(mu_s_nd);

  trial_count = 0;
  for ( i in 1:N ) { // participant
    for (j in 1:S) { // session
      // Initialize Q values
      for (k in 1:NC) {
        Q[k] = 0.5;
      }

      for ( t in 1:T ) { // trials
        // Update Q value according to the partner's choice and reward.
        Q[PC[i, j, t]] = Q[PC[i, j, t]] + alpha[i] * (PR[i, j, t] - Q[PC[i, j, t]]) + s;

        // Add the likelihood according to your own choice
        trial_count = trial_count + 1;
        log_lik[trial_count] = log_softmax(beta[i] * Q)[C[i, j, t]];
      }
    }
  }
}