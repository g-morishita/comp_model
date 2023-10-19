data {
  int<lower=1> N; // The number of participants
  int<lower=1> S; // The number of sessions
  int<lower=1> T; // The number of trials
  int C[N,S,T]; // Choices
  int PC[N,S,T]; // Partner's choice
  int PR[N,S,T]; // Partner's rewards
}

parameters {
  vector[N] alpha_nd; //
  vector[N] beta_nd; //

  // Population level parameters of mean and variance of
  // learning rate and inverse temperature
  real mu_alpha_nd;
  real sigma_alpha_nd;
  real mu_beta_nd;
  real sigma_beta_nd;
}

transformed parameters {
  vector<lower=0,upper=1.0>[N] alpha; // transformed learning rate [0, 1]
  vector<lower=0>[N] beta; // transformed inverse temperature [0, inf)

  // Transform the parameters
  alpha = inv_logit(mu_alpha_nd + exp(sigma_alpha_nd) * alpha_nd);
  beta = exp(mu_beta_nd + exp(sigma_beta_nd) * beta_nd);
}

model {
  matrix[2,1] Q; // Q values

  alpha_nd ~ normal(0,1); // learning rate (before transformation)
  beta_nd ~ normal(0,1); // inverse temperature (before transformation)

  for ( i in 1:N ) { // participant

    for (j in 1:S) { // session

      Q[1, 1] = 0.5; Q[2, 1] = 0.5; // Initialize Q values
      for ( t in 1:T ) { // trial
        // Update Q value according to the partner's choice and reward.
        Q[PC[i, j, t], 1] = Q[PC[i, j, t], 1] + alpha[i] * (PR[i, j, t] - Q[PC[i, j, t], 1]);

        // Add the likelihood according to your own choice
        // 3 - Q[C[t] - 1] means the index of the action that's not chosen.
        target += log(1.0 / (1.0 + exp(-beta[i] * (Q[C[i, j, t], 1] - Q[3 - C[i, j, t], 1]))));
      }
    }
  }
}

generated quantities {
  real<lower=0,upper=1.0> mu_alpha;
  real<lower=0> mu_beta;

  vector[N * S * T] log_lik;
  int trial_count;
  matrix[2,1] Q; // Q values

  real eps;
  eps = machine_precision();

  mu_alpha = inv_logit(mu_alpha_nd);
  mu_beta = exp(mu_beta_nd);

  trial_count = 0;
  for ( i in 1:N ) { // participant

    for (j in 1:S) { // session

      Q[1, 1] = 0.5; Q[2, 1] = 0.5; // Initialize Q values
      for ( t in 1:T ) { // trials

        // update action value
        Q[PC[i,j,t],1] = Q[PC[i,j,t],1] + alpha[i] * (PR[i,j,t] - Q[PC[i,j,t],1]);

        trial_count = trial_count + 1;
        log_lik[trial_count] = log( eps + 1.0/(1.0 + exp(-beta[i] * (Q[C[i,j,t],1] - Q[ 3 - C[i,j,t],1]) )) );
      }
    }
  }
}