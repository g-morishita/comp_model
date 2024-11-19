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
  vector[N] alpha_q_nd;
  vector[N] alpha_action_nd;
  vector[N] weight_q_nd;
  vector[N] beta_nd;

  // Population level parameters of mean and variance of
  // learning rate and inverse temperature
  real mu_alpha_q_nd;
  real sigma_alpha_q_nd;
  real mu_alpha_action_nd;
  real sigma_alpha_action_nd;
  real mu_weight_q_nd;
  real sigma_weight_q_nd;
  real mu_beta_nd;
  real sigma_beta_nd;
}

transformed parameters {
  vector<lower=0,upper=1.0>[N] alpha_q; // transformed learning rate for q values [0, 1]
  vector<lower=0,upper=1.0>[N] alpha_action; // transformed learning rate for action value [0, 1]
  vector<lower=0,upper=1.0>[N] weight_for_q; // transformed weights for q value [0, 1]
  vector<lower=0>[N] beta; // transformed inverse temperature [0, inf)

  // Transform the parameters
  alpha_q = inv_logit(mu_alpha_q_nd + sigma_alpha_q_nd * alpha_q_nd);
  alpha_action = inv_logit(mu_alpha_action_nd + sigma_alpha_action_nd * alpha_action_nd);
  weight_for_q = inv_logit(mu_weight_q_nd + sigma_weight_q_nd * weight_q_nd); // fix typo
  beta = exp(mu_beta_nd + sigma_beta_nd * beta_nd);
}

model {
  vector[NC] Q; // Q values
  vector[NC] A; // Action values
  vector[NC] combined_values; // Combined Q and A values

  alpha_q_nd ~ normal(0, 1); // learning rate for q values (before transformation)
  alpha_action_nd ~ normal(0, 1); // learning rate for action values (before transformation)
  beta_nd ~ normal(0, 1); // inverse temperature (before transformation)
  weight_q_nd ~ normal(0, 1); // weight for q value (before transformation)

  mu_alpha_q_nd ~ normal(0, 1);
  sigma_alpha_q_nd ~ cauchy(0, 3);
  mu_alpha_action_nd ~ normal(0, 1);
  sigma_alpha_action_nd ~ cauchy(0, 3);
  mu_weight_q_nd ~ normal(0, 1);
  sigma_weight_q_nd ~ cauchy(0, 3);
  mu_beta_nd ~ normal(0, 1);
  sigma_beta_nd ~ cauchy(0, 3);

  for (i in 1:N) { // participant
    for (j in 1:S) { // session

      // Initialize Q values
      for (k in 1:NC) {
        Q[k] = 0.5;
      }

      // Initialize action values
      for (k in 1:NC) {
        A[k] = 1.0 / NC;
      }

      for (t in 1:T) { // trial
        // Update Q value according to partner's choice and reward.
        Q[PC[i, j, t]] = Q[PC[i, j, t]] + alpha_q[i] * (PR[i, j, t] - Q[PC[i, j, t]]);

        // Update action value according to the partner's choice.
        for (k in 1:NC) {
          if (PC[i, j, t] == k) {
            A[k] = A[k] + alpha_action[i] * (1 - A[k]);
          } else {
            A[k] = A[k] + alpha_action[i] * (0 - A[k]);
          }
        }

        combined_values = weight_for_q[i] * Q + (1 - weight_for_q[i]) * A;

        // Add the likelihood according to your own choice
        target += log_softmax(beta[i] * combined_values)[C[i, j, t]];
      }
    }
  }
}

generated quantities {
  real<lower=0,upper=1.0> mu_alpha_q;
  real<lower=0,upper=1.0> mu_alpha_action;
  real<lower=0,upper=1.0> mu_weight_for_q;
  real<lower=0> mu_beta;

  vector[N * S * T] log_lik;
  int trial_count;
  vector[NC] Q; // Q values
  vector[NC] A; // Action values
  vector[NC] combined_values; // Combined Q and A values

  real eps = 1e-9; // small constant instead of machine_precision()

  mu_alpha_q = inv_logit(mu_alpha_q_nd);
  mu_alpha_action = inv_logit(mu_alpha_action_nd);
  mu_weight_for_q = inv_logit(mu_weight_q_nd);
  mu_beta = exp(mu_beta_nd);

  trial_count = 0;
  for (i in 1:N) { // participant
    for (j in 1:S) { // session
      // Initialize Q values
      for (k in 1:NC) {
        Q[k] = 0.5;
      }

      for (k in 1:NC) {
        A[k] = 1.0 / NC;
      }

      for (t in 1:T) { // trials
        // Update Q value according to the partner's choice and reward.
        Q[PC[i, j, t]] = Q[PC[i, j, t]] + alpha_q[i] * (PR[i, j, t] - Q[PC[i, j, t]]);

        // Update action value according to the partner's choice.
        for (k in 1:NC) {
          if (PC[i, j, t] == k) {
            A[k] = A[k] + alpha_action[i] * (1 - A[k]);
          } else {
            A[k] = A[k] + alpha_action[i] * (0 - A[k]);
          }
        }

        combined_values = weight_for_q[i] * Q + (1 - weight_for_q[i]) * A;

        // Add the likelihood according to your own choice
        trial_count = trial_count + 1;
        log_lik[trial_count] = log_softmax(beta[i] * combined_values)[C[i, j, t]];
      }
    }
  }
}
