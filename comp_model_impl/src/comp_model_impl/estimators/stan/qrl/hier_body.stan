data {
  int<lower=1> N;
  int<lower=1> A;
  int<lower=1> S;
  int<lower=1> E;

  array[E] int<lower=1, upper=N> subj;
  array[E] int<lower=1, upper=4> etype;

  array[E] int<lower=1, upper=S> state;
  array[E] int<lower=0, upper=A> choice;
  array[E] int<lower=0, upper=A> action;
  vector[E] outcome_obs;

  array[E] vector<lower=0,upper=1>[A] avail_mask;

  real<lower=1e-6> beta_lower;

  // hyperpriors
  int<lower=1, upper=8> mu_alpha_prior_family; real mu_alpha_prior_p1; real mu_alpha_prior_p2; real mu_alpha_prior_p3;
  int<lower=1, upper=8> sd_alpha_prior_family; real sd_alpha_prior_p1; real sd_alpha_prior_p2; real sd_alpha_prior_p3;

  int<lower=1, upper=8> mu_beta_prior_family;  real mu_beta_prior_p1;  real mu_beta_prior_p2;  real mu_beta_prior_p3;
  int<lower=1, upper=8> sd_beta_prior_family;  real sd_beta_prior_p1;  real sd_beta_prior_p2;  real sd_beta_prior_p3;
}
parameters {
  real mu_alpha; real<lower=0> sd_alpha; vector[N] z_alpha;
  real mu_beta;  real<lower=0> sd_beta;  vector[N] z_beta;
}
transformed parameters {
  vector<lower=0, upper=1>[N] alpha = inv_logit(mu_alpha + sd_alpha * z_alpha);
  vector<lower=beta_lower>[N] beta =
    beta_lower + exp(mu_beta + sd_beta * z_beta);
}
model {
  z_alpha ~ normal(0, 1);
  z_beta  ~ normal(0, 1);

  target += prior_lpdf(mu_alpha | mu_alpha_prior_family, mu_alpha_prior_p1, mu_alpha_prior_p2, mu_alpha_prior_p3);
  target += prior_lpdf(sd_alpha | sd_alpha_prior_family, sd_alpha_prior_p1, sd_alpha_prior_p2, sd_alpha_prior_p3);

  target += prior_lpdf(mu_beta |  mu_beta_prior_family,  mu_beta_prior_p1,  mu_beta_prior_p2,  mu_beta_prior_p3);
  target += prior_lpdf(sd_beta |  sd_beta_prior_family,  sd_beta_prior_p1,  sd_beta_prior_p2,  sd_beta_prior_p3);

  array[N] matrix[S, A] Q;
  for (n in 1:N) Q[n] = rep_matrix(0.0, S, A);

  for (e in 1:E) {
    int n = subj[e];
    int s = state[e];

    if (etype[e] == 1) {
      Q[n] = rep_matrix(0.0, S, A);

    } else if (etype[e] == 3) {
      if (choice[e] > 0) {
        vector[A] u = to_vector(Q[n][s]');
        for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
        target += categorical_logit_lpmf(choice[e] | beta[n] * u);
      }

    } else if (etype[e] == 4) {
      if (action[e] > 0) {
        int a = action[e];
        real r = outcome_obs[e];
        Q[n][s, a] = Q[n][s, a] + alpha[n] * (r - Q[n][s, a]);
      }
    }
  }
}
generated quantities {
  vector[E] log_lik = rep_vector(0.0, E);
  {
    array[N] matrix[S, A] Q;
    for (n in 1:N) Q[n] = rep_matrix(0.0, S, A);
  
    for (e in 1:E) {
      int n = subj[e];
      int s = state[e];
  
      if (etype[e] == 1) {
        Q[n] = rep_matrix(0.0, S, A);
  
      } else if (etype[e] == 3) {
        if (choice[e] > 0) {
          vector[A] u = to_vector(Q[n][s]');
          for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
          log_lik[e] = categorical_logit_lpmf(choice[e] | beta[n] * u);
        }
  
      } else if (etype[e] == 4) {
        if (action[e] > 0) {
          int a = action[e];
          real r = outcome_obs[e];
          Q[n][s, a] = Q[n][s, a] + alpha[n] * (r - Q[n][s, a]);
        }
      }
    }
  }
  real alpha_pop = inv_logit(mu_alpha);
  real beta_pop =
    beta_lower + exp(mu_beta);

  real mu_alpha_hat = mu_alpha;
  real sd_alpha_hat = sd_alpha;
  real mu_beta_hat = mu_beta;
  real sd_beta_hat = sd_beta;
}
