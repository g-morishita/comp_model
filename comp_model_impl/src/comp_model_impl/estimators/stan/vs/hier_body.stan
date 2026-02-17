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

  array[E] int<lower=0, upper=A> demo_action;
  vector[E] demo_outcome_obs;
  array[E] int<lower=0, upper=1> has_demo_outcome;

  real pseudo_reward;
  real<lower=1e-6> beta_lower;
  real<lower=0> kappa_abs_max;

  // hyperpriors (configurable)
  int<lower=1,upper=8> mu_alpha_p_prior_family; real mu_alpha_p_prior_p1; real mu_alpha_p_prior_p2; real mu_alpha_p_prior_p3;
  int<lower=1,upper=8> sd_alpha_p_prior_family; real sd_alpha_p_prior_p1; real sd_alpha_p_prior_p2; real sd_alpha_p_prior_p3;

  int<lower=1,upper=8> mu_alpha_i_prior_family; real mu_alpha_i_prior_p1; real mu_alpha_i_prior_p2; real mu_alpha_i_prior_p3;
  int<lower=1,upper=8> sd_alpha_i_prior_family; real sd_alpha_i_prior_p1; real sd_alpha_i_prior_p2; real sd_alpha_i_prior_p3;

  int<lower=1,upper=8> mu_beta_prior_family;  real mu_beta_prior_p1;  real mu_beta_prior_p2;  real mu_beta_prior_p3;
  int<lower=1,upper=8> sd_beta_prior_family;  real sd_beta_prior_p1;  real sd_beta_prior_p2;  real sd_beta_prior_p3;

  int<lower=1,upper=8> mu_kappa_prior_family;  real mu_kappa_prior_p1;  real mu_kappa_prior_p2;  real mu_kappa_prior_p3;
  int<lower=1,upper=8> sd_kappa_prior_family;  real sd_kappa_prior_p1;  real sd_kappa_prior_p2;  real sd_kappa_prior_p3;
}

parameters {
  // non-centered
  real mu_alpha_p; real<lower=0> sd_alpha_p; vector[N] z_alpha_p;
  real mu_alpha_i; real<lower=0> sd_alpha_i; vector[N] z_alpha_i;
  real mu_beta;  real<lower=0> sd_beta;  vector[N] z_beta;
  real mu_kappa;  real<lower=0> sd_kappa;  vector[N] z_kappa;
}

transformed parameters {
  vector<lower=0,upper=1>[N] alpha_p = inv_logit(mu_alpha_p + sd_alpha_p * z_alpha_p);
  vector<lower=0,upper=1>[N] alpha_i = inv_logit(mu_alpha_i + sd_alpha_i * z_alpha_i);

  vector<lower=beta_lower>[N] beta =
    beta_lower + exp(mu_beta + sd_beta * z_beta);

  vector<lower=-kappa_abs_max,upper=kappa_abs_max>[N] kappa =
    kappa_abs_max * (2 * inv_logit(mu_kappa + sd_kappa * z_kappa) - 1);
}

model {
  z_alpha_p ~ normal(0,1);
  z_alpha_i ~ normal(0,1);
  z_beta  ~ normal(0,1);
  z_kappa  ~ normal(0,1);

  target += prior_lpdf(mu_alpha_p | mu_alpha_p_prior_family, mu_alpha_p_prior_p1, mu_alpha_p_prior_p2, mu_alpha_p_prior_p3);
  target += prior_lpdf(sd_alpha_p | sd_alpha_p_prior_family, sd_alpha_p_prior_p1, sd_alpha_p_prior_p2, sd_alpha_p_prior_p3);

  target += prior_lpdf(mu_alpha_i | mu_alpha_i_prior_family, mu_alpha_i_prior_p1, mu_alpha_i_prior_p2, mu_alpha_i_prior_p3);
  target += prior_lpdf(sd_alpha_i | sd_alpha_i_prior_family, sd_alpha_i_prior_p1, sd_alpha_i_prior_p2, sd_alpha_i_prior_p3);

  target += prior_lpdf(mu_beta | mu_beta_prior_family,  mu_beta_prior_p1,  mu_beta_prior_p2,  mu_beta_prior_p3);
  target += prior_lpdf(sd_beta |  sd_beta_prior_family,  sd_beta_prior_p1,  sd_beta_prior_p2,  sd_beta_prior_p3);

  target += prior_lpdf(mu_kappa |  mu_kappa_prior_family,  mu_kappa_prior_p1,  mu_kappa_prior_p2,  mu_kappa_prior_p3);
  target += prior_lpdf(sd_kappa |  sd_kappa_prior_family,  sd_kappa_prior_p1,  sd_kappa_prior_p2,  sd_kappa_prior_p3);

  array[N] matrix[S, A] Q;
  array[N, S] int last_choice;

  for (n in 1:N) {
    Q[n] = rep_matrix(0.0, S, A);
    for (s in 1:S) last_choice[n, s] = 0;
  }

  for (e in 1:E) {
    int n = subj[e];
    int s = state[e];

    if (etype[e] == 1) {
      Q[n] = rep_matrix(0.0, S, A);
      for (s2 in 1:S) last_choice[n, s2] = 0;

    } else if (etype[e] == 2) {
      if (demo_action[e] > 0) {
        int a = demo_action[e];
        Q[n][s,a] = Q[n][s,a] + alpha_i[n] * (pseudo_reward - Q[n][s,a]);
      }

    } else if (etype[e] == 3) {
      if (choice[e] > 0) {
        vector[A] u = to_vector(Q[n][s]');
        if (last_choice[n, s] > 0) u[last_choice[n, s]] += kappa[n];
        for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
        target += categorical_logit_lpmf(choice[e] | beta[n] * u);
      }

    } else if (etype[e] == 4) {
      if (action[e] > 0) {
        int a = action[e];
        real r = outcome_obs[e];
        Q[n][s,a] = Q[n][s,a] + alpha_p[n] * (r - Q[n][s,a]);
        last_choice[n, s] = a;
      }
    }
  }
}

generated quantities {
  vector[E] log_lik = rep_vector(0.0, E);
  {
    array[N] matrix[S, A] Q;
    array[N, S] int last_choice;
  
    for (n in 1:N) {
      Q[n] = rep_matrix(0.0, S, A);
      for (s in 1:S) last_choice[n, s] = 0;
    }
  
    for (e in 1:E) {
      int n = subj[e];
      int s = state[e];
  
      if (etype[e] == 1) {
        Q[n] = rep_matrix(0.0, S, A);
        for (s2 in 1:S) last_choice[n, s2] = 0;
  
      } else if (etype[e] == 2) {
        if (demo_action[e] > 0) {
          int a = demo_action[e];
          Q[n][s,a] = Q[n][s,a] + alpha_i[n] * (pseudo_reward - Q[n][s,a]);
        }
  
      } else if (etype[e] == 3) {
        if (choice[e] > 0) {
          vector[A] u = to_vector(Q[n][s]');
          if (last_choice[n, s] > 0) u[last_choice[n, s]] += kappa[n];
          for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
          log_lik[e] = categorical_logit_lpmf(choice[e] | beta[n] * u);
        }
  
      } else if (etype[e] == 4) {
        if (action[e] > 0) {
          int a = action[e];
          real r = outcome_obs[e];
          Q[n][s,a] = Q[n][s,a] + alpha_p[n] * (r - Q[n][s,a]);
          last_choice[n, s] = a;
        }
      }
    }
  }
  // "Population-level location" on natural parameter scale
  real alpha_p_pop = inv_logit(mu_alpha_p);
  real alpha_i_pop = inv_logit(mu_alpha_i);

  real beta_pop =
    beta_lower + exp(mu_beta);

  real kappa_pop =
    kappa_abs_max * (2 * inv_logit(mu_kappa) - 1);

  // Expose hyperparameters directly
  real mu_alpha_p_hat = mu_alpha_p;
  real sd_alpha_p_hat = sd_alpha_p;
  real mu_alpha_i_hat = mu_alpha_i;
  real sd_alpha_i_hat = sd_alpha_i;
  real mu_beta_hat  = mu_beta;
  real sd_beta_hat  = sd_beta;
  real mu_kappa_hat  = mu_kappa;
  real sd_kappa_hat  = sd_kappa;
}
