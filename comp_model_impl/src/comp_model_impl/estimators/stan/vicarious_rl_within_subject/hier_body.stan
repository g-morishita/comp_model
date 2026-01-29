data {
  int<lower=1> N;
  int<lower=1> A;
  int<lower=1> S;
  int<lower=1> E;
  int<lower=1> C;
  int<lower=1,upper=C> baseline_cond;

  int<lower=1,upper=N> subj[E];
  int<lower=1,upper=4> etype[E];

  int<lower=1,upper=S> state[E];
  int<lower=0,upper=A> choice[E];
  int<lower=0,upper=A> action[E];
  vector[E] outcome_obs; // unused

  int<lower=0,upper=A> demo_action[E];
  vector[E] demo_outcome_obs;
  int<lower=0,upper=1> has_demo_outcome[E];

  int<lower=1,upper=C> cond[E];

  real<lower=1e-6> beta_lower;
  real<lower=1e-6> beta_upper;

  // hyperpriors (configurable) -- shared
  int<lower=1,upper=8> mu_ao_shared_prior_family; real mu_ao_shared_prior_p1; real mu_ao_shared_prior_p2; real mu_ao_shared_prior_p3;
  int<lower=1,upper=8> sd_ao_shared_prior_family; real sd_ao_shared_prior_p1; real sd_ao_shared_prior_p2; real sd_ao_shared_prior_p3;

  int<lower=1,upper=8> mu_b_shared_prior_family;  real mu_b_shared_prior_p1;  real mu_b_shared_prior_p2;  real mu_b_shared_prior_p3;
  int<lower=1,upper=8> sd_b_shared_prior_family;  real sd_b_shared_prior_p1;  real sd_b_shared_prior_p2;  real sd_b_shared_prior_p3;

  // hyperpriors (configurable) -- delta (for C-1 non-baseline conditions)
  int<lower=1,upper=8> mu_ao_delta_prior_family; real mu_ao_delta_prior_p1; real mu_ao_delta_prior_p2; real mu_ao_delta_prior_p3;
  int<lower=1,upper=8> sd_ao_delta_prior_family; real sd_ao_delta_prior_p1; real sd_ao_delta_prior_p2; real sd_ao_delta_prior_p3;

  int<lower=1,upper=8> mu_b_delta_prior_family;  real mu_b_delta_prior_p1;  real mu_b_delta_prior_p2;  real mu_b_delta_prior_p3;
  int<lower=1,upper=8> sd_b_delta_prior_family;  real sd_b_delta_prior_p1;  real sd_b_delta_prior_p2;  real sd_b_delta_prior_p3;
}
parameters {
  // non-centered shared
  real mu_ao_shared; real<lower=0> sd_ao_shared; vector[N] z_ao_shared;
  real mu_b_shared;  real<lower=0> sd_b_shared;  vector[N] z_b_shared;

  // non-centered delta (one delta for each non-baseline condition)
  vector[C-1] mu_ao_delta; vector<lower=0>[C-1] sd_ao_delta; matrix[N, C-1] z_ao_delta;
  vector[C-1] mu_b_delta;  vector<lower=0>[C-1] sd_b_delta;  matrix[N, C-1] z_b_delta;
}
transformed parameters {
  vector[N] ao_shared_z = mu_ao_shared + sd_ao_shared * z_ao_shared;
  vector[N] b_shared_z  = mu_b_shared  + sd_b_shared  * z_b_shared;

  matrix[N, C] ao_z;
  matrix[N, C] b_z;

  for (n in 1:N) {
    for (c in 1:C) {
      if (c == baseline_cond) {
        ao_z[n, c] = ao_shared_z[n];
        b_z[n, c]  = b_shared_z[n];
      } else {
        int idx = (c < baseline_cond) ? c : (c - 1);
        ao_z[n, c] = ao_shared_z[n] + (mu_ao_delta[idx] + sd_ao_delta[idx] * z_ao_delta[n, idx]);
        b_z[n, c]  = b_shared_z[n]  + (mu_b_delta[idx]  + sd_b_delta[idx]  * z_b_delta[n, idx]);
      }
    }
  }

  matrix<lower=0,upper=1>[N, C] alpha_o = inv_logit(ao_z);

  matrix<lower=beta_lower,upper=beta_upper>[N, C] beta =
    beta_lower + (beta_upper - beta_lower) * inv_logit(b_z);
}
model {
  z_ao_shared ~ normal(0,1);
  z_b_shared  ~ normal(0,1);
  to_vector(z_ao_delta) ~ normal(0,1);
  to_vector(z_b_delta)  ~ normal(0,1);

  target += prior_lpdf(mu_ao_shared, mu_ao_shared_prior_family, mu_ao_shared_prior_p1, mu_ao_shared_prior_p2, mu_ao_shared_prior_p3);
  target += prior_lpdf(sd_ao_shared, sd_ao_shared_prior_family, sd_ao_shared_prior_p1, sd_ao_shared_prior_p2, sd_ao_shared_prior_p3);

  target += prior_lpdf(mu_b_shared,  mu_b_shared_prior_family,  mu_b_shared_prior_p1,  mu_b_shared_prior_p2,  mu_b_shared_prior_p3);
  target += prior_lpdf(sd_b_shared,  sd_b_shared_prior_family,  sd_b_shared_prior_p1,  sd_b_shared_prior_p2,  sd_b_shared_prior_p3);

  for (i in 1:(C-1)) {
    target += prior_lpdf(mu_ao_delta[i], mu_ao_delta_prior_family, mu_ao_delta_prior_p1, mu_ao_delta_prior_p2, mu_ao_delta_prior_p3);
    target += prior_lpdf(sd_ao_delta[i], sd_ao_delta_prior_family, sd_ao_delta_prior_p1, sd_ao_delta_prior_p2, sd_ao_delta_prior_p3);

    target += prior_lpdf(mu_b_delta[i],  mu_b_delta_prior_family,  mu_b_delta_prior_p1,  mu_b_delta_prior_p2,  mu_b_delta_prior_p3);
    target += prior_lpdf(sd_b_delta[i],  sd_b_delta_prior_family,  sd_b_delta_prior_p1,  sd_b_delta_prior_p2,  sd_b_delta_prior_p3);
  }

  array[N] matrix[S, A] Q;
  for (n in 1:N) {
    Q[n] = rep_matrix(0.0, S, A);
  }

  for (e in 1:E) {
    int n = subj[e];
    int s = state[e];
    int c = cond[e];

    if (etype[e] == 1) {
      Q[n] = rep_matrix(0.0, S, A);

    } else if (etype[e] == 2) {
      if (demo_action[e] > 0 && has_demo_outcome[e] == 1) {
        int a = demo_action[e];
        real r = demo_outcome_obs[e];
        Q[n][s,a] = Q[n][s,a] + alpha_o[n,c] * (r - Q[n][s,a]);
      }

    } else if (etype[e] == 3) {
      if (choice[e] > 0) {
        vector[A] u = to_vector(Q[n][s]');
        target += categorical_logit_lpmf(choice[e] | beta[n,c] * u);
      }
    }
  }
}

generated quantities {
  matrix[N, C] alpha_o_hat = alpha_o;
  matrix[N, C] beta_hat = beta;

  // population level means by condition (shared + delta)
  vector[C] alpha_o_pop;
  vector[C] beta_pop;
  for (c in 1:C) {
    if (c == baseline_cond) {
      alpha_o_pop[c] = inv_logit(mu_ao_shared);
      beta_pop[c] = beta_lower + (beta_upper - beta_lower) * inv_logit(mu_b_shared);
    } else {
      int idx = (c < baseline_cond) ? c : (c - 1);
      alpha_o_pop[c] = inv_logit(mu_ao_shared + mu_ao_delta[idx]);
      beta_pop[c] = beta_lower + (beta_upper - beta_lower) * inv_logit(mu_b_shared + mu_b_delta[idx]);
    }
  }

  // hyperparameters (expose directly)
  real mu_ao_shared_hat = mu_ao_shared;
  real sd_ao_shared_hat = sd_ao_shared;
  real mu_b_shared_hat  = mu_b_shared;
  real sd_b_shared_hat  = sd_b_shared;

  vector[C-1] mu_ao_delta_hat = mu_ao_delta;
  vector[C-1] sd_ao_delta_hat = sd_ao_delta;
  vector[C-1] mu_b_delta_hat  = mu_b_delta;
  vector[C-1] sd_b_delta_hat  = sd_b_delta;
}
