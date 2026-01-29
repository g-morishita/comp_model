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
  vector[E] outcome_obs;

  int<lower=0,upper=A> demo_action[E];
  vector[E] demo_outcome_obs;
  int<lower=0,upper=1> has_demo_outcome[E];

  int<lower=1,upper=C> cond[E];

  real pseudo_reward;
  real<lower=1e-6> beta_lower;
  real<lower=1e-6> beta_upper;
  real<lower=0> kappa_abs_max;

  // hyperpriors on z-scale (shared)
  int<lower=1,upper=8> mu_ap_shared_prior_family; real mu_ap_shared_prior_p1; real mu_ap_shared_prior_p2; real mu_ap_shared_prior_p3;
  int<lower=1,upper=8> sd_ap_shared_prior_family; real sd_ap_shared_prior_p1; real sd_ap_shared_prior_p2; real sd_ap_shared_prior_p3;

  int<lower=1,upper=8> mu_ai_shared_prior_family; real mu_ai_shared_prior_p1; real mu_ai_shared_prior_p2; real mu_ai_shared_prior_p3;
  int<lower=1,upper=8> sd_ai_shared_prior_family; real sd_ai_shared_prior_p1; real sd_ai_shared_prior_p2; real sd_ai_shared_prior_p3;

  int<lower=1,upper=8> mu_b_shared_prior_family;  real mu_b_shared_prior_p1;  real mu_b_shared_prior_p2;  real mu_b_shared_prior_p3;
  int<lower=1,upper=8> sd_b_shared_prior_family;  real sd_b_shared_prior_p1;  real sd_b_shared_prior_p2;  real sd_b_shared_prior_p3;

  int<lower=1,upper=8> mu_k_shared_prior_family;  real mu_k_shared_prior_p1;  real mu_k_shared_prior_p2;  real mu_k_shared_prior_p3;
  int<lower=1,upper=8> sd_k_shared_prior_family;  real sd_k_shared_prior_p1;  real sd_k_shared_prior_p2;  real sd_k_shared_prior_p3;

  // hyperpriors on z-scale (delta, per non-baseline condition)
  int<lower=1,upper=8> mu_ap_delta_prior_family;  real mu_ap_delta_prior_p1;  real mu_ap_delta_prior_p2;  real mu_ap_delta_prior_p3;
  int<lower=1,upper=8> sd_ap_delta_prior_family;  real sd_ap_delta_prior_p1;  real sd_ap_delta_prior_p2;  real sd_ap_delta_prior_p3;

  int<lower=1,upper=8> mu_ai_delta_prior_family;  real mu_ai_delta_prior_p1;  real mu_ai_delta_prior_p2;  real mu_ai_delta_prior_p3;
  int<lower=1,upper=8> sd_ai_delta_prior_family;  real sd_ai_delta_prior_p1;  real sd_ai_delta_prior_p2;  real sd_ai_delta_prior_p3;

  int<lower=1,upper=8> mu_b_delta_prior_family;   real mu_b_delta_prior_p1;   real mu_b_delta_prior_p2;   real mu_b_delta_prior_p3;
  int<lower=1,upper=8> sd_b_delta_prior_family;   real sd_b_delta_prior_p1;   real sd_b_delta_prior_p2;   real sd_b_delta_prior_p3;

  int<lower=1,upper=8> mu_k_delta_prior_family;   real mu_k_delta_prior_p1;   real mu_k_delta_prior_p2;   real mu_k_delta_prior_p3;
  int<lower=1,upper=8> sd_k_delta_prior_family;   real sd_k_delta_prior_p1;   real sd_k_delta_prior_p2;   real sd_k_delta_prior_p3;
}
parameters {
  // shared (non-centered)
  real mu_ap_shared; real<lower=0> sd_ap_shared; vector[N] z_ap_shared;
  real mu_ai_shared; real<lower=0> sd_ai_shared; vector[N] z_ai_shared;
  real mu_b_shared;  real<lower=0> sd_b_shared;  vector[N] z_b_shared;
  real mu_k_shared;  real<lower=0> sd_k_shared;  vector[N] z_k_shared;

  // deltas for non-baseline conditions
  vector[C-1] mu_ap_delta; vector<lower=0>[C-1] sd_ap_delta; matrix[N, C-1] z_ap_delta;
  vector[C-1] mu_ai_delta; vector<lower=0>[C-1] sd_ai_delta; matrix[N, C-1] z_ai_delta;
  vector[C-1] mu_b_delta;  vector<lower=0>[C-1] sd_b_delta;  matrix[N, C-1] z_b_delta;
  vector[C-1] mu_k_delta;  vector<lower=0>[C-1] sd_k_delta;  matrix[N, C-1] z_k_delta;
}
transformed parameters {
  vector[N] ap_shared_z = mu_ap_shared + sd_ap_shared * z_ap_shared;
  vector[N] ai_shared_z = mu_ai_shared + sd_ai_shared * z_ai_shared;
  vector[N] b_shared_z  = mu_b_shared  + sd_b_shared  * z_b_shared;
  vector[N] k_shared_z  = mu_k_shared  + sd_k_shared  * z_k_shared;

  matrix[N, C] ap_z;
  matrix[N, C] ai_z;
  matrix[N, C] b_z;
  matrix[N, C] k_z;

  for (n in 1:N) {
    for (c in 1:C) {
      if (c == baseline_cond) {
        ap_z[n,c] = ap_shared_z[n];
        ai_z[n,c] = ai_shared_z[n];
        b_z[n,c]  = b_shared_z[n];
        k_z[n,c]  = k_shared_z[n];
      } else {
        int idx = (c < baseline_cond) ? c : (c - 1);
        ap_z[n,c] = ap_shared_z[n] + (mu_ap_delta[idx] + sd_ap_delta[idx] * z_ap_delta[n, idx]);
        ai_z[n,c] = ai_shared_z[n] + (mu_ai_delta[idx] + sd_ai_delta[idx] * z_ai_delta[n, idx]);
        b_z[n,c]  = b_shared_z[n]  + (mu_b_delta[idx]  + sd_b_delta[idx]  * z_b_delta[n, idx]);
        k_z[n,c]  = k_shared_z[n]  + (mu_k_delta[idx]  + sd_k_delta[idx]  * z_k_delta[n, idx]);
      }
    }
  }

  matrix<lower=0,upper=1>[N, C] alpha_p = inv_logit(ap_z);
  matrix<lower=0,upper=1>[N, C] alpha_i = inv_logit(ai_z);

  matrix<lower=beta_lower,upper=beta_upper>[N, C] beta =
    beta_lower + (beta_upper - beta_lower) .* inv_logit(b_z);

  matrix<lower=-kappa_abs_max,upper=kappa_abs_max>[N, C] kappa =
    kappa_abs_max * (2 * inv_logit(k_z) - 1);
}
model {
  z_ap_shared ~ normal(0,1);
  z_ai_shared ~ normal(0,1);
  z_b_shared  ~ normal(0,1);
  z_k_shared  ~ normal(0,1);

  to_vector(z_ap_delta) ~ normal(0,1);
  to_vector(z_ai_delta) ~ normal(0,1);
  to_vector(z_b_delta)  ~ normal(0,1);
  to_vector(z_k_delta)  ~ normal(0,1);

  // shared hyperpriors
  target += prior_lpdf(mu_ap_shared, mu_ap_shared_prior_family, mu_ap_shared_prior_p1, mu_ap_shared_prior_p2, mu_ap_shared_prior_p3);
  target += prior_lpdf(sd_ap_shared, sd_ap_shared_prior_family, sd_ap_shared_prior_p1, sd_ap_shared_prior_p2, sd_ap_shared_prior_p3);

  target += prior_lpdf(mu_ai_shared, mu_ai_shared_prior_family, mu_ai_shared_prior_p1, mu_ai_shared_prior_p2, mu_ai_shared_prior_p3);
  target += prior_lpdf(sd_ai_shared, sd_ai_shared_prior_family, sd_ai_shared_prior_p1, sd_ai_shared_prior_p2, sd_ai_shared_prior_p3);

  target += prior_lpdf(mu_b_shared,  mu_b_shared_prior_family,  mu_b_shared_prior_p1,  mu_b_shared_prior_p2,  mu_b_shared_prior_p3);
  target += prior_lpdf(sd_b_shared,  sd_b_shared_prior_family,  sd_b_shared_prior_p1,  sd_b_shared_prior_p2,  sd_b_shared_prior_p3);

  target += prior_lpdf(mu_k_shared,  mu_k_shared_prior_family,  mu_k_shared_prior_p1,  mu_k_shared_prior_p2,  mu_k_shared_prior_p3);
  target += prior_lpdf(sd_k_shared,  sd_k_shared_prior_family,  sd_k_shared_prior_p1,  sd_k_shared_prior_p2,  sd_k_shared_prior_p3);

  // delta hyperpriors (per non-baseline condition)
  for (c in 1:(C-1)) {
    target += prior_lpdf(mu_ap_delta[c], mu_ap_delta_prior_family, mu_ap_delta_prior_p1, mu_ap_delta_prior_p2, mu_ap_delta_prior_p3);
    target += prior_lpdf(sd_ap_delta[c], sd_ap_delta_prior_family, sd_ap_delta_prior_p1, sd_ap_delta_prior_p2, sd_ap_delta_prior_p3);

    target += prior_lpdf(mu_ai_delta[c], mu_ai_delta_prior_family, mu_ai_delta_prior_p1, mu_ai_delta_prior_p2, mu_ai_delta_prior_p3);
    target += prior_lpdf(sd_ai_delta[c], sd_ai_delta_prior_family, sd_ai_delta_prior_p1, sd_ai_delta_prior_p2, sd_ai_delta_prior_p3);

    target += prior_lpdf(mu_b_delta[c],  mu_b_delta_prior_family,  mu_b_delta_prior_p1,  mu_b_delta_prior_p2,  mu_b_delta_prior_p3);
    target += prior_lpdf(sd_b_delta[c],  sd_b_delta_prior_family,  sd_b_delta_prior_p1,  sd_b_delta_prior_p2,  sd_b_delta_prior_p3);

    target += prior_lpdf(mu_k_delta[c],  mu_k_delta_prior_family,  mu_k_delta_prior_p1,  mu_k_delta_prior_p2,  mu_k_delta_prior_p3);
    target += prior_lpdf(sd_k_delta[c],  sd_k_delta_prior_family,  sd_k_delta_prior_p1,  sd_k_delta_prior_p2,  sd_k_delta_prior_p3);
  }

  array[N] matrix[S, A] Q;
  array[N] array[S] int last_choice;
  for (n in 1:N) {
    Q[n] = rep_matrix(0.0, S, A);
    last_choice[n] = rep_array(0, S);
  }

  for (e in 1:E) {
    int n = subj[e];
    int s = state[e];
    int c = cond[e];

    if (etype[e] == 1) {
      Q[n] = rep_matrix(0.0, S, A);
      last_choice[n] = rep_array(0, S);

    } else if (etype[e] == 2) {
      if (demo_action[e] > 0) {
        int a = demo_action[e];
        Q[n][s,a] = Q[n][s,a] + alpha_i[n,c] * (pseudo_reward - Q[n][s,a]);
      }

    } else if (etype[e] == 3) {
      if (choice[e] > 0) {
        vector[A] u = to_vector(Q[n][s]');
        if (last_choice[n][s] > 0) u[last_choice[n][s]] += kappa[n,c];
        target += categorical_logit_lpmf(choice[e] | beta[n,c] * u);
      }

    } else if (etype[e] == 4) {
      if (action[e] > 0) {
        int a = action[e];
        real r = outcome_obs[e];
        Q[n][s,a] = Q[n][s,a] + alpha_p[n,c] * (r - Q[n][s,a]);
        last_choice[n][s] = a;
      }
    }
  }
}

generated quantities {
  // subject-condition parameters on natural scale
  matrix[N, C] alpha_p_hat = alpha_p;
  matrix[N, C] alpha_i_hat = alpha_i;
  matrix[N, C] beta_hat = beta;
  matrix[N, C] kappa_hat = kappa;

  // population-level (location) per condition
  vector[C] alpha_p_pop;
  vector[C] alpha_i_pop;
  vector[C] beta_pop;
  vector[C] kappa_pop;

  for (c in 1:C) {
    if (c == baseline_cond) {
      alpha_p_pop[c] = inv_logit(mu_ap_shared);
      alpha_i_pop[c] = inv_logit(mu_ai_shared);
      beta_pop[c] = beta_lower + (beta_upper - beta_lower) * inv_logit(mu_b_shared);
      kappa_pop[c] = kappa_abs_max * (2 * inv_logit(mu_k_shared) - 1);
    } else {
      int idx = (c < baseline_cond) ? c : (c - 1);
      alpha_p_pop[c] = inv_logit(mu_ap_shared + mu_ap_delta[idx]);
      alpha_i_pop[c] = inv_logit(mu_ai_shared + mu_ai_delta[idx]);
      beta_pop[c] = beta_lower + (beta_upper - beta_lower) * inv_logit(mu_b_shared + mu_b_delta[idx]);
      kappa_pop[c] = kappa_abs_max * (2 * inv_logit(mu_k_shared + mu_k_delta[idx]) - 1);
    }
  }

  // expose hyperparameters for convenience
  real mu_ap_shared_hat = mu_ap_shared;
  real sd_ap_shared_hat = sd_ap_shared;
  vector[C-1] mu_ap_delta_hat = mu_ap_delta;
  vector[C-1] sd_ap_delta_hat = sd_ap_delta;

  real mu_ai_shared_hat = mu_ai_shared;
  real sd_ai_shared_hat = sd_ai_shared;
  vector[C-1] mu_ai_delta_hat = mu_ai_delta;
  vector[C-1] sd_ai_delta_hat = sd_ai_delta;

  real mu_b_shared_hat = mu_b_shared;
  real sd_b_shared_hat = sd_b_shared;
  vector[C-1] mu_b_delta_hat = mu_b_delta;
  vector[C-1] sd_b_delta_hat = sd_b_delta;

  real mu_k_shared_hat = mu_k_shared;
  real sd_k_shared_hat = sd_k_shared;
  vector[C-1] mu_k_delta_hat = mu_k_delta;
  vector[C-1] sd_k_delta_hat = sd_k_delta;
}
