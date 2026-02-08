data {
  int<lower=1> N;
  int<lower=1> A;
  int<lower=1> S;
  int<lower=1> E;
  int<lower=1> C;
  int<lower=1, upper=C> baseline_cond;

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

  array[E] int<lower=1, upper=C> cond;

  real pseudo_reward;
  real<lower=1e-6> beta_lower;
  real<lower=1e-6> beta_upper;
  real<lower=0> kappa_abs_max;

  // hyperpriors on z-scale (shared)
  int<lower=1, upper=8> mu_alpha_p_shared_prior_family; real mu_alpha_p_shared_prior_p1; real mu_alpha_p_shared_prior_p2; real mu_alpha_p_shared_prior_p3;
  int<lower=1, upper=8> sd_alpha_p_shared_prior_family; real sd_alpha_p_shared_prior_p1; real sd_alpha_p_shared_prior_p2; real sd_alpha_p_shared_prior_p3;

  int<lower=1, upper=8> mu_alpha_i_shared_prior_family; real mu_alpha_i_shared_prior_p1; real mu_alpha_i_shared_prior_p2; real mu_alpha_i_shared_prior_p3;
  int<lower=1, upper=8> sd_alpha_i_shared_prior_family; real sd_alpha_i_shared_prior_p1; real sd_alpha_i_shared_prior_p2; real sd_alpha_i_shared_prior_p3;

  int<lower=1, upper=8> mu_beta_shared_prior_family;  real mu_beta_shared_prior_p1;  real mu_beta_shared_prior_p2;  real mu_beta_shared_prior_p3;
  int<lower=1, upper=8> sd_beta_shared_prior_family;  real sd_beta_shared_prior_p1;  real sd_beta_shared_prior_p2;  real sd_beta_shared_prior_p3;

  int<lower=1, upper=8> mu_kappa_shared_prior_family;  real mu_kappa_shared_prior_p1;  real mu_kappa_shared_prior_p2;  real mu_kappa_shared_prior_p3;
  int<lower=1, upper=8> sd_kappa_shared_prior_family;  real sd_kappa_shared_prior_p1;  real sd_kappa_shared_prior_p2;  real sd_kappa_shared_prior_p3;

  // hyperpriors on z-scale (delta, per non-baseline condition)
  int<lower=1, upper=8> mu_alpha_p_delta_prior_family;  real mu_alpha_p_delta_prior_p1;  real mu_alpha_p_delta_prior_p2;  real mu_alpha_p_delta_prior_p3;
  int<lower=1, upper=8> sd_alpha_p_delta_prior_family;  real sd_alpha_p_delta_prior_p1;  real sd_alpha_p_delta_prior_p2;  real sd_alpha_p_delta_prior_p3;

  int<lower=1, upper=8> mu_alpha_i_delta_prior_family;  real mu_alpha_i_delta_prior_p1;  real mu_alpha_i_delta_prior_p2;  real mu_alpha_i_delta_prior_p3;
  int<lower=1, upper=8> sd_alpha_i_delta_prior_family;  real sd_alpha_i_delta_prior_p1;  real sd_alpha_i_delta_prior_p2;  real sd_alpha_i_delta_prior_p3;

  int<lower=1, upper=8> mu_beta_delta_prior_family;   real mu_beta_delta_prior_p1;   real mu_beta_delta_prior_p2;   real mu_beta_delta_prior_p3;
  int<lower=1, upper=8> sd_beta_delta_prior_family;   real sd_beta_delta_prior_p1;   real sd_beta_delta_prior_p2;   real sd_beta_delta_prior_p3;

  int<lower=1, upper=8> mu_kappa_delta_prior_family;   real mu_kappa_delta_prior_p1;   real mu_kappa_delta_prior_p2;   real mu_kappa_delta_prior_p3;
  int<lower=1, upper=8> sd_kappa_delta_prior_family;   real sd_kappa_delta_prior_p1;   real sd_kappa_delta_prior_p2;   real sd_kappa_delta_prior_p3;
}
parameters {
  // shared (non-centered)
  real mu_alpha_p_shared; real<lower=0> sd_alpha_p_shared; vector[N] z_alpha_p_shared;
  real mu_alpha_i_shared; real<lower=0> sd_alpha_i_shared; vector[N] z_alpha_i_shared;
  real mu_beta_shared;  real<lower=0> sd_beta_shared;  vector[N] z_beta_shared;
  real mu_kappa_shared;  real<lower=0> sd_kappa_shared;  vector[N] z_kappa_shared;

  // deltas for non-baseline conditions
  vector[C-1] mu_alpha_p_delta; vector<lower=0>[C-1] sd_alpha_p_delta; matrix[N, C-1] z_alpha_p_delta;
  vector[C-1] mu_alpha_i_delta; vector<lower=0>[C-1] sd_alpha_i_delta; matrix[N, C-1] z_alpha_i_delta;
  vector[C-1] mu_beta_delta;  vector<lower=0>[C-1] sd_beta_delta;  matrix[N, C-1] z_beta_delta;
  vector[C-1] mu_kappa_delta;  vector<lower=0>[C-1] sd_kappa_delta;  matrix[N, C-1] z_kappa_delta;
}
transformed parameters {
  vector[N] alpha_p_shared_z = mu_alpha_p_shared + sd_alpha_p_shared * z_alpha_p_shared;
  vector[N] alpha_i_shared_z = mu_alpha_i_shared + sd_alpha_i_shared * z_alpha_i_shared;
  vector[N] beta_shared_z  = mu_beta_shared  + sd_beta_shared  * z_beta_shared;
  vector[N] kappa_shared_z  = mu_kappa_shared  + sd_kappa_shared  * z_kappa_shared;

  matrix[N, C] alpha_p_z;
  matrix[N, C] alpha_i_z;
  matrix[N, C] beta_z;
  matrix[N, C] kappa_z;

  for (n in 1:N) {
    for (c in 1:C) {
      if (c == baseline_cond) {
        alpha_p_z[n, c] = alpha_p_shared_z[n];
        alpha_i_z[n, c] = alpha_i_shared_z[n];
        beta_z[n, c]  = beta_shared_z[n];
        kappa_z[n, c]  = kappa_shared_z[n];
      } else {
        int idx = (c < baseline_cond) ? c : (c - 1);
        alpha_p_z[n, c] = alpha_p_shared_z[n] + (mu_alpha_p_delta[idx] + sd_alpha_p_delta[idx] * z_alpha_p_delta[n, idx]);
        alpha_i_z[n, c] = alpha_i_shared_z[n] + (mu_alpha_i_delta[idx] + sd_alpha_i_delta[idx] * z_alpha_i_delta[n, idx]);
        beta_z[n, c]  = beta_shared_z[n]  + (mu_beta_delta[idx]  + sd_beta_delta[idx]  * z_beta_delta[n, idx]);
        kappa_z[n, c]  = kappa_shared_z[n]  + (mu_kappa_delta[idx]  + sd_kappa_delta[idx]  * z_kappa_delta[n, idx]);
      }
    }
  }

  matrix<lower=0, upper=1>[N, C] alpha_p = inv_logit(alpha_p_z);
  matrix<lower=0, upper=1>[N, C] alpha_i = inv_logit(alpha_i_z);

  matrix<lower=beta_lower, upper=beta_upper>[N, C] beta =
    beta_lower + (beta_upper - beta_lower) .* inv_logit(beta_z);

  matrix<lower=-kappa_abs_max, upper=kappa_abs_max>[N, C] kappa =
    kappa_abs_max * (2 * inv_logit(kappa_z) - 1);
}
model {
  z_alpha_p_shared ~ normal(0, 1);
  z_alpha_i_shared ~ normal(0, 1);
  z_beta_shared  ~ normal(0, 1);
  z_kappa_shared  ~ normal(0, 1);

  to_vector(z_alpha_p_delta) ~ normal(0, 1);
  to_vector(z_alpha_i_delta) ~ normal(0, 1);
  to_vector(z_beta_delta)  ~ normal(0, 1);
  to_vector(z_kappa_delta)  ~ normal(0, 1);

  // shared hyperpriors
  target += prior_lpdf(mu_alpha_p_shared | mu_alpha_p_shared_prior_family, mu_alpha_p_shared_prior_p1, mu_alpha_p_shared_prior_p2, mu_alpha_p_shared_prior_p3);
  target += prior_lpdf(sd_alpha_p_shared | sd_alpha_p_shared_prior_family, sd_alpha_p_shared_prior_p1, sd_alpha_p_shared_prior_p2, sd_alpha_p_shared_prior_p3);

  target += prior_lpdf(mu_alpha_i_shared | mu_alpha_i_shared_prior_family, mu_alpha_i_shared_prior_p1, mu_alpha_i_shared_prior_p2, mu_alpha_i_shared_prior_p3);
  target += prior_lpdf(sd_alpha_i_shared | sd_alpha_i_shared_prior_family, sd_alpha_i_shared_prior_p1, sd_alpha_i_shared_prior_p2, sd_alpha_i_shared_prior_p3);

  target += prior_lpdf(mu_beta_shared  | mu_beta_shared_prior_family,  mu_beta_shared_prior_p1,  mu_beta_shared_prior_p2,  mu_beta_shared_prior_p3);
  target += prior_lpdf(sd_beta_shared  | sd_beta_shared_prior_family,  sd_beta_shared_prior_p1,  sd_beta_shared_prior_p2,  sd_beta_shared_prior_p3);

  target += prior_lpdf(mu_kappa_shared  | mu_kappa_shared_prior_family,  mu_kappa_shared_prior_p1,  mu_kappa_shared_prior_p2,  mu_kappa_shared_prior_p3);
  target += prior_lpdf(sd_kappa_shared  | sd_kappa_shared_prior_family,  sd_kappa_shared_prior_p1,  sd_kappa_shared_prior_p2,  sd_kappa_shared_prior_p3);

  // delta hyperpriors (per non-baseline condition)
  for (cc in 1:(C - 1)) {
    target += prior_lpdf(mu_alpha_p_delta[cc] | mu_alpha_p_delta_prior_family, mu_alpha_p_delta_prior_p1, mu_alpha_p_delta_prior_p2, mu_alpha_p_delta_prior_p3);
    target += prior_lpdf(sd_alpha_p_delta[cc] | sd_alpha_p_delta_prior_family, sd_alpha_p_delta_prior_p1, sd_alpha_p_delta_prior_p2, sd_alpha_p_delta_prior_p3);

    target += prior_lpdf(mu_alpha_i_delta[cc] | mu_alpha_i_delta_prior_family, mu_alpha_i_delta_prior_p1, mu_alpha_i_delta_prior_p2, mu_alpha_i_delta_prior_p3);
    target += prior_lpdf(sd_alpha_i_delta[cc] | sd_alpha_i_delta_prior_family, sd_alpha_i_delta_prior_p1, sd_alpha_i_delta_prior_p2, sd_alpha_i_delta_prior_p3);

    target += prior_lpdf(mu_beta_delta[cc]  | mu_beta_delta_prior_family,  mu_beta_delta_prior_p1,  mu_beta_delta_prior_p2,  mu_beta_delta_prior_p3);
    target += prior_lpdf(sd_beta_delta[cc]  | sd_beta_delta_prior_family,  sd_beta_delta_prior_p1,  sd_beta_delta_prior_p2,  sd_beta_delta_prior_p3);

    target += prior_lpdf(mu_kappa_delta[cc]  | mu_kappa_delta_prior_family,  mu_kappa_delta_prior_p1,  mu_kappa_delta_prior_p2,  mu_kappa_delta_prior_p3);
    target += prior_lpdf(sd_kappa_delta[cc]  | sd_kappa_delta_prior_family,  sd_kappa_delta_prior_p1,  sd_kappa_delta_prior_p2,  sd_kappa_delta_prior_p3);
  }

  array[N] matrix[S, A] Q;
  array[N, S] int last_choice;
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
        Q[n][s, a] = Q[n][s, a] + alpha_i[n, c] * (pseudo_reward - Q[n][s, a]);
      }

    } else if (etype[e] == 3) {
      if (choice[e] > 0) {
        vector[A] u = to_vector(Q[n][s]');
        if (last_choice[n][s] > 0) u[last_choice[n][s]] += kappa[n, c];
        for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
        target += categorical_logit_lpmf(choice[e] | beta[n, c] * u);
      }

    } else if (etype[e] == 4) {
      if (action[e] > 0) {
        int a = action[e];
        real r = outcome_obs[e];
        Q[n][s, a] = Q[n][s, a] + alpha_p[n, c] * (r - Q[n][s, a]);
        last_choice[n][s] = a;
      }
    }
  }
}
generated quantities {
  // z-scale shared summaries (for compatibility with Python wrapper)
  vector[N] alpha_p__shared_z_hat = alpha_p_shared_z;
  vector[N] alpha_i__shared_z_hat = alpha_i_shared_z;
  vector[N] beta__shared_z_hat    = beta_shared_z;
  vector[N] kappa__shared_z_hat   = kappa_shared_z;

  matrix[N, C - 1] alpha_p__delta_z_hat;
  matrix[N, C - 1] alpha_i__delta_z_hat;
  matrix[N, C - 1] beta__delta_z_hat;
  matrix[N, C - 1] kappa__delta_z_hat;

  for (n in 1:N) {
    for (i in 1:(C - 1)) {
      alpha_p__delta_z_hat[n, i] = mu_alpha_p_delta[i] + sd_alpha_p_delta[i] * z_alpha_p_delta[n, i];
      alpha_i__delta_z_hat[n, i] = mu_alpha_i_delta[i] + sd_alpha_i_delta[i] * z_alpha_i_delta[n, i];
      beta__delta_z_hat[n, i]    = mu_beta_delta[i]  + sd_beta_delta[i]  * z_beta_delta[n, i];
      kappa__delta_z_hat[n, i]   = mu_kappa_delta[i]  + sd_kappa_delta[i]  * z_kappa_delta[n, i];
    }
  }

  // subject-condition parameters on natural scale
  matrix[N, C] alpha_p_hat = alpha_p;
  matrix[N, C] alpha_i_hat = alpha_i;
  matrix[N, C] beta_hat    = beta;
  matrix[N, C] kappa_hat   = kappa;

  // population-level (location) per condition
  vector[C] alpha_p_pop;
  vector[C] alpha_i_pop;
  vector[C] beta_pop;
  vector[C] kappa_pop;

  for (c in 1:C) {
    if (c == baseline_cond) {
      alpha_p_pop[c] = inv_logit(mu_alpha_p_shared);
      alpha_i_pop[c] = inv_logit(mu_alpha_i_shared);
      beta_pop[c] = beta_lower + (beta_upper - beta_lower) * inv_logit(mu_beta_shared);
      kappa_pop[c] = kappa_abs_max * (2 * inv_logit(mu_kappa_shared) - 1);
    } else {
      int idx = (c < baseline_cond) ? c : (c - 1);
      alpha_p_pop[c] = inv_logit(mu_alpha_p_shared + mu_alpha_p_delta[idx]);
      alpha_i_pop[c] = inv_logit(mu_alpha_i_shared + mu_alpha_i_delta[idx]);
      beta_pop[c] = beta_lower + (beta_upper - beta_lower) * inv_logit(mu_beta_shared + mu_beta_delta[idx]);
      kappa_pop[c] = kappa_abs_max * (2 * inv_logit(mu_kappa_shared + mu_kappa_delta[idx]) - 1);
    }
  }

  // expose hyperparameters for convenience
  real mu_alpha_p_shared_hat = mu_alpha_p_shared;
  real sd_alpha_p_shared_hat = sd_alpha_p_shared;
  vector[C - 1] mu_alpha_p_delta_hat = mu_alpha_p_delta;
  vector[C - 1] sd_alpha_p_delta_hat = sd_alpha_p_delta;

  real mu_alpha_i_shared_hat = mu_alpha_i_shared;
  real sd_alpha_i_shared_hat = sd_alpha_i_shared;
  vector[C - 1] mu_alpha_i_delta_hat = mu_alpha_i_delta;
  vector[C - 1] sd_alpha_i_delta_hat = sd_alpha_i_delta;

  real mu_beta_shared_hat = mu_beta_shared;
  real sd_beta_shared_hat = sd_beta_shared;
  vector[C - 1] mu_beta_delta_hat = mu_beta_delta;
  vector[C - 1] sd_beta_delta_hat = sd_beta_delta;

  real mu_kappa_shared_hat = mu_kappa_shared;
  real sd_kappa_shared_hat = sd_kappa_shared;
  vector[C - 1] mu_kappa_delta_hat = mu_kappa_delta;
  vector[C - 1] sd_kappa_delta_hat = sd_kappa_delta;
}
