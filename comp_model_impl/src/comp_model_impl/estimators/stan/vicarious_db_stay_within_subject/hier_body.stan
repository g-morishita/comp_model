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

  real<lower=1e-6> beta_lower;
  real<lower=1e-6> beta_upper;
  real<lower=0> kappa_abs_max;
  real<lower=0> demo_bias_abs_max;

  // hyperpriors on z-scale (shared)
  int<lower=1, upper=8> mu_alpha_o__shared_prior_family; real mu_alpha_o__shared_prior_p1; real mu_alpha_o__shared_prior_p2; real mu_alpha_o__shared_prior_p3;
  int<lower=1, upper=8> sd_alpha_o__shared_prior_family; real sd_alpha_o__shared_prior_p1; real sd_alpha_o__shared_prior_p2; real sd_alpha_o__shared_prior_p3;

  int<lower=1, upper=8> mu_demo_bias__shared_prior_family; real mu_demo_bias__shared_prior_p1; real mu_demo_bias__shared_prior_p2; real mu_demo_bias__shared_prior_p3;
  int<lower=1, upper=8> sd_demo_bias__shared_prior_family; real sd_demo_bias__shared_prior_p1; real sd_demo_bias__shared_prior_p2; real sd_demo_bias__shared_prior_p3;

  int<lower=1, upper=8> mu_beta__shared_prior_family;  real mu_beta__shared_prior_p1;  real mu_beta__shared_prior_p2;  real mu_beta__shared_prior_p3;
  int<lower=1, upper=8> sd_beta__shared_prior_family;  real sd_beta__shared_prior_p1;  real sd_beta__shared_prior_p2;  real sd_beta__shared_prior_p3;

  int<lower=1, upper=8> mu_kappa__shared_prior_family;  real mu_kappa__shared_prior_p1;  real mu_kappa__shared_prior_p2;  real mu_kappa__shared_prior_p3;
  int<lower=1, upper=8> sd_kappa__shared_prior_family;  real sd_kappa__shared_prior_p1;  real sd_kappa__shared_prior_p2;  real sd_kappa__shared_prior_p3;

  // hyperpriors on z-scale (delta, per non-baseline condition)
  int<lower=1, upper=8> mu_alpha_o__delta_prior_family;  real mu_alpha_o__delta_prior_p1;  real mu_alpha_o__delta_prior_p2;  real mu_alpha_o__delta_prior_p3;
  int<lower=1, upper=8> sd_alpha_o__delta_prior_family;  real sd_alpha_o__delta_prior_p1;  real sd_alpha_o__delta_prior_p2;  real sd_alpha_o__delta_prior_p3;

  int<lower=1, upper=8> mu_demo_bias__delta_prior_family;  real mu_demo_bias__delta_prior_p1;  real mu_demo_bias__delta_prior_p2;  real mu_demo_bias__delta_prior_p3;
  int<lower=1, upper=8> sd_demo_bias__delta_prior_family;  real sd_demo_bias__delta_prior_p1;  real sd_demo_bias__delta_prior_p2;  real sd_demo_bias__delta_prior_p3;

  int<lower=1, upper=8> mu_beta__delta_prior_family;   real mu_beta__delta_prior_p1;   real mu_beta__delta_prior_p2;   real mu_beta__delta_prior_p3;
  int<lower=1, upper=8> sd_beta__delta_prior_family;   real sd_beta__delta_prior_p1;   real sd_beta__delta_prior_p2;   real sd_beta__delta_prior_p3;

  int<lower=1, upper=8> mu_kappa__delta_prior_family;   real mu_kappa__delta_prior_p1;   real mu_kappa__delta_prior_p2;   real mu_kappa__delta_prior_p3;
  int<lower=1, upper=8> sd_kappa__delta_prior_family;   real sd_kappa__delta_prior_p1;   real sd_kappa__delta_prior_p2;   real sd_kappa__delta_prior_p3;
}
parameters {
  // shared (non-centered)
  real mu_alpha_o__shared; real<lower=0> sd_alpha_o__shared; vector[N] z_alpha_o__shared;
  real mu_demo_bias__shared; real<lower=0> sd_demo_bias__shared; vector[N] z_demo_bias__shared;
  real mu_beta__shared;  real<lower=0> sd_beta__shared;  vector[N] z_beta__shared;
  real mu_kappa__shared;  real<lower=0> sd_kappa__shared;  vector[N] z_kappa__shared;

  // deltas for non-baseline conditions
  vector[C-1] mu_alpha_o__delta; vector<lower=0>[C-1] sd_alpha_o__delta; matrix[N, C-1] z_alpha_o__delta;
  vector[C-1] mu_demo_bias__delta; vector<lower=0>[C-1] sd_demo_bias__delta; matrix[N, C-1] z_demo_bias__delta;
  vector[C-1] mu_beta__delta;  vector<lower=0>[C-1] sd_beta__delta;  matrix[N, C-1] z_beta__delta;
  vector[C-1] mu_kappa__delta;  vector<lower=0>[C-1] sd_kappa__delta;  matrix[N, C-1] z_kappa__delta;
}
transformed parameters {
  vector[N] alpha_o__shared_z = mu_alpha_o__shared + sd_alpha_o__shared * z_alpha_o__shared;
  vector[N] demo_bias__shared_z = mu_demo_bias__shared + sd_demo_bias__shared * z_demo_bias__shared;
  vector[N] beta__shared_z  = mu_beta__shared  + sd_beta__shared  * z_beta__shared;
  vector[N] kappa__shared_z  = mu_kappa__shared  + sd_kappa__shared  * z_kappa__shared;

  matrix[N, C] alpha_o_z;
  matrix[N, C] demo_bias_z;
  matrix[N, C] beta_z;
  matrix[N, C] kappa_z;

  matrix[N, C - 1] alpha_o__delta_z;
  matrix[N, C - 1] demo_bias__delta_z;
  matrix[N, C - 1] beta__delta_z;
  matrix[N, C - 1] kappa__delta_z;

  for (n in 1:N) {
    for (i in 1:(C - 1)) {
      alpha_o__delta_z[n, i] = mu_alpha_o__delta[i] + sd_alpha_o__delta[i] * z_alpha_o__delta[n, i];
      demo_bias__delta_z[n, i] = mu_demo_bias__delta[i] + sd_demo_bias__delta[i] * z_demo_bias__delta[n, i];
      beta__delta_z[n, i]    = mu_beta__delta[i]  + sd_beta__delta[i]  * z_beta__delta[n, i];
      kappa__delta_z[n, i]   = mu_kappa__delta[i]  + sd_kappa__delta[i]  * z_kappa__delta[n, i];
    }
  }

  for (n in 1:N) {
    for (c in 1:C) {
      if (c == baseline_cond) {
        alpha_o_z[n, c] = alpha_o__shared_z[n];
        demo_bias_z[n, c] = demo_bias__shared_z[n];
        beta_z[n, c]  = beta__shared_z[n];
        kappa_z[n, c]  = kappa__shared_z[n];
      } else {
        int idx = (c < baseline_cond) ? c : (c - 1);
        alpha_o_z[n, c] = alpha_o__shared_z[n] + alpha_o__delta_z[n, idx];
        demo_bias_z[n, c] = demo_bias__shared_z[n] + demo_bias__delta_z[n, idx];
        beta_z[n, c]  = beta__shared_z[n]  + beta__delta_z[n, idx];
        kappa_z[n, c]  = kappa__shared_z[n]  + kappa__delta_z[n, idx];
      }
    }
  }

  matrix<lower=0, upper=1>[N, C] alpha_o = inv_logit(alpha_o_z);
  matrix<lower=-demo_bias_abs_max, upper=demo_bias_abs_max>[N, C] demo_bias =
    demo_bias_abs_max * (2 * inv_logit(demo_bias_z) - 1);

  matrix<lower=beta_lower, upper=beta_upper>[N, C] beta =
    beta_lower + (beta_upper - beta_lower) .* inv_logit(beta_z);

  matrix<lower=-kappa_abs_max, upper=kappa_abs_max>[N, C] kappa =
    kappa_abs_max * (2 * inv_logit(kappa_z) - 1);

  // population-level (location) per condition
  vector[C] alpha_o_pop;
  vector[C] demo_bias_pop;
  vector[C] beta_pop;
  vector[C] kappa_pop;

  for (c in 1:C) {
    if (c == baseline_cond) {
      alpha_o_pop[c] = inv_logit(mu_alpha_o__shared);
      demo_bias_pop[c] = demo_bias_abs_max * (2 * inv_logit(mu_demo_bias__shared) - 1);
      beta_pop[c] = beta_lower + (beta_upper - beta_lower) * inv_logit(mu_beta__shared);
      kappa_pop[c] = kappa_abs_max * (2 * inv_logit(mu_kappa__shared) - 1);
    } else {
      int idx = (c < baseline_cond) ? c : (c - 1);
      alpha_o_pop[c] = inv_logit(mu_alpha_o__shared + mu_alpha_o__delta[idx]);
      demo_bias_pop[c] = demo_bias_abs_max * (2 * inv_logit(mu_demo_bias__shared + mu_demo_bias__delta[idx]) - 1);
      beta_pop[c] = beta_lower + (beta_upper - beta_lower) * inv_logit(mu_beta__shared + mu_beta__delta[idx]);
      kappa_pop[c] = kappa_abs_max * (2 * inv_logit(mu_kappa__shared + mu_kappa__delta[idx]) - 1);
    }
  }
}
model {
  z_alpha_o__shared ~ normal(0, 1);
  z_demo_bias__shared ~ normal(0, 1);
  z_beta__shared  ~ normal(0, 1);
  z_kappa__shared  ~ normal(0, 1);

  to_vector(z_alpha_o__delta) ~ normal(0, 1);
  to_vector(z_demo_bias__delta) ~ normal(0, 1);
  to_vector(z_beta__delta)  ~ normal(0, 1);
  to_vector(z_kappa__delta)  ~ normal(0, 1);

  // shared hyperpriors
  target += prior_lpdf(mu_alpha_o__shared | mu_alpha_o__shared_prior_family, mu_alpha_o__shared_prior_p1, mu_alpha_o__shared_prior_p2, mu_alpha_o__shared_prior_p3);
  target += prior_lpdf(sd_alpha_o__shared | sd_alpha_o__shared_prior_family, sd_alpha_o__shared_prior_p1, sd_alpha_o__shared_prior_p2, sd_alpha_o__shared_prior_p3);

  target += prior_lpdf(mu_demo_bias__shared | mu_demo_bias__shared_prior_family, mu_demo_bias__shared_prior_p1, mu_demo_bias__shared_prior_p2, mu_demo_bias__shared_prior_p3);
  target += prior_lpdf(sd_demo_bias__shared | sd_demo_bias__shared_prior_family, sd_demo_bias__shared_prior_p1, sd_demo_bias__shared_prior_p2, sd_demo_bias__shared_prior_p3);

  target += prior_lpdf(mu_beta__shared  | mu_beta__shared_prior_family,  mu_beta__shared_prior_p1,  mu_beta__shared_prior_p2,  mu_beta__shared_prior_p3);
  target += prior_lpdf(sd_beta__shared  | sd_beta__shared_prior_family,  sd_beta__shared_prior_p1,  sd_beta__shared_prior_p2,  sd_beta__shared_prior_p3);

  target += prior_lpdf(mu_kappa__shared  | mu_kappa__shared_prior_family,  mu_kappa__shared_prior_p1,  mu_kappa__shared_prior_p2,  mu_kappa__shared_prior_p3);
  target += prior_lpdf(sd_kappa__shared  | sd_kappa__shared_prior_family,  sd_kappa__shared_prior_p1,  sd_kappa__shared_prior_p2,  sd_kappa__shared_prior_p3);

  // delta hyperpriors (per non-baseline condition)
  for (cc in 1:(C - 1)) {
    target += prior_lpdf(mu_alpha_o__delta[cc] | mu_alpha_o__delta_prior_family, mu_alpha_o__delta_prior_p1, mu_alpha_o__delta_prior_p2, mu_alpha_o__delta_prior_p3);
    target += prior_lpdf(sd_alpha_o__delta[cc] | sd_alpha_o__delta_prior_family, sd_alpha_o__delta_prior_p1, sd_alpha_o__delta_prior_p2, sd_alpha_o__delta_prior_p3);

    target += prior_lpdf(mu_demo_bias__delta[cc] | mu_demo_bias__delta_prior_family, mu_demo_bias__delta_prior_p1, mu_demo_bias__delta_prior_p2, mu_demo_bias__delta_prior_p3);
    target += prior_lpdf(sd_demo_bias__delta[cc] | sd_demo_bias__delta_prior_family, sd_demo_bias__delta_prior_p1, sd_demo_bias__delta_prior_p2, sd_demo_bias__delta_prior_p3);

    target += prior_lpdf(mu_beta__delta[cc]  | mu_beta__delta_prior_family,  mu_beta__delta_prior_p1,  mu_beta__delta_prior_p2,  mu_beta__delta_prior_p3);
    target += prior_lpdf(sd_beta__delta[cc]  | sd_beta__delta_prior_family,  sd_beta__delta_prior_p1,  sd_beta__delta_prior_p2,  sd_beta__delta_prior_p3);

    target += prior_lpdf(mu_kappa__delta[cc]  | mu_kappa__delta_prior_family,  mu_kappa__delta_prior_p1,  mu_kappa__delta_prior_p2,  mu_kappa__delta_prior_p3);
    target += prior_lpdf(sd_kappa__delta[cc]  | sd_kappa__delta_prior_family,  sd_kappa__delta_prior_p1,  sd_kappa__delta_prior_p2,  sd_kappa__delta_prior_p3);
  }

  array[N] matrix[S, A] Q;
  array[N, S] int last_choice;
  array[N] int recent_demo_choice;
  for (n in 1:N) {
    Q[n] = rep_matrix(0.0, S, A);
    last_choice[n] = rep_array(0, S);
    recent_demo_choice[n] = 0;
  }

  for (e in 1:E) {
    int n = subj[e];
    int s = state[e];
    int c = cond[e];

    if (etype[e] == 1) {
      Q[n] = rep_matrix(0.0, S, A);
      last_choice[n] = rep_array(0, S);
      recent_demo_choice[n] = 0;

    } else if (etype[e] == 2) {
      if (demo_action[e] > 0) {
        int a = demo_action[e];
        recent_demo_choice[n] = a;

        if (has_demo_outcome[e] == 1) {
          real r = demo_outcome_obs[e];
          Q[n][s, a] = Q[n][s, a] + alpha_o[n, c] * (r - Q[n][s, a]);
        }
      }

    } else if (etype[e] == 3) {
      if (choice[e] > 0) {
        vector[A] u = beta[n, c] * to_vector(Q[n][s]');
        if (last_choice[n][s] > 0) u[last_choice[n][s]] += kappa[n, c];
        if (recent_demo_choice[n] > 0) u[recent_demo_choice[n]] += demo_bias[n, c];
        for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
        target += categorical_logit_lpmf(choice[e] | u);
      }

    } else if (etype[e] == 4) {
      if (action[e] > 0) {
        last_choice[n][s] = action[e];
      }
    }
  }
}

generated quantities {
  // z-scale shared summaries (for compatibility with Python wrapper)
  vector[N] alpha_o__shared_z_hat = alpha_o__shared_z;
  vector[N] demo_bias__shared_z_hat = demo_bias__shared_z;
  vector[N] beta__shared_z_hat    = beta__shared_z;
  vector[N] kappa__shared_z_hat   = kappa__shared_z;

  matrix[N, C - 1] alpha_o__delta_z_hat;
  matrix[N, C - 1] demo_bias__delta_z_hat;
  matrix[N, C - 1] beta__delta_z_hat;
  matrix[N, C - 1] kappa__delta_z_hat;

  for (n in 1:N) {
    for (i in 1:(C - 1)) {
      alpha_o__delta_z_hat[n, i] = mu_alpha_o__delta[i] + sd_alpha_o__delta[i] * z_alpha_o__delta[n, i];
      demo_bias__delta_z_hat[n, i] = mu_demo_bias__delta[i] + sd_demo_bias__delta[i] * z_demo_bias__delta[n, i];
      beta__delta_z_hat[n, i]    = mu_beta__delta[i]  + sd_beta__delta[i]  * z_beta__delta[n, i];
      kappa__delta_z_hat[n, i]   = mu_kappa__delta[i]  + sd_kappa__delta[i]  * z_kappa__delta[n, i];
    }
  }

  // subject-condition parameters on natural scale
  matrix[N, C] alpha_o_hat = alpha_o;
  matrix[N, C] demo_bias_hat = demo_bias;
  matrix[N, C] beta_hat    = beta;
  matrix[N, C] kappa_hat   = kappa;

  // expose hyperparameters for convenience
  real mu_alpha_o__shared_hat = mu_alpha_o__shared;
  real sd_alpha_o__shared_hat = sd_alpha_o__shared;
  vector[C - 1] mu_alpha_o__delta_hat = mu_alpha_o__delta;
  vector[C - 1] sd_alpha_o__delta_hat = sd_alpha_o__delta;

  real mu_demo_bias__shared_hat = mu_demo_bias__shared;
  real sd_demo_bias__shared_hat = sd_demo_bias__shared;
  vector[C - 1] mu_demo_bias__delta_hat = mu_demo_bias__delta;
  vector[C - 1] sd_demo_bias__delta_hat = sd_demo_bias__delta;

  real mu_beta__shared_hat = mu_beta__shared;
  real sd_beta__shared_hat = sd_beta__shared;
  vector[C - 1] mu_beta__delta_hat = mu_beta__delta;
  vector[C - 1] sd_beta__delta_hat = sd_beta__delta;

  real mu_kappa__shared_hat = mu_kappa__shared;
  real sd_kappa__shared_hat = sd_kappa__shared;
  vector[C - 1] mu_kappa__delta_hat = mu_kappa__delta;
  vector[C - 1] sd_kappa__delta_hat = sd_kappa__delta;
}
