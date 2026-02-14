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

  array[E] vector<lower=0,upper=1>[A] avail_mask;

  array[E] int<lower=0, upper=A> demo_action;
  vector[E] demo_outcome_obs;
  array[E] int<lower=0, upper=1> has_demo_outcome;

  array[E] int<lower=1, upper=C> cond;

  real<lower=1e-6> beta_lower;
  real<lower=1e-6> beta_upper;

  // hyperpriors on z-scale (shared)
  int<lower=1, upper=8> mu_alpha_o__shared_prior_family; real mu_alpha_o__shared_prior_p1; real mu_alpha_o__shared_prior_p2; real mu_alpha_o__shared_prior_p3;
  int<lower=1, upper=8> sd_alpha_o__shared_prior_family; real sd_alpha_o__shared_prior_p1; real sd_alpha_o__shared_prior_p2; real sd_alpha_o__shared_prior_p3;

  int<lower=1, upper=8> mu_beta__shared_prior_family; real mu_beta__shared_prior_p1; real mu_beta__shared_prior_p2; real mu_beta__shared_prior_p3;
  int<lower=1, upper=8> sd_beta__shared_prior_family; real sd_beta__shared_prior_p1; real sd_beta__shared_prior_p2; real sd_beta__shared_prior_p3;

  // hyperpriors on z-scale (delta, per non-baseline condition)
  int<lower=1, upper=8> mu_alpha_o__delta_prior_family; real mu_alpha_o__delta_prior_p1; real mu_alpha_o__delta_prior_p2; real mu_alpha_o__delta_prior_p3;
  int<lower=1, upper=8> sd_alpha_o__delta_prior_family; real sd_alpha_o__delta_prior_p1; real sd_alpha_o__delta_prior_p2; real sd_alpha_o__delta_prior_p3;

  int<lower=1, upper=8> mu_beta__delta_prior_family; real mu_beta__delta_prior_p1; real mu_beta__delta_prior_p2; real mu_beta__delta_prior_p3;
  int<lower=1, upper=8> sd_beta__delta_prior_family; real sd_beta__delta_prior_p1; real sd_beta__delta_prior_p2; real sd_beta__delta_prior_p3;
}
parameters {
  // shared (non-centered)
  real mu_alpha_o__shared;
  real<lower=0> sd_alpha_o__shared;
  vector[N] z_alpha_o_shared;

  real mu_beta__shared;
  real<lower=0> sd_beta__shared;
  vector[N] z_beta_shared;

  // deltas for non-baseline conditions
  vector[C-1] mu_alpha_o__delta;
  vector<lower=0>[C-1] sd_alpha_o__delta;
  matrix[N, C-1] z_alpha_o_delta;

  vector[C-1] mu_beta__delta;
  vector<lower=0>[C-1] sd_beta__delta;
  matrix[N, C-1] z_beta_delta;
}
transformed parameters {
  vector[N] alpha_o_shared_z = mu_alpha_o__shared + sd_alpha_o__shared * z_alpha_o_shared;
  vector[N] beta_shared_z = mu_beta__shared + sd_beta__shared * z_beta_shared;

  matrix[N, C - 1] alpha_o_delta_z;
  matrix[N, C - 1] beta_delta_z;

  matrix[N, C] alpha_o_z;
  matrix[N, C] beta_z;

  for (n in 1:N) {
    for (i in 1:(C - 1)) {
      alpha_o_delta_z[n, i] = mu_alpha_o__delta[i] + sd_alpha_o__delta[i] * z_alpha_o_delta[n, i];
      beta_delta_z[n, i] = mu_beta__delta[i] + sd_beta__delta[i] * z_beta_delta[n, i];
    }
  }

  for (n in 1:N) {
    for (c in 1:C) {
      if (c == baseline_cond) {
        alpha_o_z[n, c] = alpha_o_shared_z[n];
        beta_z[n, c] = beta_shared_z[n];
      } else {
        int idx = (c < baseline_cond) ? c : (c - 1);
        alpha_o_z[n, c] = alpha_o_shared_z[n] + alpha_o_delta_z[n, idx];
        beta_z[n, c] = beta_shared_z[n] + beta_delta_z[n, idx];
      }
    }
  }

  matrix<lower=0, upper=1>[N, C] alpha_o = inv_logit(alpha_o_z);
  matrix<lower=beta_lower, upper=beta_upper>[N, C] beta =
    beta_lower + (beta_upper - beta_lower) * inv_logit(beta_z);

  vector[C] alpha_o_pop;
  vector[C] beta_pop;

  for (c in 1:C) {
    if (c == baseline_cond) {
      alpha_o_pop[c] = inv_logit(mu_alpha_o__shared);
      beta_pop[c] = beta_lower + (beta_upper - beta_lower) * inv_logit(mu_beta__shared);
    } else {
      int idx = (c < baseline_cond) ? c : (c - 1);
      alpha_o_pop[c] = inv_logit(mu_alpha_o__shared + mu_alpha_o__delta[idx]);
      beta_pop[c] = beta_lower + (beta_upper - beta_lower) * inv_logit(mu_beta__shared + mu_beta__delta[idx]);
    }
  }
}
model {
  z_alpha_o_shared ~ normal(0, 1);
  z_beta_shared ~ normal(0, 1);

  to_vector(z_alpha_o_delta) ~ normal(0, 1);
  to_vector(z_beta_delta) ~ normal(0, 1);

  // shared hyperpriors
  target += prior_lpdf(mu_alpha_o__shared | mu_alpha_o__shared_prior_family, mu_alpha_o__shared_prior_p1, mu_alpha_o__shared_prior_p2, mu_alpha_o__shared_prior_p3);
  target += prior_lpdf(sd_alpha_o__shared | sd_alpha_o__shared_prior_family, sd_alpha_o__shared_prior_p1, sd_alpha_o__shared_prior_p2, sd_alpha_o__shared_prior_p3);

  target += prior_lpdf(mu_beta__shared | mu_beta__shared_prior_family, mu_beta__shared_prior_p1, mu_beta__shared_prior_p2, mu_beta__shared_prior_p3);
  target += prior_lpdf(sd_beta__shared | sd_beta__shared_prior_family, sd_beta__shared_prior_p1, sd_beta__shared_prior_p2, sd_beta__shared_prior_p3);

  // delta hyperpriors
  for (cc in 1:(C - 1)) {
    target += prior_lpdf(mu_alpha_o__delta[cc] | mu_alpha_o__delta_prior_family, mu_alpha_o__delta_prior_p1, mu_alpha_o__delta_prior_p2, mu_alpha_o__delta_prior_p3);
    target += prior_lpdf(sd_alpha_o__delta[cc] | sd_alpha_o__delta_prior_family, sd_alpha_o__delta_prior_p1, sd_alpha_o__delta_prior_p2, sd_alpha_o__delta_prior_p3);

    target += prior_lpdf(mu_beta__delta[cc] | mu_beta__delta_prior_family, mu_beta__delta_prior_p1, mu_beta__delta_prior_p2, mu_beta__delta_prior_p3);
    target += prior_lpdf(sd_beta__delta[cc] | sd_beta__delta_prior_family, sd_beta__delta_prior_p1, sd_beta__delta_prior_p2, sd_beta__delta_prior_p3);
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
        Q[n][s, a] = Q[n][s, a] + alpha_o[n, c] * (r - Q[n][s, a]);
      }

    } else if (etype[e] == 3) {
      if (choice[e] > 0) {
        vector[A] u = to_vector(Q[n][s]');
        for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
        target += categorical_logit_lpmf(choice[e] | beta[n, c] * u);
      }
    }
  }
}
generated quantities {
  vector[E] log_lik = rep_vector(0.0, E);
  {
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
          Q[n][s, a] = Q[n][s, a] + alpha_o[n, c] * (r - Q[n][s, a]);
        }

      } else if (etype[e] == 3) {
        if (choice[e] > 0) {
          vector[A] u = to_vector(Q[n][s]');
          for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
          log_lik[e] = categorical_logit_lpmf(choice[e] | beta[n, c] * u);
        }
      }
    }
  }

  matrix[N, C] alpha_o_hat = alpha_o;
  matrix[N, C] beta_hat = beta;

  vector[N] alpha_o__shared_z_hat = alpha_o_shared_z;
  vector[N] beta__shared_z_hat = beta_shared_z;

  matrix[N, C - 1] alpha_o__delta_z_hat = alpha_o_delta_z;
  matrix[N, C - 1] beta__delta_z_hat = beta_delta_z;

  real mu_alpha_o__shared_hat = mu_alpha_o__shared;
  real sd_alpha_o__shared_hat = sd_alpha_o__shared;
  real mu_beta__shared_hat = mu_beta__shared;
  real sd_beta__shared_hat = sd_beta__shared;

  vector[C - 1] mu_alpha_o__delta_hat = mu_alpha_o__delta;
  vector[C - 1] sd_alpha_o__delta_hat = sd_alpha_o__delta;
  vector[C - 1] mu_beta__delta_hat = mu_beta__delta;
  vector[C - 1] sd_beta__delta_hat = sd_beta__delta;
}
