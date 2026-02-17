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
  vector[E] outcome_obs; // unused

  array[E] vector<lower=0,upper=1>[A] avail_mask;

  array[E] int<lower=0, upper=A> demo_action;
  vector[E] demo_outcome_obs;
  array[E] int<lower=0, upper=1> has_demo_outcome;

  array[E] int<lower=1, upper=C> cond;

  real<lower=1e-6> beta_lower;

  // hyperpriors on z-scale (shared)
  int<lower=1, upper=8> mu_alpha_o__shared_prior_family; real mu_alpha_o__shared_prior_p1; real mu_alpha_o__shared_prior_p2; real mu_alpha_o__shared_prior_p3;
  int<lower=1, upper=8> sd_alpha_o__shared_prior_family; real sd_alpha_o__shared_prior_p1; real sd_alpha_o__shared_prior_p2; real sd_alpha_o__shared_prior_p3;

  int<lower=1, upper=8> mu_alpha_a__shared_prior_family; real mu_alpha_a__shared_prior_p1; real mu_alpha_a__shared_prior_p2; real mu_alpha_a__shared_prior_p3;
  int<lower=1, upper=8> sd_alpha_a__shared_prior_family; real sd_alpha_a__shared_prior_p1; real sd_alpha_a__shared_prior_p2; real sd_alpha_a__shared_prior_p3;

  int<lower=1, upper=8> mu_beta__shared_prior_family; real mu_beta__shared_prior_p1; real mu_beta__shared_prior_p2; real mu_beta__shared_prior_p3;
  int<lower=1, upper=8> sd_beta__shared_prior_family; real sd_beta__shared_prior_p1; real sd_beta__shared_prior_p2; real sd_beta__shared_prior_p3;

  int<lower=1, upper=8> mu_w__shared_prior_family; real mu_w__shared_prior_p1; real mu_w__shared_prior_p2; real mu_w__shared_prior_p3;
  int<lower=1, upper=8> sd_w__shared_prior_family; real sd_w__shared_prior_p1; real sd_w__shared_prior_p2; real sd_w__shared_prior_p3;

  // hyperpriors on z-scale (delta, per non-baseline condition)
  int<lower=1, upper=8> mu_alpha_o__delta_prior_family; real mu_alpha_o__delta_prior_p1; real mu_alpha_o__delta_prior_p2; real mu_alpha_o__delta_prior_p3;
  int<lower=1, upper=8> sd_alpha_o__delta_prior_family; real sd_alpha_o__delta_prior_p1; real sd_alpha_o__delta_prior_p2; real sd_alpha_o__delta_prior_p3;

  int<lower=1, upper=8> mu_alpha_a__delta_prior_family; real mu_alpha_a__delta_prior_p1; real mu_alpha_a__delta_prior_p2; real mu_alpha_a__delta_prior_p3;
  int<lower=1, upper=8> sd_alpha_a__delta_prior_family; real sd_alpha_a__delta_prior_p1; real sd_alpha_a__delta_prior_p2; real sd_alpha_a__delta_prior_p3;

  int<lower=1, upper=8> mu_beta__delta_prior_family; real mu_beta__delta_prior_p1; real mu_beta__delta_prior_p2; real mu_beta__delta_prior_p3;
  int<lower=1, upper=8> sd_beta__delta_prior_family; real sd_beta__delta_prior_p1; real sd_beta__delta_prior_p2; real sd_beta__delta_prior_p3;

  int<lower=1, upper=8> mu_w__delta_prior_family; real mu_w__delta_prior_p1; real mu_w__delta_prior_p2; real mu_w__delta_prior_p3;
  int<lower=1, upper=8> sd_w__delta_prior_family; real sd_w__delta_prior_p1; real sd_w__delta_prior_p2; real sd_w__delta_prior_p3;
}
parameters {
  // shared (non-centered)
  real mu_alpha_o__shared; real<lower=0> sd_alpha_o__shared; vector[N] z_alpha_o__shared;
  real mu_alpha_a__shared; real<lower=0> sd_alpha_a__shared; vector[N] z_alpha_a__shared;
  real mu_beta__shared; real<lower=0> sd_beta__shared; vector[N] z_beta__shared;
  real mu_w__shared; real<lower=0> sd_w__shared; vector[N] z_w__shared;

  // deltas for non-baseline conditions
  vector[C-1] mu_alpha_o__delta; vector<lower=0>[C-1] sd_alpha_o__delta; matrix[N, C-1] z_alpha_o__delta;
  vector[C-1] mu_alpha_a__delta; vector<lower=0>[C-1] sd_alpha_a__delta; matrix[N, C-1] z_alpha_a__delta;
  vector[C-1] mu_beta__delta; vector<lower=0>[C-1] sd_beta__delta; matrix[N, C-1] z_beta__delta;
  vector[C-1] mu_w__delta; vector<lower=0>[C-1] sd_w__delta; matrix[N, C-1] z_w__delta;
}
transformed parameters {
  vector[N] alpha_o__shared_z = mu_alpha_o__shared + sd_alpha_o__shared * z_alpha_o__shared;
  vector[N] alpha_a__shared_z = mu_alpha_a__shared + sd_alpha_a__shared * z_alpha_a__shared;
  vector[N] beta__shared_z = mu_beta__shared + sd_beta__shared * z_beta__shared;
  vector[N] w__shared_z = mu_w__shared + sd_w__shared * z_w__shared;

  matrix[N, C - 1] alpha_o__delta_z;
  matrix[N, C - 1] alpha_a__delta_z;
  matrix[N, C - 1] beta__delta_z;
  matrix[N, C - 1] w__delta_z;

  matrix[N, C] alpha_o_z;
  matrix[N, C] alpha_a_z;
  matrix[N, C] beta_z;
  matrix[N, C] w_z;

  for (n in 1:N) {
    for (i in 1:(C - 1)) {
      alpha_o__delta_z[n, i] = mu_alpha_o__delta[i] + sd_alpha_o__delta[i] * z_alpha_o__delta[n, i];
      alpha_a__delta_z[n, i] = mu_alpha_a__delta[i] + sd_alpha_a__delta[i] * z_alpha_a__delta[n, i];
      beta__delta_z[n, i] = mu_beta__delta[i] + sd_beta__delta[i] * z_beta__delta[n, i];
      w__delta_z[n, i] = mu_w__delta[i] + sd_w__delta[i] * z_w__delta[n, i];
    }
  }

  for (n in 1:N) {
    for (c in 1:C) {
      if (c == baseline_cond) {
        alpha_o_z[n, c] = alpha_o__shared_z[n];
        alpha_a_z[n, c] = alpha_a__shared_z[n];
        beta_z[n, c] = beta__shared_z[n];
        w_z[n, c] = w__shared_z[n];
      } else {
        int idx = (c < baseline_cond) ? c : (c - 1);
        alpha_o_z[n, c] = alpha_o__shared_z[n] + alpha_o__delta_z[n, idx];
        alpha_a_z[n, c] = alpha_a__shared_z[n] + alpha_a__delta_z[n, idx];
        beta_z[n, c] = beta__shared_z[n] + beta__delta_z[n, idx];
        w_z[n, c] = w__shared_z[n] + w__delta_z[n, idx];
      }
    }
  }

  matrix<lower=0, upper=1>[N, C] alpha_o = inv_logit(alpha_o_z);
  matrix<lower=0, upper=1>[N, C] alpha_a = inv_logit(alpha_a_z);
  matrix<lower=beta_lower>[N, C] beta =
    beta_lower + exp(beta_z);
  matrix<lower=0, upper=1>[N, C] w = inv_logit(w_z);

  vector[C] alpha_o_pop;
  vector[C] alpha_a_pop;
  vector[C] beta_pop;
  vector[C] w_pop;

  for (c in 1:C) {
    if (c == baseline_cond) {
      alpha_o_pop[c] = inv_logit(mu_alpha_o__shared);
      alpha_a_pop[c] = inv_logit(mu_alpha_a__shared);
      beta_pop[c] = beta_lower + exp(mu_beta__shared);
      w_pop[c] = inv_logit(mu_w__shared);
    } else {
      int idx = (c < baseline_cond) ? c : (c - 1);
      alpha_o_pop[c] = inv_logit(mu_alpha_o__shared + mu_alpha_o__delta[idx]);
      alpha_a_pop[c] = inv_logit(mu_alpha_a__shared + mu_alpha_a__delta[idx]);
      beta_pop[c] = beta_lower + exp(mu_beta__shared + mu_beta__delta[idx]);
      w_pop[c] = inv_logit(mu_w__shared + mu_w__delta[idx]);
    }
  }
}
model {
  z_alpha_o__shared ~ normal(0, 1);
  z_alpha_a__shared ~ normal(0, 1);
  z_beta__shared ~ normal(0, 1);
  z_w__shared ~ normal(0, 1);

  to_vector(z_alpha_o__delta) ~ normal(0, 1);
  to_vector(z_alpha_a__delta) ~ normal(0, 1);
  to_vector(z_beta__delta) ~ normal(0, 1);
  to_vector(z_w__delta) ~ normal(0, 1);

  // shared hyperpriors
  target += prior_lpdf(mu_alpha_o__shared | mu_alpha_o__shared_prior_family, mu_alpha_o__shared_prior_p1, mu_alpha_o__shared_prior_p2, mu_alpha_o__shared_prior_p3);
  target += prior_lpdf(sd_alpha_o__shared | sd_alpha_o__shared_prior_family, sd_alpha_o__shared_prior_p1, sd_alpha_o__shared_prior_p2, sd_alpha_o__shared_prior_p3);

  target += prior_lpdf(mu_alpha_a__shared | mu_alpha_a__shared_prior_family, mu_alpha_a__shared_prior_p1, mu_alpha_a__shared_prior_p2, mu_alpha_a__shared_prior_p3);
  target += prior_lpdf(sd_alpha_a__shared | sd_alpha_a__shared_prior_family, sd_alpha_a__shared_prior_p1, sd_alpha_a__shared_prior_p2, sd_alpha_a__shared_prior_p3);

  target += prior_lpdf(mu_beta__shared | mu_beta__shared_prior_family, mu_beta__shared_prior_p1, mu_beta__shared_prior_p2, mu_beta__shared_prior_p3);
  target += prior_lpdf(sd_beta__shared | sd_beta__shared_prior_family, sd_beta__shared_prior_p1, sd_beta__shared_prior_p2, sd_beta__shared_prior_p3);

  target += prior_lpdf(mu_w__shared | mu_w__shared_prior_family, mu_w__shared_prior_p1, mu_w__shared_prior_p2, mu_w__shared_prior_p3);
  target += prior_lpdf(sd_w__shared | sd_w__shared_prior_family, sd_w__shared_prior_p1, sd_w__shared_prior_p2, sd_w__shared_prior_p3);

  // delta hyperpriors
  for (cc in 1:(C - 1)) {
    target += prior_lpdf(mu_alpha_o__delta[cc] | mu_alpha_o__delta_prior_family, mu_alpha_o__delta_prior_p1, mu_alpha_o__delta_prior_p2, mu_alpha_o__delta_prior_p3);
    target += prior_lpdf(sd_alpha_o__delta[cc] | sd_alpha_o__delta_prior_family, sd_alpha_o__delta_prior_p1, sd_alpha_o__delta_prior_p2, sd_alpha_o__delta_prior_p3);

    target += prior_lpdf(mu_alpha_a__delta[cc] | mu_alpha_a__delta_prior_family, mu_alpha_a__delta_prior_p1, mu_alpha_a__delta_prior_p2, mu_alpha_a__delta_prior_p3);
    target += prior_lpdf(sd_alpha_a__delta[cc] | sd_alpha_a__delta_prior_family, sd_alpha_a__delta_prior_p1, sd_alpha_a__delta_prior_p2, sd_alpha_a__delta_prior_p3);

    target += prior_lpdf(mu_beta__delta[cc] | mu_beta__delta_prior_family, mu_beta__delta_prior_p1, mu_beta__delta_prior_p2, mu_beta__delta_prior_p3);
    target += prior_lpdf(sd_beta__delta[cc] | sd_beta__delta_prior_family, sd_beta__delta_prior_p1, sd_beta__delta_prior_p2, sd_beta__delta_prior_p3);

    target += prior_lpdf(mu_w__delta[cc] | mu_w__delta_prior_family, mu_w__delta_prior_p1, mu_w__delta_prior_p2, mu_w__delta_prior_p3);
    target += prior_lpdf(sd_w__delta[cc] | sd_w__delta_prior_family, sd_w__delta_prior_p1, sd_w__delta_prior_p2, sd_w__delta_prior_p3);
  }

  array[N] matrix[S, A] Q;
  array[N] vector[A] demo_pi;

  for (n in 1:N) {
    Q[n] = rep_matrix(0.0, S, A);
    demo_pi[n] = rep_vector(1.0 / A, A);
  }

  for (e in 1:E) {
    int n = subj[e];
    int s = state[e];
    int c = cond[e];

    if (etype[e] == 1) {
      Q[n] = rep_matrix(0.0, S, A);
      demo_pi[n] = rep_vector(1.0 / A, A);

    } else if (etype[e] == 2) {
      if (demo_action[e] > 0) {
        int a = demo_action[e];
        vector[A] onehot = rep_vector(0.0, A);
        onehot[a] = 1.0;
        demo_pi[n] = demo_pi[n] + alpha_a[n, c] * (onehot - demo_pi[n]);

        if (has_demo_outcome[e] == 1) {
          real r = demo_outcome_obs[e];
          Q[n][s, a] = Q[n][s, a] + alpha_o[n, c] * (r - Q[n][s, a]);
        }
      }

    } else if (etype[e] == 3) {
      if (choice[e] > 0) {
        vector[A] g = (demo_pi[n] - (1.0 / A)) / (1.0 - (1.0 / A));
        vector[A] social_drive = w[n, c] * to_vector(Q[n][s]') + (1.0 - w[n, c]) * g;
        vector[A] u = beta[n, c] * social_drive;
        for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
        target += categorical_logit_lpmf(choice[e] | u);
      }
    }
  }
}
generated quantities {
  vector[E] log_lik = rep_vector(0.0, E);
  {
    array[N] matrix[S, A] Q;
    array[N] vector[A] demo_pi;

    for (n in 1:N) {
      Q[n] = rep_matrix(0.0, S, A);
      demo_pi[n] = rep_vector(1.0 / A, A);
    }

    for (e in 1:E) {
      int n = subj[e];
      int s = state[e];
      int c = cond[e];

      if (etype[e] == 1) {
        Q[n] = rep_matrix(0.0, S, A);
        demo_pi[n] = rep_vector(1.0 / A, A);

      } else if (etype[e] == 2) {
        if (demo_action[e] > 0) {
          int a = demo_action[e];
          vector[A] onehot = rep_vector(0.0, A);
          onehot[a] = 1.0;
          demo_pi[n] = demo_pi[n] + alpha_a[n, c] * (onehot - demo_pi[n]);

          if (has_demo_outcome[e] == 1) {
            real r = demo_outcome_obs[e];
            Q[n][s, a] = Q[n][s, a] + alpha_o[n, c] * (r - Q[n][s, a]);
          }
        }

      } else if (etype[e] == 3) {
        if (choice[e] > 0) {
          vector[A] g = (demo_pi[n] - (1.0 / A)) / (1.0 - (1.0 / A));
          vector[A] social_drive = w[n, c] * to_vector(Q[n][s]') + (1.0 - w[n, c]) * g;
          vector[A] u = beta[n, c] * social_drive;
          for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
          log_lik[e] = categorical_logit_lpmf(choice[e] | u);
        }
      }
    }
  }

  matrix[N, C] alpha_o_hat = alpha_o;
  matrix[N, C] alpha_a_hat = alpha_a;
  matrix[N, C] beta_hat = beta;
  matrix[N, C] w_hat = w;

  vector[N] alpha_o__shared_z_hat = alpha_o__shared_z;
  vector[N] alpha_a__shared_z_hat = alpha_a__shared_z;
  vector[N] beta__shared_z_hat = beta__shared_z;
  vector[N] w__shared_z_hat = w__shared_z;

  matrix[N, C - 1] alpha_o__delta_z_hat = alpha_o__delta_z;
  matrix[N, C - 1] alpha_a__delta_z_hat = alpha_a__delta_z;
  matrix[N, C - 1] beta__delta_z_hat = beta__delta_z;
  matrix[N, C - 1] w__delta_z_hat = w__delta_z;

  real mu_alpha_o__shared_hat = mu_alpha_o__shared;
  real sd_alpha_o__shared_hat = sd_alpha_o__shared;
  real mu_alpha_a__shared_hat = mu_alpha_a__shared;
  real sd_alpha_a__shared_hat = sd_alpha_a__shared;
  real mu_beta__shared_hat = mu_beta__shared;
  real sd_beta__shared_hat = sd_beta__shared;
  real mu_w__shared_hat = mu_w__shared;
  real sd_w__shared_hat = sd_w__shared;

  vector[C - 1] mu_alpha_o__delta_hat = mu_alpha_o__delta;
  vector[C - 1] sd_alpha_o__delta_hat = sd_alpha_o__delta;
  vector[C - 1] mu_alpha_a__delta_hat = mu_alpha_a__delta;
  vector[C - 1] sd_alpha_a__delta_hat = sd_alpha_a__delta;
  vector[C - 1] mu_beta__delta_hat = mu_beta__delta;
  vector[C - 1] sd_beta__delta_hat = sd_beta__delta;
  vector[C - 1] mu_w__delta_hat = mu_w__delta;
  vector[C - 1] sd_w__delta_hat = sd_w__delta;
}
