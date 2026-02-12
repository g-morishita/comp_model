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
  vector[E] demo_outcome_obs; // unused
  array[E] int<lower=0, upper=1> has_demo_outcome; // unused

  array[E] int<lower=1, upper=C> cond;

  real<lower=1e-6> beta_lower;
  real<lower=1e-6> beta_upper;
  real<lower=0> kappa_abs_max;

  // hyperpriors on z-scale (shared)
  int<lower=1, upper=8> mu_alpha_a__shared_prior_family; real mu_alpha_a__shared_prior_p1; real mu_alpha_a__shared_prior_p2; real mu_alpha_a__shared_prior_p3;
  int<lower=1, upper=8> sd_alpha_a__shared_prior_family; real sd_alpha_a__shared_prior_p1; real sd_alpha_a__shared_prior_p2; real sd_alpha_a__shared_prior_p3;

  int<lower=1, upper=8> mu_beta__shared_prior_family; real mu_beta__shared_prior_p1; real mu_beta__shared_prior_p2; real mu_beta__shared_prior_p3;
  int<lower=1, upper=8> sd_beta__shared_prior_family; real sd_beta__shared_prior_p1; real sd_beta__shared_prior_p2; real sd_beta__shared_prior_p3;

  int<lower=1, upper=8> mu_kappa__shared_prior_family; real mu_kappa__shared_prior_p1; real mu_kappa__shared_prior_p2; real mu_kappa__shared_prior_p3;
  int<lower=1, upper=8> sd_kappa__shared_prior_family; real sd_kappa__shared_prior_p1; real sd_kappa__shared_prior_p2; real sd_kappa__shared_prior_p3;

  // hyperpriors on z-scale (delta)
  int<lower=1, upper=8> mu_alpha_a__delta_prior_family; real mu_alpha_a__delta_prior_p1; real mu_alpha_a__delta_prior_p2; real mu_alpha_a__delta_prior_p3;
  int<lower=1, upper=8> sd_alpha_a__delta_prior_family; real sd_alpha_a__delta_prior_p1; real sd_alpha_a__delta_prior_p2; real sd_alpha_a__delta_prior_p3;

  int<lower=1, upper=8> mu_beta__delta_prior_family; real mu_beta__delta_prior_p1; real mu_beta__delta_prior_p2; real mu_beta__delta_prior_p3;
  int<lower=1, upper=8> sd_beta__delta_prior_family; real sd_beta__delta_prior_p1; real sd_beta__delta_prior_p2; real sd_beta__delta_prior_p3;

  int<lower=1, upper=8> mu_kappa__delta_prior_family; real mu_kappa__delta_prior_p1; real mu_kappa__delta_prior_p2; real mu_kappa__delta_prior_p3;
  int<lower=1, upper=8> sd_kappa__delta_prior_family; real sd_kappa__delta_prior_p1; real sd_kappa__delta_prior_p2; real sd_kappa__delta_prior_p3;
}
parameters {
  // shared (non-centered)
  real mu_alpha_a__shared; real<lower=0> sd_alpha_a__shared; vector[N] z_alpha_a__shared;
  real mu_beta__shared; real<lower=0> sd_beta__shared; vector[N] z_beta__shared;
  real mu_kappa__shared; real<lower=0> sd_kappa__shared; vector[N] z_kappa__shared;

  // deltas for non-baseline conditions
  vector[C-1] mu_alpha_a__delta; vector<lower=0>[C-1] sd_alpha_a__delta; matrix[N, C-1] z_alpha_a__delta;
  vector[C-1] mu_beta__delta; vector<lower=0>[C-1] sd_beta__delta; matrix[N, C-1] z_beta__delta;
  vector[C-1] mu_kappa__delta; vector<lower=0>[C-1] sd_kappa__delta; matrix[N, C-1] z_kappa__delta;
}
transformed parameters {
  vector[N] alpha_a__shared_z = mu_alpha_a__shared + sd_alpha_a__shared * z_alpha_a__shared;
  vector[N] beta__shared_z = mu_beta__shared + sd_beta__shared * z_beta__shared;
  vector[N] kappa__shared_z = mu_kappa__shared + sd_kappa__shared * z_kappa__shared;

  matrix[N, C - 1] alpha_a__delta_z;
  matrix[N, C - 1] beta__delta_z;
  matrix[N, C - 1] kappa__delta_z;

  matrix[N, C] alpha_a_z;
  matrix[N, C] beta_z;
  matrix[N, C] kappa_z;

  for (n in 1:N) {
    for (i in 1:(C - 1)) {
      alpha_a__delta_z[n, i] = mu_alpha_a__delta[i] + sd_alpha_a__delta[i] * z_alpha_a__delta[n, i];
      beta__delta_z[n, i] = mu_beta__delta[i] + sd_beta__delta[i] * z_beta__delta[n, i];
      kappa__delta_z[n, i] = mu_kappa__delta[i] + sd_kappa__delta[i] * z_kappa__delta[n, i];
    }
  }

  for (n in 1:N) {
    for (c in 1:C) {
      if (c == baseline_cond) {
        alpha_a_z[n, c] = alpha_a__shared_z[n];
        beta_z[n, c] = beta__shared_z[n];
        kappa_z[n, c] = kappa__shared_z[n];
      } else {
        int idx = (c < baseline_cond) ? c : (c - 1);
        alpha_a_z[n, c] = alpha_a__shared_z[n] + alpha_a__delta_z[n, idx];
        beta_z[n, c] = beta__shared_z[n] + beta__delta_z[n, idx];
        kappa_z[n, c] = kappa__shared_z[n] + kappa__delta_z[n, idx];
      }
    }
  }

  matrix<lower=0, upper=1>[N, C] alpha_a = inv_logit(alpha_a_z);
  matrix<lower=beta_lower, upper=beta_upper>[N, C] beta =
    beta_lower + (beta_upper - beta_lower) * (tanh(beta_z) + 1) * 0.5;
  matrix<lower=-kappa_abs_max, upper=kappa_abs_max>[N, C] kappa =
    kappa_abs_max * tanh(kappa_z);

  vector[C] alpha_a_pop;
  vector[C] beta_pop;
  vector[C] kappa_pop;

  for (c in 1:C) {
    if (c == baseline_cond) {
      alpha_a_pop[c] = inv_logit(mu_alpha_a__shared);
      beta_pop[c] = beta_lower + (beta_upper - beta_lower) * (tanh(mu_beta__shared) + 1) * 0.5;
      kappa_pop[c] = kappa_abs_max * tanh(mu_kappa__shared);
    } else {
      int idx = (c < baseline_cond) ? c : (c - 1);
      alpha_a_pop[c] = inv_logit(mu_alpha_a__shared + mu_alpha_a__delta[idx]);
      beta_pop[c] = beta_lower + (beta_upper - beta_lower) * (tanh(mu_beta__shared + mu_beta__delta[idx]) + 1) * 0.5;
      kappa_pop[c] = kappa_abs_max * tanh(mu_kappa__shared + mu_kappa__delta[idx]);
    }
  }
}
model {
  z_alpha_a__shared ~ normal(0, 1);
  z_beta__shared ~ normal(0, 1);
  z_kappa__shared ~ normal(0, 1);

  to_vector(z_alpha_a__delta) ~ normal(0, 1);
  to_vector(z_beta__delta) ~ normal(0, 1);
  to_vector(z_kappa__delta) ~ normal(0, 1);

  // shared hyperpriors
  target += prior_lpdf(mu_alpha_a__shared | mu_alpha_a__shared_prior_family, mu_alpha_a__shared_prior_p1, mu_alpha_a__shared_prior_p2, mu_alpha_a__shared_prior_p3);
  target += prior_lpdf(sd_alpha_a__shared | sd_alpha_a__shared_prior_family, sd_alpha_a__shared_prior_p1, sd_alpha_a__shared_prior_p2, sd_alpha_a__shared_prior_p3);

  target += prior_lpdf(mu_beta__shared | mu_beta__shared_prior_family, mu_beta__shared_prior_p1, mu_beta__shared_prior_p2, mu_beta__shared_prior_p3);
  target += prior_lpdf(sd_beta__shared | sd_beta__shared_prior_family, sd_beta__shared_prior_p1, sd_beta__shared_prior_p2, sd_beta__shared_prior_p3);

  target += prior_lpdf(mu_kappa__shared | mu_kappa__shared_prior_family, mu_kappa__shared_prior_p1, mu_kappa__shared_prior_p2, mu_kappa__shared_prior_p3);
  target += prior_lpdf(sd_kappa__shared | sd_kappa__shared_prior_family, sd_kappa__shared_prior_p1, sd_kappa__shared_prior_p2, sd_kappa__shared_prior_p3);

  // delta hyperpriors
  for (cc in 1:(C - 1)) {
    target += prior_lpdf(mu_alpha_a__delta[cc] | mu_alpha_a__delta_prior_family, mu_alpha_a__delta_prior_p1, mu_alpha_a__delta_prior_p2, mu_alpha_a__delta_prior_p3);
    target += prior_lpdf(sd_alpha_a__delta[cc] | sd_alpha_a__delta_prior_family, sd_alpha_a__delta_prior_p1, sd_alpha_a__delta_prior_p2, sd_alpha_a__delta_prior_p3);

    target += prior_lpdf(mu_beta__delta[cc] | mu_beta__delta_prior_family, mu_beta__delta_prior_p1, mu_beta__delta_prior_p2, mu_beta__delta_prior_p3);
    target += prior_lpdf(sd_beta__delta[cc] | sd_beta__delta_prior_family, sd_beta__delta_prior_p1, sd_beta__delta_prior_p2, sd_beta__delta_prior_p3);

    target += prior_lpdf(mu_kappa__delta[cc] | mu_kappa__delta_prior_family, mu_kappa__delta_prior_p1, mu_kappa__delta_prior_p2, mu_kappa__delta_prior_p3);
    target += prior_lpdf(sd_kappa__delta[cc] | sd_kappa__delta_prior_family, sd_kappa__delta_prior_p1, sd_kappa__delta_prior_p2, sd_kappa__delta_prior_p3);
  }

  array[N, S] int last_choice;
  array[N] vector[A] demo_pi;

  for (n in 1:N) {
    for (s in 1:S) last_choice[n, s] = 0;
    demo_pi[n] = rep_vector(1.0 / A, A);
  }

  for (e in 1:E) {
    int n = subj[e];
    int s = state[e];
    int c = cond[e];

    if (etype[e] == 1) {
      for (s2 in 1:S) last_choice[n, s2] = 0;
      demo_pi[n] = rep_vector(1.0 / A, A);

    } else if (etype[e] == 2) {
      if (demo_action[e] > 0) {
        int a = demo_action[e];
        vector[A] onehot = rep_vector(0.0, A);
        onehot[a] = 1.0;
        demo_pi[n] = demo_pi[n] + alpha_a[n, c] * (onehot - demo_pi[n]);
      }

    } else if (etype[e] == 3) {
      if (choice[e] > 0) {
        vector[A] g = (demo_pi[n] - (1.0 / A)) / (1.0 - (1.0 / A));
        vector[A] u = beta[n, c] * g;
        if (last_choice[n, s] > 0) u[last_choice[n, s]] += kappa[n, c];
        for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
        target += categorical_logit_lpmf(choice[e] | u);
      }

    } else if (etype[e] == 4) {
      if (action[e] > 0) {
        last_choice[n, s] = action[e];
      }
    }
  }
}
generated quantities {
  vector[E] log_lik = rep_vector(0.0, E);
  {
    array[N, S] int last_choice;
    array[N] vector[A] demo_pi;

    for (n in 1:N) {
      for (s in 1:S) last_choice[n, s] = 0;
      demo_pi[n] = rep_vector(1.0 / A, A);
    }

    for (e in 1:E) {
      int n = subj[e];
      int s = state[e];
      int c = cond[e];

      if (etype[e] == 1) {
        for (s2 in 1:S) last_choice[n, s2] = 0;
        demo_pi[n] = rep_vector(1.0 / A, A);

      } else if (etype[e] == 2) {
        if (demo_action[e] > 0) {
          int a = demo_action[e];
          vector[A] onehot = rep_vector(0.0, A);
          onehot[a] = 1.0;
          demo_pi[n] = demo_pi[n] + alpha_a[n, c] * (onehot - demo_pi[n]);
        }

      } else if (etype[e] == 3) {
        if (choice[e] > 0) {
          vector[A] g = (demo_pi[n] - (1.0 / A)) / (1.0 - (1.0 / A));
          vector[A] u = beta[n, c] * g;
          if (last_choice[n, s] > 0) u[last_choice[n, s]] += kappa[n, c];
          for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
          log_lik[e] = categorical_logit_lpmf(choice[e] | u);
        }

      } else if (etype[e] == 4) {
        if (action[e] > 0) {
          last_choice[n, s] = action[e];
        }
      }
    }
  }

  matrix[N, C] alpha_a_hat = alpha_a;
  matrix[N, C] beta_hat = beta;
  matrix[N, C] kappa_hat = kappa;

  vector[N] alpha_a__shared_z_hat = alpha_a__shared_z;
  vector[N] beta__shared_z_hat = beta__shared_z;
  vector[N] kappa__shared_z_hat = kappa__shared_z;

  matrix[N, C - 1] alpha_a__delta_z_hat = alpha_a__delta_z;
  matrix[N, C - 1] beta__delta_z_hat = beta__delta_z;
  matrix[N, C - 1] kappa__delta_z_hat = kappa__delta_z;

  real mu_alpha_a__shared_hat = mu_alpha_a__shared;
  real sd_alpha_a__shared_hat = sd_alpha_a__shared;
  real mu_beta__shared_hat = mu_beta__shared;
  real sd_beta__shared_hat = sd_beta__shared;
  real mu_kappa__shared_hat = mu_kappa__shared;
  real sd_kappa__shared_hat = sd_kappa__shared;

  vector[C - 1] mu_alpha_a__delta_hat = mu_alpha_a__delta;
  vector[C - 1] sd_alpha_a__delta_hat = sd_alpha_a__delta;
  vector[C - 1] mu_beta__delta_hat = mu_beta__delta;
  vector[C - 1] sd_beta__delta_hat = sd_beta__delta;
  vector[C - 1] mu_kappa__delta_hat = mu_kappa__delta;
  vector[C - 1] sd_kappa__delta_hat = sd_kappa__delta;
}
