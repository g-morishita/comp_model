data {
  int<lower=1> A;
  int<lower=1> S;
  int<lower=1> E;
  int<lower=1> C;
  int<lower=1, upper=C> baseline_cond;

  array[E] int<lower=1, upper=4> etype;
  array[E] int<lower=1, upper=S> state;
  array[E] int<lower=0, upper=A> choice;
  array[E] int<lower=0, upper=A> action;
  vector[E] outcome_obs;

  array[E] int<lower=0, upper=A> demo_action;
  vector[E] demo_outcome_obs;
  array[E] int<lower=0, upper=1> has_demo_outcome;

  array[E] int<lower=1, upper=C> cond;

  real pseudo_reward;
  real<lower=1e-6> beta_lower;
  real<lower=1e-6> beta_upper;
  real<lower=0> kappa_abs_max;

  // priors on z-scale (configurable)
  int<lower=1, upper=8> alpha_p_shared_prior_family;
  real alpha_p_shared_prior_p1;
  real alpha_p_shared_prior_p2;
  real alpha_p_shared_prior_p3;

  int<lower=1, upper=8> alpha_p_delta_prior_family;
  real alpha_p_delta_prior_p1;
  real alpha_p_delta_prior_p2;
  real alpha_p_delta_prior_p3;

  int<lower=1, upper=8> alpha_i_shared_prior_family;
  real alpha_i_shared_prior_p1;
  real alpha_i_shared_prior_p2;
  real alpha_i_shared_prior_p3;

  int<lower=1, upper=8> alpha_i_delta_prior_family;
  real alpha_i_delta_prior_p1;
  real alpha_i_delta_prior_p2;
  real alpha_i_delta_prior_p3;

  int<lower=1, upper=8> beta_shared_prior_family;
  real beta_shared_prior_p1;
  real beta_shared_prior_p2;
  real beta_shared_prior_p3;

  int<lower=1, upper=8> beta_delta_prior_family;
  real beta_delta_prior_p1;
  real beta_delta_prior_p2;
  real beta_delta_prior_p3;

  int<lower=1, upper=8> kappa_shared_prior_family;
  real kappa_shared_prior_p1;
  real kappa_shared_prior_p2;
  real kappa_shared_prior_p3;

  int<lower=1, upper=8> kappa_delta_prior_family;
  real kappa_delta_prior_p1;
  real kappa_delta_prior_p2;
  real kappa_delta_prior_p3;
}
parameters {
  real alpha_p__shared_z;
  vector[C-1] alpha_p__delta_z;

  real alpha_i__shared_z;
  vector[C-1] alpha_i__delta_z;

  real beta__shared_z;
  vector[C-1] beta__delta_z;

  real kappa__shared_z;
  vector[C-1] kappa__delta_z;
}
transformed parameters {
  vector[C] alpha_p_z;
  vector[C] alpha_i_z;
  vector[C] beta_z;
  vector[C] kappa_z;

  for (c in 1:C) {
    if (c == baseline_cond) {
      alpha_p_z[c] = alpha_p__shared_z;
      alpha_i_z[c] = alpha_i__shared_z;
      beta_z[c]    = beta__shared_z;
      kappa_z[c]   = kappa__shared_z;
    } else {
      int idx = (c < baseline_cond) ? c : (c - 1);
      alpha_p_z[c] = alpha_p__shared_z + alpha_p__delta_z[idx];
      alpha_i_z[c] = alpha_i__shared_z + alpha_i__delta_z[idx];
      beta_z[c]    = beta__shared_z    + beta__delta_z[idx];
      kappa_z[c]   = kappa__shared_z   + kappa__delta_z[idx];
    }
  }

  vector<lower=0, upper=1>[C] alpha_p = inv_logit(alpha_p_z);
  vector<lower=0, upper=1>[C] alpha_i = inv_logit(alpha_i_z);

  vector<lower=beta_lower, upper=beta_upper>[C] beta =
    beta_lower + (beta_upper - beta_lower) * inv_logit(beta_z);

  vector<lower=-kappa_abs_max, upper=kappa_abs_max>[C] kappa =
    kappa_abs_max * (2 * inv_logit(kappa_z) - 1);
}
model {
  target += prior_lpdf(alpha_p__shared_z | alpha_p_shared_prior_family, alpha_p_shared_prior_p1, alpha_p_shared_prior_p2, alpha_p_shared_prior_p3);
  target += prior_lpdf(alpha_i__shared_z | alpha_i_shared_prior_family, alpha_i_shared_prior_p1, alpha_i_shared_prior_p2, alpha_i_shared_prior_p3);
  target += prior_lpdf(beta__shared_z    | beta_shared_prior_family,    beta_shared_prior_p1,    beta_shared_prior_p2,    beta_shared_prior_p3);
  target += prior_lpdf(kappa__shared_z   | kappa_shared_prior_family,   kappa_shared_prior_p1,   kappa_shared_prior_p2,   kappa_shared_prior_p3);

  for (i in 1:(C - 1)) {
    target += prior_lpdf(alpha_p__delta_z[i] | alpha_p_delta_prior_family, alpha_p_delta_prior_p1, alpha_p_delta_prior_p2, alpha_p_delta_prior_p3);
    target += prior_lpdf(alpha_i__delta_z[i] | alpha_i_delta_prior_family, alpha_i_delta_prior_p1, alpha_i_delta_prior_p2, alpha_i_delta_prior_p3);
    target += prior_lpdf(beta__delta_z[i]    | beta_delta_prior_family,    beta_delta_prior_p1,    beta_delta_prior_p2,    beta_delta_prior_p3);
    target += prior_lpdf(kappa__delta_z[i]   | kappa_delta_prior_family,   kappa_delta_prior_p1,   kappa_delta_prior_p2,   kappa_delta_prior_p3);
  }

  matrix[S, A] Q = rep_matrix(0.0, S, A);
  array[S] int last_choice = rep_array(0, S);

  for (e in 1:E) {
    int s = state[e];
    int c = cond[e];

    if (etype[e] == 1) {
      Q = rep_matrix(0.0, S, A);
      last_choice = rep_array(0, S);

    } else if (etype[e] == 2) {
      if (demo_action[e] > 0) {
        int a = demo_action[e];
        Q[s, a] = Q[s, a] + alpha_i[c] * (pseudo_reward - Q[s, a]);
      }

    } else if (etype[e] == 3) {
      if (choice[e] > 0) {
        vector[A] u = to_vector(Q[s]');
        if (last_choice[s] > 0) u[last_choice[s]] += kappa[c];
        target += categorical_logit_lpmf(choice[e] | beta[c] * u);
      }

    } else if (etype[e] == 4) {
      if (action[e] > 0) {
        int a = action[e];
        real r = outcome_obs[e];
        Q[s, a] = Q[s, a] + alpha_p[c] * (r - Q[s, a]);
        last_choice[s] = a;
      }
    }
  }
}
generated quantities {
  vector[C] alpha_p_hat = alpha_p;
  vector[C] alpha_i_hat = alpha_i;
  vector[C] beta_hat    = beta;
  vector[C] kappa_hat   = kappa;

  real alpha_p__shared_z_hat = alpha_p__shared_z;
  real alpha_i__shared_z_hat = alpha_i__shared_z;
  real beta__shared_z_hat    = beta__shared_z;
  real kappa__shared_z_hat   = kappa__shared_z;

  vector[C-1] alpha_p__delta_z_hat = alpha_p__delta_z;
  vector[C-1] alpha_i__delta_z_hat = alpha_i__delta_z;
  vector[C-1] beta__delta_z_hat    = beta__delta_z;
  vector[C-1] kappa__delta_z_hat   = kappa__delta_z;
}
