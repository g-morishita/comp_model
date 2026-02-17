data {
  int<lower=1> A;
  int<lower=1> S;
  int<lower=1> E;
  int<lower=1> C;
  int<lower=1,upper=C> baseline_cond;

  array[E] int<lower=1,upper=4> etype;

  array[E] int<lower=1,upper=S> state;
  array[E] int<lower=0,upper=A> choice;
  array[E] int<lower=0,upper=A> action;
  vector[E] outcome_obs; // unused

  array[E] vector<lower=0,upper=1>[A] avail_mask;

  array[E] int<lower=0,upper=A> demo_action;
  vector[E] demo_outcome_obs;
  array[E] int<lower=0,upper=1> has_demo_outcome;

  array[E] int<lower=1,upper=C> cond;

  real<lower=1e-6> beta_lower;
  real<lower=0> kappa_abs_max;

  // priors on z-scale (configurable)
  int<lower=1,upper=8> alpha_o__shared_prior_family;
  real alpha_o__shared_prior_p1;
  real alpha_o__shared_prior_p2;
  real alpha_o__shared_prior_p3;

  int<lower=1,upper=8> alpha_o__delta_prior_family;
  real alpha_o__delta_prior_p1;
  real alpha_o__delta_prior_p2;
  real alpha_o__delta_prior_p3;

  int<lower=1,upper=8> beta__shared_prior_family;
  real beta__shared_prior_p1;
  real beta__shared_prior_p2;
  real beta__shared_prior_p3;

  int<lower=1,upper=8> beta__delta_prior_family;
  real beta__delta_prior_p1;
  real beta__delta_prior_p2;
  real beta__delta_prior_p3;

  int<lower=1,upper=8> kappa__shared_prior_family;
  real kappa__shared_prior_p1;
  real kappa__shared_prior_p2;
  real kappa__shared_prior_p3;

  int<lower=1,upper=8> kappa__delta_prior_family;
  real kappa__delta_prior_p1;
  real kappa__delta_prior_p2;
  real kappa__delta_prior_p3;
}
parameters {
  real alpha_o__shared_z;
  vector[C-1] alpha_o__delta_z;

  real beta__shared_z;
  vector[C-1] beta__delta_z;

  real kappa__shared_z;
  vector[C-1] kappa__delta_z;
}
transformed parameters {
  vector[C] alpha_o_z;
  vector[C] beta_z;
  vector[C] kappa_z;

  for (c in 1:C) {
    if (c == baseline_cond) {
      alpha_o_z[c] = alpha_o__shared_z;
      beta_z[c] = beta__shared_z;
      kappa_z[c] = kappa__shared_z;
    } else {
      int idx = (c < baseline_cond) ? c : (c - 1);
      alpha_o_z[c] = alpha_o__shared_z + alpha_o__delta_z[idx];
      beta_z[c] = beta__shared_z + beta__delta_z[idx];
      kappa_z[c] = kappa__shared_z + kappa__delta_z[idx];
    }
  }

  vector<lower=0,upper=1>[C] alpha_o = inv_logit(alpha_o_z);
  vector<lower=beta_lower>[C] beta =
    beta_lower + exp(beta_z);
  vector<lower=-kappa_abs_max,upper=kappa_abs_max>[C] kappa =
    kappa_abs_max * tanh(kappa_z);
}
model {
  target += prior_lpdf(alpha_o__shared_z | alpha_o__shared_prior_family, alpha_o__shared_prior_p1, alpha_o__shared_prior_p2, alpha_o__shared_prior_p3);
  target += prior_lpdf(beta__shared_z | beta__shared_prior_family, beta__shared_prior_p1, beta__shared_prior_p2, beta__shared_prior_p3);
  target += prior_lpdf(kappa__shared_z | kappa__shared_prior_family, kappa__shared_prior_p1, kappa__shared_prior_p2, kappa__shared_prior_p3);

  for (i in 1:(C-1)) {
    target += prior_lpdf(alpha_o__delta_z[i] | alpha_o__delta_prior_family, alpha_o__delta_prior_p1, alpha_o__delta_prior_p2, alpha_o__delta_prior_p3);
    target += prior_lpdf(beta__delta_z[i] | beta__delta_prior_family, beta__delta_prior_p1, beta__delta_prior_p2, beta__delta_prior_p3);
    target += prior_lpdf(kappa__delta_z[i] | kappa__delta_prior_family, kappa__delta_prior_p1, kappa__delta_prior_p2, kappa__delta_prior_p3);
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
      if (demo_action[e] > 0 && has_demo_outcome[e] == 1) {
        int a = demo_action[e];
        real r = demo_outcome_obs[e];
        Q[s,a] = Q[s,a] + alpha_o[c] * (r - Q[s,a]);
      }

    } else if (etype[e] == 3) {
      if (choice[e] > 0) {
        vector[A] u = to_vector(Q[s]');
        if (last_choice[s] > 0) u[last_choice[s]] += kappa[c];
        for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
        target += categorical_logit_lpmf(choice[e] | beta[c] * u);
      }

    } else if (etype[e] == 4) {
      if (action[e] > 0) {
        last_choice[s] = action[e];
      }
    }
  }
}

generated quantities {
  vector[E] log_lik = rep_vector(0.0, E);
  {
    matrix[S, A] Q = rep_matrix(0.0, S, A);
    array[S] int last_choice = rep_array(0, S);

    for (e in 1:E) {
      int s = state[e];
      int c = cond[e];

      if (etype[e] == 1) {
        Q = rep_matrix(0.0, S, A);
        last_choice = rep_array(0, S);

      } else if (etype[e] == 2) {
        if (demo_action[e] > 0 && has_demo_outcome[e] == 1) {
          int a = demo_action[e];
          real r = demo_outcome_obs[e];
          Q[s,a] = Q[s,a] + alpha_o[c] * (r - Q[s,a]);
        }

      } else if (etype[e] == 3) {
        if (choice[e] > 0) {
          vector[A] u = to_vector(Q[s]');
          if (last_choice[s] > 0) u[last_choice[s]] += kappa[c];
          for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
          log_lik[e] = categorical_logit_lpmf(choice[e] | beta[c] * u);
        }

      } else if (etype[e] == 4) {
        if (action[e] > 0) {
          last_choice[s] = action[e];
        }
      }
    }
  }
  vector[C] alpha_o_hat = alpha_o;
  vector[C] beta_hat = beta;
  vector[C] kappa_hat = kappa;

  real alpha_o__shared_z_hat = alpha_o__shared_z;
  real beta__shared_z_hat = beta__shared_z;
  real kappa__shared_z_hat = kappa__shared_z;

  vector[C-1] alpha_o__delta_z_hat = alpha_o__delta_z;
  vector[C-1] beta__delta_z_hat = beta__delta_z;
  vector[C-1] kappa__delta_z_hat = kappa__delta_z;
}
