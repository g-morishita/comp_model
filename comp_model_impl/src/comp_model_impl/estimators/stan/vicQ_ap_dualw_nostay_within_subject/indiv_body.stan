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
  vector[E] outcome_obs; // unused

  array[E] vector<lower=0,upper=1>[A] avail_mask;

  array[E] int<lower=0, upper=A> demo_action;
  vector[E] demo_outcome_obs;
  array[E] int<lower=0, upper=1> has_demo_outcome;

  array[E] int<lower=1, upper=C> cond;

  real<lower=1e-6> beta_lower;
  real<lower=1e-6> beta_upper;

  // priors on z-scale (configurable)
  int<lower=1, upper=8> alpha_o__shared_prior_family;
  real alpha_o__shared_prior_p1;
  real alpha_o__shared_prior_p2;
  real alpha_o__shared_prior_p3;

  int<lower=1, upper=8> alpha_o__delta_prior_family;
  real alpha_o__delta_prior_p1;
  real alpha_o__delta_prior_p2;
  real alpha_o__delta_prior_p3;

  int<lower=1, upper=8> alpha_a__shared_prior_family;
  real alpha_a__shared_prior_p1;
  real alpha_a__shared_prior_p2;
  real alpha_a__shared_prior_p3;

  int<lower=1, upper=8> alpha_a__delta_prior_family;
  real alpha_a__delta_prior_p1;
  real alpha_a__delta_prior_p2;
  real alpha_a__delta_prior_p3;

  int<lower=1, upper=8> beta__shared_prior_family;
  real beta__shared_prior_p1;
  real beta__shared_prior_p2;
  real beta__shared_prior_p3;

  int<lower=1, upper=8> beta__delta_prior_family;
  real beta__delta_prior_p1;
  real beta__delta_prior_p2;
  real beta__delta_prior_p3;

  int<lower=1, upper=8> w__shared_prior_family;
  real w__shared_prior_p1;
  real w__shared_prior_p2;
  real w__shared_prior_p3;

  int<lower=1, upper=8> w__delta_prior_family;
  real w__delta_prior_p1;
  real w__delta_prior_p2;
  real w__delta_prior_p3;
}
parameters {
  real alpha_o__shared_z;
  vector[C-1] alpha_o__delta_z;

  real alpha_a__shared_z;
  vector[C-1] alpha_a__delta_z;

  real beta__shared_z;
  vector[C-1] beta__delta_z;

  real w__shared_z;
  vector[C-1] w__delta_z;
}
transformed parameters {
  vector[C] alpha_o_z;
  vector[C] alpha_a_z;
  vector[C] beta_z;
  vector[C] w_z;

  for (c in 1:C) {
    if (c == baseline_cond) {
      alpha_o_z[c] = alpha_o__shared_z;
      alpha_a_z[c] = alpha_a__shared_z;
      beta_z[c] = beta__shared_z;
      w_z[c] = w__shared_z;
    } else {
      int idx = (c < baseline_cond) ? c : (c - 1);
      alpha_o_z[c] = alpha_o__shared_z + alpha_o__delta_z[idx];
      alpha_a_z[c] = alpha_a__shared_z + alpha_a__delta_z[idx];
      beta_z[c] = beta__shared_z + beta__delta_z[idx];
      w_z[c] = w__shared_z + w__delta_z[idx];
    }
  }

  vector<lower=0, upper=1>[C] alpha_o = inv_logit(alpha_o_z);
  vector<lower=0, upper=1>[C] alpha_a = inv_logit(alpha_a_z);
  vector<lower=beta_lower, upper=beta_upper>[C] beta =
    beta_lower + (beta_upper - beta_lower) * (tanh(beta_z) + 1) * 0.5;
  vector<lower=0, upper=1>[C] w = inv_logit(w_z);
}
model {
  target += prior_lpdf(alpha_o__shared_z | alpha_o__shared_prior_family, alpha_o__shared_prior_p1, alpha_o__shared_prior_p2, alpha_o__shared_prior_p3);
  target += prior_lpdf(alpha_a__shared_z | alpha_a__shared_prior_family, alpha_a__shared_prior_p1, alpha_a__shared_prior_p2, alpha_a__shared_prior_p3);
  target += prior_lpdf(beta__shared_z | beta__shared_prior_family, beta__shared_prior_p1, beta__shared_prior_p2, beta__shared_prior_p3);
  target += prior_lpdf(w__shared_z | w__shared_prior_family, w__shared_prior_p1, w__shared_prior_p2, w__shared_prior_p3);

  for (i in 1:(C - 1)) {
    target += prior_lpdf(alpha_o__delta_z[i] | alpha_o__delta_prior_family, alpha_o__delta_prior_p1, alpha_o__delta_prior_p2, alpha_o__delta_prior_p3);
    target += prior_lpdf(alpha_a__delta_z[i] | alpha_a__delta_prior_family, alpha_a__delta_prior_p1, alpha_a__delta_prior_p2, alpha_a__delta_prior_p3);
    target += prior_lpdf(beta__delta_z[i] | beta__delta_prior_family, beta__delta_prior_p1, beta__delta_prior_p2, beta__delta_prior_p3);
    target += prior_lpdf(w__delta_z[i] | w__delta_prior_family, w__delta_prior_p1, w__delta_prior_p2, w__delta_prior_p3);
  }

  matrix[S, A] Q = rep_matrix(0.0, S, A);
  vector[A] demo_pi = rep_vector(1.0 / A, A);

  for (e in 1:E) {
    int s = state[e];
    int c = cond[e];

    if (etype[e] == 1) {
      Q = rep_matrix(0.0, S, A);
      demo_pi = rep_vector(1.0 / A, A);

    } else if (etype[e] == 2) {
      if (demo_action[e] > 0) {
        int a = demo_action[e];
        vector[A] onehot = rep_vector(0.0, A);
        onehot[a] = 1.0;
        demo_pi = demo_pi + alpha_a[c] * (onehot - demo_pi);

        if (has_demo_outcome[e] == 1) {
          real r = demo_outcome_obs[e];
          Q[s, a] = Q[s, a] + alpha_o[c] * (r - Q[s, a]);
        }
      }

    } else if (etype[e] == 3) {
      if (choice[e] > 0) {
        vector[A] g = (demo_pi - (1.0 / A)) / (1.0 - (1.0 / A));
        vector[A] social_drive = w[c] * to_vector(Q[s]') + (1.0 - w[c]) * g;
        vector[A] u = beta[c] * social_drive;
        for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
        target += categorical_logit_lpmf(choice[e] | u);
      }
    }
  }
}
generated quantities {
  vector[E] log_lik = rep_vector(0.0, E);
  {
    matrix[S, A] Q = rep_matrix(0.0, S, A);
    vector[A] demo_pi = rep_vector(1.0 / A, A);

    for (e in 1:E) {
      int s = state[e];
      int c = cond[e];

      if (etype[e] == 1) {
        Q = rep_matrix(0.0, S, A);
        demo_pi = rep_vector(1.0 / A, A);

      } else if (etype[e] == 2) {
        if (demo_action[e] > 0) {
          int a = demo_action[e];
          vector[A] onehot = rep_vector(0.0, A);
          onehot[a] = 1.0;
          demo_pi = demo_pi + alpha_a[c] * (onehot - demo_pi);

          if (has_demo_outcome[e] == 1) {
            real r = demo_outcome_obs[e];
            Q[s, a] = Q[s, a] + alpha_o[c] * (r - Q[s, a]);
          }
        }

      } else if (etype[e] == 3) {
        if (choice[e] > 0) {
          vector[A] g = (demo_pi - (1.0 / A)) / (1.0 - (1.0 / A));
          vector[A] social_drive = w[c] * to_vector(Q[s]') + (1.0 - w[c]) * g;
          vector[A] u = beta[c] * social_drive;
          for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
          log_lik[e] = categorical_logit_lpmf(choice[e] | u);
        }
      }
    }
  }

  vector[C] alpha_o_hat = alpha_o;
  vector[C] alpha_a_hat = alpha_a;
  vector[C] beta_hat = beta;
  vector[C] w_hat = w;

  real alpha_o__shared_z_hat = alpha_o__shared_z;
  real alpha_a__shared_z_hat = alpha_a__shared_z;
  real beta__shared_z_hat = beta__shared_z;
  real w__shared_z_hat = w__shared_z;

  vector[C-1] alpha_o__delta_z_hat = alpha_o__delta_z;
  vector[C-1] alpha_a__delta_z_hat = alpha_a__delta_z;
  vector[C-1] beta__delta_z_hat = beta__delta_z;
  vector[C-1] w__delta_z_hat = w__delta_z;
}
