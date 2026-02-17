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

  array[E] vector<lower=0, upper=1>[A] avail_mask;

  array[E] int<lower=0, upper=A> demo_action; // unused
  vector[E] demo_outcome_obs; // unused
  array[E] int<lower=0, upper=1> has_demo_outcome; // unused

  array[E] int<lower=1, upper=C> cond;

  real<lower=1e-6> beta_lower;

  // priors on z-scale
  int<lower=1, upper=8> alpha__shared_prior_family;
  real alpha__shared_prior_p1;
  real alpha__shared_prior_p2;
  real alpha__shared_prior_p3;

  int<lower=1, upper=8> alpha__delta_prior_family;
  real alpha__delta_prior_p1;
  real alpha__delta_prior_p2;
  real alpha__delta_prior_p3;

  int<lower=1, upper=8> beta__shared_prior_family;
  real beta__shared_prior_p1;
  real beta__shared_prior_p2;
  real beta__shared_prior_p3;

  int<lower=1, upper=8> beta__delta_prior_family;
  real beta__delta_prior_p1;
  real beta__delta_prior_p2;
  real beta__delta_prior_p3;
}
parameters {
  real alpha__shared_z;
  vector[C-1] alpha__delta_z;

  real beta__shared_z;
  vector[C-1] beta__delta_z;
}
transformed parameters {
  vector[C] alpha_z;
  vector[C] beta_z;

  for (c in 1:C) {
    if (c == baseline_cond) {
      alpha_z[c] = alpha__shared_z;
      beta_z[c] = beta__shared_z;
    } else {
      int idx = (c < baseline_cond) ? c : (c - 1);
      alpha_z[c] = alpha__shared_z + alpha__delta_z[idx];
      beta_z[c] = beta__shared_z + beta__delta_z[idx];
    }
  }

  vector<lower=0, upper=1>[C] alpha = inv_logit(alpha_z);
  vector<lower=beta_lower>[C] beta =
    beta_lower + exp(beta_z);
}
model {
  target += prior_lpdf(alpha__shared_z | alpha__shared_prior_family, alpha__shared_prior_p1, alpha__shared_prior_p2, alpha__shared_prior_p3);
  target += prior_lpdf(beta__shared_z | beta__shared_prior_family, beta__shared_prior_p1, beta__shared_prior_p2, beta__shared_prior_p3);

  for (i in 1:(C - 1)) {
    target += prior_lpdf(alpha__delta_z[i] | alpha__delta_prior_family, alpha__delta_prior_p1, alpha__delta_prior_p2, alpha__delta_prior_p3);
    target += prior_lpdf(beta__delta_z[i] | beta__delta_prior_family, beta__delta_prior_p1, beta__delta_prior_p2, beta__delta_prior_p3);
  }

  matrix[S, A] Q = rep_matrix(0.0, S, A);

  for (e in 1:E) {
    int s = state[e];
    int c = cond[e];

    if (etype[e] == 1) {
      Q = rep_matrix(0.0, S, A);

    } else if (etype[e] == 3) {
      if (choice[e] > 0) {
        vector[A] u = to_vector(Q[s]');
        for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
        target += categorical_logit_lpmf(choice[e] | beta[c] * u);
      }

    } else if (etype[e] == 4) {
      if (action[e] > 0) {
        int a = action[e];
        real r = outcome_obs[e];
        Q[s, a] = Q[s, a] + alpha[c] * (r - Q[s, a]);
      }
    }
  }
}
generated quantities {
  vector[E] log_lik = rep_vector(0.0, E);
  {
    matrix[S, A] Q = rep_matrix(0.0, S, A);

    for (e in 1:E) {
      int s = state[e];
      int c = cond[e];

      if (etype[e] == 1) {
        Q = rep_matrix(0.0, S, A);

      } else if (etype[e] == 3) {
        if (choice[e] > 0) {
          vector[A] u = to_vector(Q[s]');
          for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
          log_lik[e] = categorical_logit_lpmf(choice[e] | beta[c] * u);
        }

      } else if (etype[e] == 4) {
        if (action[e] > 0) {
          int a = action[e];
          real r = outcome_obs[e];
          Q[s, a] = Q[s, a] + alpha[c] * (r - Q[s, a]);
        }
      }
    }
  }

  vector[C] alpha_hat = alpha;
  vector[C] beta_hat = beta;

  real alpha__shared_z_hat = alpha__shared_z;
  real beta__shared_z_hat = beta__shared_z;

  vector[C-1] alpha__delta_z_hat = alpha__delta_z;
  vector[C-1] beta__delta_z_hat = beta__delta_z;
}

