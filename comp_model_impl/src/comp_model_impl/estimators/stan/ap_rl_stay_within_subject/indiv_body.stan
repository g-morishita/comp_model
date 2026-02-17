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
  vector[E] demo_outcome_obs; // unused
  array[E] int<lower=0, upper=1> has_demo_outcome; // unused

  array[E] int<lower=1, upper=C> cond;

  real<lower=1e-6> beta_lower;
  real<lower=0> kappa_abs_max;

  // priors on z-scale
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

  int<lower=1, upper=8> kappa__shared_prior_family;
  real kappa__shared_prior_p1;
  real kappa__shared_prior_p2;
  real kappa__shared_prior_p3;

  int<lower=1, upper=8> kappa__delta_prior_family;
  real kappa__delta_prior_p1;
  real kappa__delta_prior_p2;
  real kappa__delta_prior_p3;
}
parameters {
  real alpha_a__shared_z;
  vector[C-1] alpha_a__delta_z;

  real beta__shared_z;
  vector[C-1] beta__delta_z;

  real kappa__shared_z;
  vector[C-1] kappa__delta_z;
}
transformed parameters {
  vector[C] alpha_a_z;
  vector[C] beta_z;
  vector[C] kappa_z;

  for (c in 1:C) {
    if (c == baseline_cond) {
      alpha_a_z[c] = alpha_a__shared_z;
      beta_z[c] = beta__shared_z;
      kappa_z[c] = kappa__shared_z;
    } else {
      int idx = (c < baseline_cond) ? c : (c - 1);
      alpha_a_z[c] = alpha_a__shared_z + alpha_a__delta_z[idx];
      beta_z[c] = beta__shared_z + beta__delta_z[idx];
      kappa_z[c] = kappa__shared_z + kappa__delta_z[idx];
    }
  }

  vector<lower=0, upper=1>[C] alpha_a = inv_logit(alpha_a_z);
  vector<lower=beta_lower>[C] beta =
    beta_lower + exp(beta_z);
  vector<lower=-kappa_abs_max, upper=kappa_abs_max>[C] kappa =
    kappa_abs_max * tanh(kappa_z);
}
model {
  target += prior_lpdf(alpha_a__shared_z | alpha_a__shared_prior_family, alpha_a__shared_prior_p1, alpha_a__shared_prior_p2, alpha_a__shared_prior_p3);
  target += prior_lpdf(beta__shared_z | beta__shared_prior_family, beta__shared_prior_p1, beta__shared_prior_p2, beta__shared_prior_p3);
  target += prior_lpdf(kappa__shared_z | kappa__shared_prior_family, kappa__shared_prior_p1, kappa__shared_prior_p2, kappa__shared_prior_p3);

  for (i in 1:(C - 1)) {
    target += prior_lpdf(alpha_a__delta_z[i] | alpha_a__delta_prior_family, alpha_a__delta_prior_p1, alpha_a__delta_prior_p2, alpha_a__delta_prior_p3);
    target += prior_lpdf(beta__delta_z[i] | beta__delta_prior_family, beta__delta_prior_p1, beta__delta_prior_p2, beta__delta_prior_p3);
    target += prior_lpdf(kappa__delta_z[i] | kappa__delta_prior_family, kappa__delta_prior_p1, kappa__delta_prior_p2, kappa__delta_prior_p3);
  }

  array[S] int last_choice = rep_array(0, S);
  vector[A] demo_pi = rep_vector(1.0 / A, A);

  for (e in 1:E) {
    int s = state[e];
    int c = cond[e];

    if (etype[e] == 1) {
      last_choice = rep_array(0, S);
      demo_pi = rep_vector(1.0 / A, A);

    } else if (etype[e] == 2) {
      if (demo_action[e] > 0) {
        int a = demo_action[e];
        vector[A] onehot = rep_vector(0.0, A);
        onehot[a] = 1.0;
        demo_pi = demo_pi + alpha_a[c] * (onehot - demo_pi);
      }

    } else if (etype[e] == 3) {
      if (choice[e] > 0) {
        vector[A] g = (demo_pi - (1.0 / A)) / (1.0 - (1.0 / A));
        vector[A] u = beta[c] * g;
        if (last_choice[s] > 0) u[last_choice[s]] += kappa[c];
        for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
        target += categorical_logit_lpmf(choice[e] | u);
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
    array[S] int last_choice = rep_array(0, S);
    vector[A] demo_pi = rep_vector(1.0 / A, A);

    for (e in 1:E) {
      int s = state[e];
      int c = cond[e];

      if (etype[e] == 1) {
        last_choice = rep_array(0, S);
        demo_pi = rep_vector(1.0 / A, A);

      } else if (etype[e] == 2) {
        if (demo_action[e] > 0) {
          int a = demo_action[e];
          vector[A] onehot = rep_vector(0.0, A);
          onehot[a] = 1.0;
          demo_pi = demo_pi + alpha_a[c] * (onehot - demo_pi);
        }

      } else if (etype[e] == 3) {
        if (choice[e] > 0) {
          vector[A] g = (demo_pi - (1.0 / A)) / (1.0 - (1.0 / A));
          vector[A] u = beta[c] * g;
          if (last_choice[s] > 0) u[last_choice[s]] += kappa[c];
          for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
          log_lik[e] = categorical_logit_lpmf(choice[e] | u);
        }

      } else if (etype[e] == 4) {
        if (action[e] > 0) {
          last_choice[s] = action[e];
        }
      }
    }
  }

  vector[C] alpha_a_hat = alpha_a;
  vector[C] beta_hat = beta;
  vector[C] kappa_hat = kappa;

  real alpha_a__shared_z_hat = alpha_a__shared_z;
  real beta__shared_z_hat = beta__shared_z;
  real kappa__shared_z_hat = kappa__shared_z;

  vector[C-1] alpha_a__delta_z_hat = alpha_a__delta_z;
  vector[C-1] beta__delta_z_hat = beta__delta_z;
  vector[C-1] kappa__delta_z_hat = kappa__delta_z;
}
