data {
  int<lower=1> A;
  int<lower=1> S;
  int<lower=1> E;

  array[E] int<lower=1,upper=4> etype;

  array[E] int<lower=1,upper=S> state;
  array[E] int<lower=0,upper=A> choice;
  array[E] int<lower=0,upper=A> action;
  vector[E] outcome_obs; // unused

  array[E] vector<lower=0,upper=1>[A] avail_mask;

  array[E] int<lower=0,upper=A> demo_action;
  vector[E] demo_outcome_obs;
  array[E] int<lower=0,upper=1> has_demo_outcome;

  real<lower=1e-6> beta_lower;
  real<lower=0> kappa_abs_max;

  // priors
  int<lower=1,upper=8> alpha_o_prior_family;
  real alpha_o_prior_p1;
  real alpha_o_prior_p2;
  real alpha_o_prior_p3;

  int<lower=1,upper=8> alpha_a_prior_family;
  real alpha_a_prior_p1;
  real alpha_a_prior_p2;
  real alpha_a_prior_p3;

  int<lower=1,upper=8> beta_q_prior_family;
  real beta_q_prior_p1;
  real beta_q_prior_p2;
  real beta_q_prior_p3;

  int<lower=1,upper=8> beta_a_prior_family;
  real beta_a_prior_p1;
  real beta_a_prior_p2;
  real beta_a_prior_p3;

  int<lower=1,upper=8> kappa_prior_family;
  real kappa_prior_p1;
  real kappa_prior_p2;
  real kappa_prior_p3;
}
parameters {
  real<lower=0,upper=1> alpha_o;
  real<lower=0,upper=1> alpha_a;
  real<lower=beta_lower> beta_q;
  real<lower=beta_lower> beta_a;
  real<lower=-kappa_abs_max,upper=kappa_abs_max> kappa;
}
model {
  target += prior_lpdf(alpha_o | alpha_o_prior_family, alpha_o_prior_p1, alpha_o_prior_p2, alpha_o_prior_p3);
  target += prior_lpdf(alpha_a | alpha_a_prior_family, alpha_a_prior_p1, alpha_a_prior_p2, alpha_a_prior_p3);
  target += prior_lpdf(beta_q | beta_q_prior_family, beta_q_prior_p1, beta_q_prior_p2, beta_q_prior_p3);
  target += prior_lpdf(beta_a | beta_a_prior_family, beta_a_prior_p1, beta_a_prior_p2, beta_a_prior_p3);
  target += prior_lpdf(kappa | kappa_prior_family, kappa_prior_p1, kappa_prior_p2, kappa_prior_p3);

  matrix[S, A] Q = rep_matrix(0.0, S, A);
  array[S] int last_choice = rep_array(0, S);
  vector[A] demo_pi = rep_vector(1.0 / A, A);

  for (e in 1:E) {
    int s = state[e];

    if (etype[e] == 1) {
      Q = rep_matrix(0.0, S, A);
      last_choice = rep_array(0, S);
      demo_pi = rep_vector(1.0 / A, A);

    } else if (etype[e] == 2) {
      if (demo_action[e] > 0) {
        int a = demo_action[e];
        vector[A] onehot = rep_vector(0.0, A);
        onehot[a] = 1.0;

        demo_pi = demo_pi + alpha_a * (onehot - demo_pi);

        if (has_demo_outcome[e] == 1) {
          real r = demo_outcome_obs[e];
          Q[s,a] = Q[s,a] + alpha_o * (r - Q[s,a]);
        }
      }

    } else if (etype[e] == 3) {
      if (choice[e] > 0) {
        vector[A] g = (demo_pi - (1.0 / A)) / (1.0 - (1.0 / A));
        vector[A] u = beta_q * to_vector(Q[s]') + beta_a * g;
        if (last_choice[s] > 0) u[last_choice[s]] += kappa;
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
    matrix[S, A] Q = rep_matrix(0.0, S, A);
    array[S] int last_choice = rep_array(0, S);
    vector[A] demo_pi = rep_vector(1.0 / A, A);

    for (e in 1:E) {
      int s = state[e];

      if (etype[e] == 1) {
        Q = rep_matrix(0.0, S, A);
        last_choice = rep_array(0, S);
        demo_pi = rep_vector(1.0 / A, A);

      } else if (etype[e] == 2) {
        if (demo_action[e] > 0) {
          int a = demo_action[e];
          vector[A] onehot = rep_vector(0.0, A);
          onehot[a] = 1.0;

          demo_pi = demo_pi + alpha_a * (onehot - demo_pi);

          if (has_demo_outcome[e] == 1) {
            real r = demo_outcome_obs[e];
            Q[s,a] = Q[s,a] + alpha_o * (r - Q[s,a]);
          }
        }

      } else if (etype[e] == 3) {
        if (choice[e] > 0) {
          vector[A] g = (demo_pi - (1.0 / A)) / (1.0 - (1.0 / A));
          vector[A] u = beta_q * to_vector(Q[s]') + beta_a * g;
          if (last_choice[s] > 0) u[last_choice[s]] += kappa;
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
}
