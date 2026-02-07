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
  real<lower=1e-6> beta_upper;
  real<lower=0> kappa_abs_max;
  real<lower=0> demo_bias_abs_max;

  // priors
  int<lower=1,upper=8> alpha_o_prior_family;
  real alpha_o_prior_p1;
  real alpha_o_prior_p2;
  real alpha_o_prior_p3;

  int<lower=1,upper=8> demo_bias_prior_family;
  real demo_bias_prior_p1;
  real demo_bias_prior_p2;
  real demo_bias_prior_p3;

  int<lower=1,upper=8> beta_prior_family;
  real beta_prior_p1;
  real beta_prior_p2;
  real beta_prior_p3;

  int<lower=1,upper=8> kappa_prior_family;
  real kappa_prior_p1;
  real kappa_prior_p2;
  real kappa_prior_p3;
}
parameters {
  real<lower=0,upper=1> alpha_o;
  real<lower=-demo_bias_abs_max,upper=demo_bias_abs_max> demo_bias;
  real<lower=beta_lower,upper=beta_upper> beta;
  real<lower=-kappa_abs_max,upper=kappa_abs_max> kappa;
}
model {
  target += prior_lpdf(alpha_o | alpha_o_prior_family, alpha_o_prior_p1, alpha_o_prior_p2, alpha_o_prior_p3);
  target += prior_lpdf(
    demo_bias | demo_bias_prior_family, demo_bias_prior_p1, demo_bias_prior_p2, demo_bias_prior_p3
  );
  target += prior_lpdf(beta | beta_prior_family, beta_prior_p1, beta_prior_p2, beta_prior_p3);
  target += prior_lpdf(kappa | kappa_prior_family, kappa_prior_p1, kappa_prior_p2, kappa_prior_p3);

  matrix[S, A] Q = rep_matrix(0.0, S, A);
  array[S] int last_choice = rep_array(0, S);
  int recent_demo_choice = 0;

  for (e in 1:E) {
    int s = state[e];

    if (etype[e] == 1) {
      Q = rep_matrix(0.0, S, A);
      last_choice = rep_array(0, S);
      recent_demo_choice = 0;

    } else if (etype[e] == 2) {
      if (demo_action[e] > 0) {
        int a = demo_action[e];
        recent_demo_choice = a;

        if (has_demo_outcome[e] == 1) {
          real r = demo_outcome_obs[e];
          Q[s,a] = Q[s,a] + alpha_o * (r - Q[s,a]);
        }
      }

    } else if (etype[e] == 3) {
      if (choice[e] > 0) {
        vector[A] u = beta * to_vector(Q[s]');
        if (last_choice[s] > 0) u[last_choice[s]] += kappa;
        if (recent_demo_choice > 0) u[recent_demo_choice] += demo_bias;
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
  real log_lik = 0.0;
  matrix[S, A] Q = rep_matrix(0.0, S, A);
  array[S] int last_choice = rep_array(0, S);
  int recent_demo_choice = 0;

  for (e in 1:E) {
    int s = state[e];

    if (etype[e] == 1) {
      Q = rep_matrix(0.0, S, A);
      last_choice = rep_array(0, S);
      recent_demo_choice = 0;

    } else if (etype[e] == 2) {
      if (demo_action[e] > 0) {
        int a = demo_action[e];
        recent_demo_choice = a;

        if (has_demo_outcome[e] == 1) {
          real r = demo_outcome_obs[e];
          Q[s,a] = Q[s,a] + alpha_o * (r - Q[s,a]);
        }
      }

    } else if (etype[e] == 3) {
      if (choice[e] > 0) {
        vector[A] u = beta * to_vector(Q[s]');
        if (last_choice[s] > 0) u[last_choice[s]] += kappa;
        if (recent_demo_choice > 0) u[recent_demo_choice] += demo_bias;
        for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
        log_lik += categorical_logit_lpmf(choice[e] | u);
      }

    } else if (etype[e] == 4) {
      if (action[e] > 0) {
        last_choice[s] = action[e];
      }
    }
  }
}
