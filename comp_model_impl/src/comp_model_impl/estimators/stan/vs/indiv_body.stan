data {
  int<lower=1> A;
  int<lower=1> S;
  int<lower=1> E;
  int<lower=1,upper=4> etype[E];

  int<lower=1,upper=S> state[E];
  int<lower=0,upper=A> choice[E];
  int<lower=0,upper=A> action[E];
  vector[E] outcome_obs;

  vector<lower=0,upper=1>[A] avail_mask[E];

  int<lower=0,upper=A> demo_action[E];
  vector[E] demo_outcome_obs;
  int<lower=0,upper=1> has_demo_outcome[E];

  real pseudo_reward;
  real<lower=1e-6> beta_lower;
  real<lower=1e-6> beta_upper;
  real<lower=0> kappa_abs_max;

  // priors (configurable)
  int<lower=1,upper=8> alpha_p_prior_family;
  real alpha_p_prior_p1;
  real alpha_p_prior_p2;
  real alpha_p_prior_p3;

  int<lower=1,upper=8> alpha_i_prior_family;
  real alpha_i_prior_p1;
  real alpha_i_prior_p2;
  real alpha_i_prior_p3;

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
  real<lower=0,upper=1> alpha_p;
  real<lower=0,upper=1> alpha_i;
  real<lower=beta_lower,upper=beta_upper> beta;
  real<lower=-kappa_abs_max,upper=kappa_abs_max> kappa;
}
model {
  target += prior_lpdf(alpha_p, alpha_p_prior_family, alpha_p_prior_p1, alpha_p_prior_p2, alpha_p_prior_p3);
  target += prior_lpdf(alpha_i, alpha_i_prior_family, alpha_i_prior_p1, alpha_i_prior_p2, alpha_i_prior_p3);
  target += prior_lpdf(beta,    beta_prior_family,    beta_prior_p1,    beta_prior_p2,    beta_prior_p3);
  target += prior_lpdf(kappa,   kappa_prior_family,   kappa_prior_p1,   kappa_prior_p2,   kappa_prior_p3);

  matrix[S, A] Q = rep_matrix(0.0, S, A);
  array[S] int last_choice = rep_array(0, S);

  for (e in 1:E) {
    int s = state[e];

    if (etype[e] == 1) {
      Q = rep_matrix(0.0, S, A);
      last_choice = rep_array(0, S);

    } else if (etype[e] == 2) {
      if (demo_action[e] > 0) {
        int a = demo_action[e];
        Q[s,a] = Q[s,a] + alpha_i * (pseudo_reward - Q[s,a]);
      }

    } else if (etype[e] == 3) {
      if (choice[e] > 0) {
        vector[A] u = to_vector(Q[s]');
        if (last_choice[s] > 0) u[last_choice[s]] += kappa;
        for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
        target += categorical_logit_lpmf(choice[e] | beta * u);
      }

    } else if (etype[e] == 4) {
      if (action[e] > 0) {
        int a = action[e];
        real r = outcome_obs[e];
        Q[s,a] = Q[s,a] + alpha_p * (r - Q[s,a]);
        last_choice[s] = a;
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
  
      if (etype[e] == 1) {
        Q = rep_matrix(0.0, S, A);
        last_choice = rep_array(0, S);
  
      } else if (etype[e] == 2) {
        if (demo_action[e] > 0) {
          int a = demo_action[e];
          Q[s,a] = Q[s,a] + alpha_i * (pseudo_reward - Q[s,a]);
        }
  
      } else if (etype[e] == 3) {
        if (choice[e] > 0) {
          vector[A] u = to_vector(Q[s]');
          if (last_choice[s] > 0) u[last_choice[s]] += kappa;
          for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
          log_lik[e] = categorical_logit_lpmf(choice[e] | beta * u);
        }
  
      } else if (etype[e] == 4) {
        if (action[e] > 0) {
          int a = action[e];
          real r = outcome_obs[e];
          Q[s,a] = Q[s,a] + alpha_p * (r - Q[s,a]);
          last_choice[s] = a;
        }
      }
    }
  }
}
