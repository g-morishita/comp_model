data {
  int<lower=1> A;
  int<lower=1> S;
  int<lower=1> E;

  array[E] int<lower=1,upper=4> etype;

  array[E] int<lower=1,upper=S> state;
  array[E] int<lower=0,upper=A> choice;
  array[E] int<lower=0,upper=A> action;
  vector[E] outcome_obs;

  array[E] vector<lower=0,upper=1>[A] avail_mask;

  real<lower=1e-6> beta_lower;

  // priors
  int<lower=1,upper=8> alpha_prior_family;
  real alpha_prior_p1;
  real alpha_prior_p2;
  real alpha_prior_p3;

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
  real<lower=0,upper=1> alpha;
  real<lower=beta_lower> beta;
  real kappa;
}
model {
  target += prior_lpdf(alpha | alpha_prior_family, alpha_prior_p1, alpha_prior_p2, alpha_prior_p3);
  target += prior_lpdf(beta | beta_prior_family, beta_prior_p1, beta_prior_p2, beta_prior_p3);
  target += prior_lpdf(kappa | kappa_prior_family, kappa_prior_p1, kappa_prior_p2, kappa_prior_p3);

  matrix[S, A] Q = rep_matrix(0.0, S, A);
  array[S] int last_choice = rep_array(0, S);

  for (e in 1:E) {
    int s = state[e];

    if (etype[e] == 1) {
      Q = rep_matrix(0.0, S, A);
      last_choice = rep_array(0, S);

    } else if (etype[e] == 3) {
      if (choice[e] > 0) {
        vector[A] u = beta * to_vector(Q[s]');
        if (last_choice[s] > 0) u[last_choice[s]] += kappa;
        for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
        target += categorical_logit_lpmf(choice[e] | u);
      }

    } else if (etype[e] == 4) {
      if (action[e] > 0) {
        int a = action[e];
        real r = outcome_obs[e];
        Q[s, a] = Q[s, a] + alpha * (r - Q[s, a]);
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

      } else if (etype[e] == 3) {
        if (choice[e] > 0) {
          vector[A] u = beta * to_vector(Q[s]');
          if (last_choice[s] > 0) u[last_choice[s]] += kappa;
          for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
          log_lik[e] = categorical_logit_lpmf(choice[e] | u);
        }

      } else if (etype[e] == 4) {
        if (action[e] > 0) {
          int a = action[e];
          real r = outcome_obs[e];
          Q[s, a] = Q[s, a] + alpha * (r - Q[s, a]);
          last_choice[s] = a;
        }
      }
    }
  }
}
