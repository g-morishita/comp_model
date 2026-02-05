data {
  int<lower=1> A;
  int<lower=1> S;
  int<lower=1> E;
  array[E] int<lower=1,upper=4> etype;

  array[E] int<lower=1,upper=S> state;
  array[E] int<lower=0,upper=A> choice;
  array[E] int<lower=0,upper=A> action;
  vector[E] outcome_obs;

  real<lower=1e-6> beta_lower;
  real<lower=1e-6> beta_upper;

  // priors
  int<lower=1,upper=8> alpha_prior_family;
  real alpha_prior_p1;
  real alpha_prior_p2;
  real alpha_prior_p3;

  int<lower=1,upper=8> beta_prior_family;
  real beta_prior_p1;
  real beta_prior_p2;
  real beta_prior_p3;
}
parameters {
  real<lower=0,upper=1> alpha;
  real<lower=beta_lower,upper=beta_upper> beta;
}
model {
  target += prior_lpdf(alpha | alpha_prior_family, alpha_prior_p1, alpha_prior_p2, alpha_prior_p3);
  target += prior_lpdf(beta  | beta_prior_family,  beta_prior_p1,  beta_prior_p2,  beta_prior_p3);

  matrix[S, A] Q = rep_matrix(0.0, S, A);

  for (e in 1:E) {
    int s = state[e];

    if (etype[e] == 1) {
      Q = rep_matrix(0.0, S, A);

    } else if (etype[e] == 3) {
      if (choice[e] > 0) {
        vector[A] u = to_vector(Q[s]');
        target += categorical_logit_lpmf(choice[e] | beta * u);
      }

    } else if (etype[e] == 4) {
      if (action[e] > 0) {
        int a = action[e];
        real r = outcome_obs[e];
        Q[s,a] = Q[s,a] + alpha * (r - Q[s,a]);
      }
    }
  }
}
