data {
  int<lower=1> A;
  int<lower=1> S;
  int<lower=1> E;
  int<lower=1,upper=4> etype[E];

  int<lower=1,upper=S> state[E];
  int<lower=0,upper=A> choice[E];
  int<lower=0,upper=A> action[E];
  vector[E] outcome_obs; // unused

  int<lower=0,upper=A> demo_action[E];
  vector[E] demo_outcome_obs;
  int<lower=0,upper=1> has_demo_outcome[E];

  real<lower=1e-6> beta_lower;
  real<lower=1e-6> beta_upper;

  // priors
  int<lower=1,upper=8> alpha_o_prior_family;
  real alpha_o_prior_p1; real alpha_o_prior_p2; real alpha_o_prior_p3;

  int<lower=1,upper=8> beta_prior_family;
  real beta_prior_p1; real beta_prior_p2; real beta_prior_p3;
}
parameters {
  real<lower=0,upper=1> alpha_o;
  real<lower=beta_lower,upper=beta_upper> beta;
}
model {
  target += prior_lpdf(alpha_o, alpha_o_prior_family, alpha_o_prior_p1, alpha_o_prior_p2, alpha_o_prior_p3);
  target += prior_lpdf(beta,    beta_prior_family,    beta_prior_p1,    beta_prior_p2,    beta_prior_p3);

  matrix[S, A] Q = rep_matrix(0.0, S, A);

  for (e in 1:E) {
    int s = state[e];

    if (etype[e] == 1) {
      Q = rep_matrix(0.0, S, A);

    } else if (etype[e] == 2) {
      if (demo_action[e] > 0 && has_demo_outcome[e] == 1) {
        int a = demo_action[e];
        real r = demo_outcome_obs[e];
        Q[s,a] = Q[s,a] + alpha_o * (r - Q[s,a]);
      }

    } else if (etype[e] == 3) {
      if (choice[e] > 0) {
        vector[A] u = to_vector(Q[s]');
        target += categorical_logit_lpmf(choice[e] | beta * u);
      }
    }
  }
}
