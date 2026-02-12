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
  vector[E] demo_outcome_obs; // unused
  array[E] int<lower=0,upper=1> has_demo_outcome; // unused

  real<lower=1e-6> beta_lower;
  real<lower=1e-6> beta_upper;

  // priors
  int<lower=1,upper=8> alpha_a_prior_family;
  real alpha_a_prior_p1;
  real alpha_a_prior_p2;
  real alpha_a_prior_p3;

  int<lower=1,upper=8> beta_prior_family;
  real beta_prior_p1;
  real beta_prior_p2;
  real beta_prior_p3;
}
parameters {
  real<lower=0,upper=1> alpha_a;
  real<lower=beta_lower,upper=beta_upper> beta;
}
model {
  target += prior_lpdf(alpha_a | alpha_a_prior_family, alpha_a_prior_p1, alpha_a_prior_p2, alpha_a_prior_p3);
  target += prior_lpdf(beta | beta_prior_family, beta_prior_p1, beta_prior_p2, beta_prior_p3);

  vector[A] demo_pi = rep_vector(1.0 / A, A);

  for (e in 1:E) {
    if (etype[e] == 1) {
      demo_pi = rep_vector(1.0 / A, A);

    } else if (etype[e] == 2) {
      if (demo_action[e] > 0) {
        int a = demo_action[e];
        vector[A] onehot = rep_vector(0.0, A);
        onehot[a] = 1.0;
        demo_pi = demo_pi + alpha_a * (onehot - demo_pi);
      }

    } else if (etype[e] == 3) {
      if (choice[e] > 0) {
        vector[A] g = (demo_pi - (1.0 / A)) / (1.0 - (1.0 / A));
        vector[A] u = beta * g;
        for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
        target += categorical_logit_lpmf(choice[e] | u);
      }
    }
  }
}

generated quantities {
  vector[E] log_lik = rep_vector(0.0, E);
  {
    vector[A] demo_pi = rep_vector(1.0 / A, A);

    for (e in 1:E) {
      if (etype[e] == 1) {
        demo_pi = rep_vector(1.0 / A, A);

      } else if (etype[e] == 2) {
        if (demo_action[e] > 0) {
          int a = demo_action[e];
          vector[A] onehot = rep_vector(0.0, A);
          onehot[a] = 1.0;
          demo_pi = demo_pi + alpha_a * (onehot - demo_pi);
        }

      } else if (etype[e] == 3) {
        if (choice[e] > 0) {
          vector[A] g = (demo_pi - (1.0 / A)) / (1.0 - (1.0 / A));
          vector[A] u = beta * g;
          for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
          log_lik[e] = categorical_logit_lpmf(choice[e] | u);
        }
      }
    }
  }
}
