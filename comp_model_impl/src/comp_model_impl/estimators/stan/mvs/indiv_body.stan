data {
  int<lower=1> A;
  int<lower=1> E;
  array[E] int<lower=1,upper=4> etype;
  array[E] int<lower=0,upper=A> choice;
  array[E] vector<lower=0,upper=1>[A] avail_mask;

  array[E] vector[A] action_mean;
  array[E] vector[A] action_variance;
  array[E] vector[A] action_skewness;

  real<lower=0> lambda_abs_max;
  real<lower=0> delta_abs_max;
  real<lower=1e-6> beta_lower;
  real<lower=1e-6> beta_upper;

  // priors
  int<lower=1,upper=8> lambda_var_prior_family;
  real lambda_var_prior_p1;
  real lambda_var_prior_p2;
  real lambda_var_prior_p3;

  int<lower=1,upper=8> delta_prior_family;
  real delta_prior_p1;
  real delta_prior_p2;
  real delta_prior_p3;

  int<lower=1,upper=8> beta_prior_family;
  real beta_prior_p1;
  real beta_prior_p2;
  real beta_prior_p3;
}

parameters {
  real<lower=-lambda_abs_max, upper=lambda_abs_max> lambda_var;
  real<lower=-delta_abs_max, upper=delta_abs_max> delta;
  real<lower=beta_lower, upper=beta_upper> beta;
}

model {
  target += prior_lpdf(lambda_var | lambda_var_prior_family, lambda_var_prior_p1, lambda_var_prior_p2, lambda_var_prior_p3);
  target += prior_lpdf(delta      | delta_prior_family,      delta_prior_p1,      delta_prior_p2,      delta_prior_p3);
  target += prior_lpdf(beta       | beta_prior_family,       beta_prior_p1,       beta_prior_p2,       beta_prior_p3);

  for (e in 1:E) {
    if (etype[e] == 3 && choice[e] > 0) {
      vector[A] u =
        action_mean[e]
        + lambda_var * action_variance[e]
        + delta * action_skewness[e];

      for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
      target += categorical_logit_lpmf(choice[e] | beta * u);
    }
  }
}

generated quantities {
  vector[E] log_lik = rep_vector(0.0, E);
  for (e in 1:E) {
    if (etype[e] == 3 && choice[e] > 0) {
      vector[A] u =
        action_mean[e]
        + lambda_var * action_variance[e]
        + delta * action_skewness[e];
      for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
      log_lik[e] = categorical_logit_lpmf(choice[e] | beta * u);
    }
  }
}
