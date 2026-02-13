data {
  int<lower=1> N;
  int<lower=1> A;
  int<lower=1> E;
  array[E] int<lower=1,upper=N> subj;
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

  // hyperpriors
  int<lower=1,upper=8> mu_lambda_var_prior_family;
  real mu_lambda_var_prior_p1;
  real mu_lambda_var_prior_p2;
  real mu_lambda_var_prior_p3;

  int<lower=1,upper=8> sd_lambda_var_prior_family;
  real sd_lambda_var_prior_p1;
  real sd_lambda_var_prior_p2;
  real sd_lambda_var_prior_p3;

  int<lower=1,upper=8> mu_delta_prior_family;
  real mu_delta_prior_p1;
  real mu_delta_prior_p2;
  real mu_delta_prior_p3;

  int<lower=1,upper=8> sd_delta_prior_family;
  real sd_delta_prior_p1;
  real sd_delta_prior_p2;
  real sd_delta_prior_p3;

  int<lower=1,upper=8> mu_beta_prior_family;
  real mu_beta_prior_p1;
  real mu_beta_prior_p2;
  real mu_beta_prior_p3;

  int<lower=1,upper=8> sd_beta_prior_family;
  real sd_beta_prior_p1;
  real sd_beta_prior_p2;
  real sd_beta_prior_p3;
}

parameters {
  real mu_lambda_var;
  real<lower=0> sd_lambda_var;
  vector[N] z_lambda_var;

  real mu_delta;
  real<lower=0> sd_delta;
  vector[N] z_delta;

  real mu_beta;
  real<lower=0> sd_beta;
  vector[N] z_beta;
}

transformed parameters {
  vector<lower=-lambda_abs_max, upper=lambda_abs_max>[N] lambda_var =
    lambda_abs_max * tanh(mu_lambda_var + sd_lambda_var * z_lambda_var);

  vector<lower=-delta_abs_max, upper=delta_abs_max>[N] delta =
    delta_abs_max * tanh(mu_delta + sd_delta * z_delta);

  vector<lower=beta_lower, upper=beta_upper>[N] beta =
    beta_lower + (beta_upper - beta_lower) * inv_logit(mu_beta + sd_beta * z_beta);
}

model {
  z_lambda_var ~ normal(0, 1);
  z_delta ~ normal(0, 1);
  z_beta ~ normal(0, 1);

  target += prior_lpdf(mu_lambda_var | mu_lambda_var_prior_family, mu_lambda_var_prior_p1, mu_lambda_var_prior_p2, mu_lambda_var_prior_p3);
  target += prior_lpdf(sd_lambda_var | sd_lambda_var_prior_family, sd_lambda_var_prior_p1, sd_lambda_var_prior_p2, sd_lambda_var_prior_p3);

  target += prior_lpdf(mu_delta | mu_delta_prior_family, mu_delta_prior_p1, mu_delta_prior_p2, mu_delta_prior_p3);
  target += prior_lpdf(sd_delta | sd_delta_prior_family, sd_delta_prior_p1, sd_delta_prior_p2, sd_delta_prior_p3);

  target += prior_lpdf(mu_beta | mu_beta_prior_family, mu_beta_prior_p1, mu_beta_prior_p2, mu_beta_prior_p3);
  target += prior_lpdf(sd_beta | sd_beta_prior_family, sd_beta_prior_p1, sd_beta_prior_p2, sd_beta_prior_p3);

  for (e in 1:E) {
    int n = subj[e];
    if (etype[e] == 3 && choice[e] > 0) {
      vector[A] u =
        action_mean[e]
        + lambda_var[n] * action_variance[e]
        + delta[n] * action_skewness[e];

      for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
      target += categorical_logit_lpmf(choice[e] | beta[n] * u);
    }
  }
}

generated quantities {
  vector[E] log_lik = rep_vector(0.0, E);
  for (e in 1:E) {
    int n = subj[e];
    if (etype[e] == 3 && choice[e] > 0) {
      vector[A] u =
        action_mean[e]
        + lambda_var[n] * action_variance[e]
        + delta[n] * action_skewness[e];
      for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
      log_lik[e] = categorical_logit_lpmf(choice[e] | beta[n] * u);
    }
  }

  real lambda_var_pop = lambda_abs_max * tanh(mu_lambda_var);
  real delta_pop = delta_abs_max * tanh(mu_delta);
  real beta_pop = beta_lower + (beta_upper - beta_lower) * inv_logit(mu_beta);

  real mu_lambda_var_hat = mu_lambda_var;
  real sd_lambda_var_hat = sd_lambda_var;
  real mu_delta_hat = mu_delta;
  real sd_delta_hat = sd_delta;
  real mu_beta_hat = mu_beta;
  real sd_beta_hat = sd_beta;
}
