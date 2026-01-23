data {
  int<lower=1> N;
  int<lower=1> A;
  int<lower=1> S;
  int<lower=1> E;

  int<lower=1,upper=N> subj[E];
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

  // hyperpriors
  int<lower=1,upper=8> mu_ao_prior_family; real mu_ao_prior_p1; real mu_ao_prior_p2; real mu_ao_prior_p3;
  int<lower=1,upper=8> sd_ao_prior_family; real sd_ao_prior_p1; real sd_ao_prior_p2; real sd_ao_prior_p3;

  int<lower=1,upper=8> mu_b_prior_family;  real mu_b_prior_p1;  real mu_b_prior_p2;  real mu_b_prior_p3;
  int<lower=1,upper=8> sd_b_prior_family;  real sd_b_prior_p1;  real sd_b_prior_p2;  real sd_b_prior_p3;
}
parameters {
  real mu_ao; real<lower=0> sd_ao; vector[N] z_ao;
  real mu_b;  real<lower=0> sd_b;  vector[N] z_b;
}
transformed parameters {
  vector<lower=0,upper=1>[N] alpha_o = inv_logit(mu_ao + sd_ao * z_ao);
  vector<lower=beta_lower,upper=beta_upper>[N] beta =
    beta_lower + (beta_upper - beta_lower) * inv_logit(mu_b + sd_b * z_b);
}
model {
  z_ao ~ normal(0,1);
  z_b  ~ normal(0,1);

  target += prior_lpdf(mu_ao, mu_ao_prior_family, mu_ao_prior_p1, mu_ao_prior_p2, mu_ao_prior_p3);
  target += prior_lpdf(sd_ao, sd_ao_prior_family, sd_ao_prior_p1, sd_ao_prior_p2, sd_ao_prior_p3);

  target += prior_lpdf(mu_b,  mu_b_prior_family,  mu_b_prior_p1,  mu_b_prior_p2,  mu_b_prior_p3);
  target += prior_lpdf(sd_b,  sd_b_prior_family,  sd_b_prior_p1,  sd_b_prior_p2,  sd_b_prior_p3);

  array[N] matrix[S, A] Q;
  for (n in 1:N) Q[n] = rep_matrix(0.0, S, A);

  for (e in 1:E) {
    int n = subj[e];
    int s = state[e];

    if (etype[e] == 1) {
      Q[n] = rep_matrix(0.0, S, A);

    } else if (etype[e] == 2) {
      if (demo_action[e] > 0 && has_demo_outcome[e] == 1) {
        int a = demo_action[e];
        real r = demo_outcome_obs[e];
        Q[n][s,a] = Q[n][s,a] + alpha_o[n] * (r - Q[n][s,a]);
      }

    } else if (etype[e] == 3) {
      if (choice[e] > 0) {
        vector[A] u = to_vector(Q[n][s]');
        target += categorical_logit_lpmf(choice[e] | beta[n] * u);
      }
    }
  }
}

generated quantities {
  real alpha_o_pop = inv_logit(mu_ao);

  real beta_pop =
    beta_lower + (beta_upper - beta_lower) * inv_logit(mu_b);

  real mu_ao_hat = mu_ao;
  real sd_ao_hat = sd_ao;
  real mu_b_hat  = mu_b;
  real sd_b_hat  = sd_b;
}
