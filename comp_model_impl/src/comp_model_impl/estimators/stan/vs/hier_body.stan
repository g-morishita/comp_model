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
  vector[E] outcome_obs;

  int<lower=0,upper=A> demo_action[E];
  vector[E] demo_outcome_obs;
  int<lower=0,upper=1> has_demo_outcome[E];

  real pseudo_reward;
  real<lower=1e-6> beta_lower;
  real<lower=1e-6> beta_upper;
  real<lower=0> kappa_abs_max;

  // hyperpriors (configurable)
  int<lower=1,upper=8> mu_ap_prior_family; real mu_ap_prior_p1; real mu_ap_prior_p2; real mu_ap_prior_p3;
  int<lower=1,upper=8> sd_ap_prior_family; real sd_ap_prior_p1; real sd_ap_prior_p2; real sd_ap_prior_p3;

  int<lower=1,upper=8> mu_ai_prior_family; real mu_ai_prior_p1; real mu_ai_prior_p2; real mu_ai_prior_p3;
  int<lower=1,upper=8> sd_ai_prior_family; real sd_ai_prior_p1; real sd_ai_prior_p2; real sd_ai_prior_p3;

  int<lower=1,upper=8> mu_b_prior_family;  real mu_b_prior_p1;  real mu_b_prior_p2;  real mu_b_prior_p3;
  int<lower=1,upper=8> sd_b_prior_family;  real sd_b_prior_p1;  real sd_b_prior_p2;  real sd_b_prior_p3;

  int<lower=1,upper=8> mu_k_prior_family;  real mu_k_prior_p1;  real mu_k_prior_p2;  real mu_k_prior_p3;
  int<lower=1,upper=8> sd_k_prior_family;  real sd_k_prior_p1;  real sd_k_prior_p2;  real sd_k_prior_p3;
}
parameters {
  // non-centered
  real mu_ap; real<lower=0> sd_ap; vector[N] z_ap;
  real mu_ai; real<lower=0> sd_ai; vector[N] z_ai;
  real mu_b;  real<lower=0> sd_b;  vector[N] z_b;
  real mu_k;  real<lower=0> sd_k;  vector[N] z_k;
}
transformed parameters {
  vector<lower=0,upper=1>[N] alpha_p = inv_logit(mu_ap + sd_ap * z_ap);
  vector<lower=0,upper=1>[N] alpha_i = inv_logit(mu_ai + sd_ai * z_ai);

  vector<lower=beta_lower,upper=beta_upper>[N] beta =
    beta_lower + (beta_upper - beta_lower) * inv_logit(mu_b + sd_b * z_b);

  vector<lower=-kappa_abs_max,upper=kappa_abs_max>[N] kappa =
    kappa_abs_max * (2 * inv_logit(mu_k + sd_k * z_k) - 1);
}
model {
  z_ap ~ normal(0,1); z_ai ~ normal(0,1); z_b ~ normal(0,1); z_k ~ normal(0,1);

  target += prior_lpdf(mu_ap, mu_ap_prior_family, mu_ap_prior_p1, mu_ap_prior_p2, mu_ap_prior_p3);
  target += prior_lpdf(sd_ap, sd_ap_prior_family, sd_ap_prior_p1, sd_ap_prior_p2, sd_ap_prior_p3);

  target += prior_lpdf(mu_ai, mu_ai_prior_family, mu_ai_prior_p1, mu_ai_prior_p2, mu_ai_prior_p3);
  target += prior_lpdf(sd_ai, sd_ai_prior_family, sd_ai_prior_p1, sd_ai_prior_p2, sd_ai_prior_p3);

  target += prior_lpdf(mu_b,  mu_b_prior_family,  mu_b_prior_p1,  mu_b_prior_p2,  mu_b_prior_p3);
  target += prior_lpdf(sd_b,  sd_b_prior_family,  sd_b_prior_p1,  sd_b_prior_p2,  sd_b_prior_p3);

  target += prior_lpdf(mu_k,  mu_k_prior_family,  mu_k_prior_p1,  mu_k_prior_p2,  mu_k_prior_p3);
  target += prior_lpdf(sd_k,  sd_k_prior_family,  sd_k_prior_p1,  sd_k_prior_p2,  sd_k_prior_p3);

  array[N] matrix[S, A] Q;
  array[N] array[S] int last_choice;
  for (n in 1:N) {
    Q[n] = rep_matrix(0.0, S, A);
    last_choice[n] = rep_array(0, S);
  }

  for (e in 1:E) {
    int n = subj[e];
    int s = state[e];

    if (etype[e] == 1) {
      Q[n] = rep_matrix(0.0, S, A);
      last_choice[n] = rep_array(0, S);

    } else if (etype[e] == 2) {
      if (demo_action[e] > 0) {
        int a = demo_action[e];
        Q[n][s,a] = Q[n][s,a] + alpha_i[n] * (pseudo_reward - Q[n][s,a]);
      }

    } else if (etype[e] == 3) {
      if (choice[e] > 0) {
        vector[A] u = to_vector(Q[n][s]');
        if (last_choice[n][s] > 0) u[last_choice[n][s]] += kappa[n];
        target += categorical_logit_lpmf(choice[e] | beta[n] * u);
      }

    } else if (etype[e] == 4) {
      if (action[e] > 0) {
        int a = action[e];
        real r = outcome_obs[e];
        Q[n][s,a] = Q[n][s,a] + alpha_p[n] * (r - Q[n][s,a]);
        last_choice[n][s] = a;
      }
    }
  }
}
