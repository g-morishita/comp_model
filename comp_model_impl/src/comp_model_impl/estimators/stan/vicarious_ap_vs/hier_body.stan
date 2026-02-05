data {
  int<lower=1> N;
  int<lower=1> A;
  int<lower=1> S;
  int<lower=1> E;

  array[E] int<lower=1,upper=N> subj;
  array[E] int<lower=1,upper=4> etype;

  array[E] int<lower=1,upper=S> state;
  array[E] int<lower=0,upper=A> choice;
  array[E] int<lower=0,upper=A> action;
  vector[E] outcome_obs; // unused

  array[E] int<lower=0,upper=A> demo_action;
  vector[E] demo_outcome_obs;
  array[E] int<lower=0,upper=1> has_demo_outcome;

  real pseudo_reward;
  real<lower=1e-6> beta_lower;
  real<lower=1e-6> beta_upper;
  real<lower=0> kappa_abs_max;

  // hyperpriors
  int<lower=1,upper=8> mu_alpha_o_prior_family; real mu_alpha_o_prior_p1; real mu_alpha_o_prior_p2; real mu_alpha_o_prior_p3;
  int<lower=1,upper=8> sd_alpha_o_prior_family; real sd_alpha_o_prior_p1; real sd_alpha_o_prior_p2; real sd_alpha_o_prior_p3;

  int<lower=1,upper=8> mu_alpha_vs_base_prior_family; real mu_alpha_vs_base_prior_p1; real mu_alpha_vs_base_prior_p2; real mu_alpha_vs_base_prior_p3;
  int<lower=1,upper=8> sd_alpha_vs_base_prior_family; real sd_alpha_vs_base_prior_p1; real sd_alpha_vs_base_prior_p2; real sd_alpha_vs_base_prior_p3;

  int<lower=1,upper=8> mu_alpha_a_prior_family; real mu_alpha_a_prior_p1; real mu_alpha_a_prior_p2; real mu_alpha_a_prior_p3;
  int<lower=1,upper=8> sd_alpha_a_prior_family; real sd_alpha_a_prior_p1; real sd_alpha_a_prior_p2; real sd_alpha_a_prior_p3;

  int<lower=1,upper=8> mu_beta_prior_family;  real mu_beta_prior_p1;  real mu_beta_prior_p2;  real mu_beta_prior_p3;
  int<lower=1,upper=8> sd_beta_prior_family;  real sd_beta_prior_p1;  real sd_beta_prior_p2;  real sd_beta_prior_p3;

  int<lower=1,upper=8> mu_kappa_prior_family;  real mu_kappa_prior_p1;  real mu_kappa_prior_p2;  real mu_kappa_prior_p3;
  int<lower=1,upper=8> sd_kappa_prior_family;  real sd_kappa_prior_p1;  real sd_kappa_prior_p2;  real sd_kappa_prior_p3;
}
parameters {
  real mu_alpha_o; real<lower=0> sd_alpha_o; vector[N] z_alpha_o;
  real mu_alpha_vs_base; real<lower=0> sd_alpha_vs_base; vector[N] z_alpha_vs_base;
  real mu_alpha_a; real<lower=0> sd_alpha_a; vector[N] z_alpha_a;
  real mu_beta;  real<lower=0> sd_beta;  vector[N] z_beta;
  real mu_kappa; real<lower=0> sd_kappa; vector[N] z_kappa;
}
transformed parameters {
  vector<lower=0,upper=1>[N] alpha_o = inv_logit(mu_alpha_o + sd_alpha_o * z_alpha_o);
  vector<lower=0,upper=1>[N] alpha_vs_base = inv_logit(mu_alpha_vs_base + sd_alpha_vs_base * z_alpha_vs_base);
  vector<lower=0,upper=1>[N] alpha_a = inv_logit(mu_alpha_a + sd_alpha_a * z_alpha_a);

  vector<lower=beta_lower,upper=beta_upper>[N] beta =
    beta_lower + (beta_upper - beta_lower) * inv_logit(mu_beta + sd_beta * z_beta);

  vector<lower=-kappa_abs_max,upper=kappa_abs_max>[N] kappa =
    kappa_abs_max * (2 * inv_logit(mu_kappa + sd_kappa * z_kappa) - 1);
}
model {
  z_alpha_o ~ normal(0,1);
  z_alpha_vs_base ~ normal(0,1);
  z_alpha_a ~ normal(0,1);
  z_beta ~ normal(0,1);
  z_kappa ~ normal(0,1);

  target += prior_lpdf(mu_alpha_o | mu_alpha_o_prior_family, mu_alpha_o_prior_p1, mu_alpha_o_prior_p2, mu_alpha_o_prior_p3);
  target += prior_lpdf(sd_alpha_o | sd_alpha_o_prior_family, sd_alpha_o_prior_p1, sd_alpha_o_prior_p2, sd_alpha_o_prior_p3);

  target += prior_lpdf(
    mu_alpha_vs_base | mu_alpha_vs_base_prior_family, mu_alpha_vs_base_prior_p1, mu_alpha_vs_base_prior_p2, mu_alpha_vs_base_prior_p3
  );
  target += prior_lpdf(
    sd_alpha_vs_base | sd_alpha_vs_base_prior_family, sd_alpha_vs_base_prior_p1, sd_alpha_vs_base_prior_p2, sd_alpha_vs_base_prior_p3
  );

  target += prior_lpdf(mu_alpha_a | mu_alpha_a_prior_family, mu_alpha_a_prior_p1, mu_alpha_a_prior_p2, mu_alpha_a_prior_p3);
  target += prior_lpdf(sd_alpha_a | sd_alpha_a_prior_family, sd_alpha_a_prior_p1, sd_alpha_a_prior_p2, sd_alpha_a_prior_p3);

  target += prior_lpdf(mu_beta | mu_beta_prior_family, mu_beta_prior_p1, mu_beta_prior_p2, mu_beta_prior_p3);
  target += prior_lpdf(sd_beta | sd_beta_prior_family, sd_beta_prior_p1, sd_beta_prior_p2, sd_beta_prior_p3);

  target += prior_lpdf(mu_kappa | mu_kappa_prior_family, mu_kappa_prior_p1, mu_kappa_prior_p2, mu_kappa_prior_p3);
  target += prior_lpdf(sd_kappa | sd_kappa_prior_family, sd_kappa_prior_p1, sd_kappa_prior_p2, sd_kappa_prior_p3);

  array[N] matrix[S, A] Q;
  array[N, S] int last_choice;
  array[N] vector[A] demo_pi;

  for (n in 1:N) {
    Q[n] = rep_matrix(0.0, S, A);
    for (s in 1:S) last_choice[n, s] = 0;
    demo_pi[n] = rep_vector(1.0 / A, A);
  }

  for (e in 1:E) {
    int n = subj[e];
    int s = state[e];

    if (etype[e] == 1) {
      Q[n] = rep_matrix(0.0, S, A);
      for (s2 in 1:S) last_choice[n, s2] = 0;
      demo_pi[n] = rep_vector(1.0 / A, A);

    } else if (etype[e] == 2) {
      if (demo_action[e] > 0) {
        int a = demo_action[e];
        vector[A] onehot = rep_vector(0.0, A);
        onehot[a] = 1.0;

        demo_pi[n] = demo_pi[n] + alpha_a[n] * (onehot - demo_pi[n]);

        real maxp = max(demo_pi[n]);
        real rel = (maxp - 1.0 / A) / (1.0 - 1.0 / A);
        rel = fmin(1.0, fmax(0.0, rel));
        real alpha_vs = alpha_vs_base[n] * rel;

        Q[n][s,a] = Q[n][s,a] + alpha_vs * (pseudo_reward - Q[n][s,a]);

        if (has_demo_outcome[e] == 1) {
          real r = demo_outcome_obs[e];
          Q[n][s,a] = Q[n][s,a] + alpha_o[n] * (r - Q[n][s,a]);
        }
      }

    } else if (etype[e] == 3) {
      if (choice[e] > 0) {
        vector[A] u = beta[n] * to_vector(Q[n][s]');
        if (last_choice[n, s] > 0) u[last_choice[n, s]] += kappa[n];
        target += categorical_logit_lpmf(choice[e] | u);
      }

    } else if (etype[e] == 4) {
      if (action[e] > 0) {
        last_choice[n, s] = action[e];
      }
    }
  }
}

generated quantities {
  real alpha_o_pop = inv_logit(mu_alpha_o);
  real alpha_vs_base_pop = inv_logit(mu_alpha_vs_base);
  real alpha_a_pop = inv_logit(mu_alpha_a);

  real beta_pop =
    beta_lower + (beta_upper - beta_lower) * inv_logit(mu_beta);

  real kappa_pop =
    kappa_abs_max * (2 * inv_logit(mu_kappa) - 1);

  real mu_alpha_o_hat = mu_alpha_o;
  real sd_alpha_o_hat = sd_alpha_o;
  real mu_alpha_vs_base_hat = mu_alpha_vs_base;
  real sd_alpha_vs_base_hat = sd_alpha_vs_base;
  real mu_alpha_a_hat = mu_alpha_a;
  real sd_alpha_a_hat = sd_alpha_a;
  real mu_beta_hat = mu_beta;
  real sd_beta_hat = sd_beta;
  real mu_kappa_hat = mu_kappa;
  real sd_kappa_hat = sd_kappa;
}
