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

  array[E] vector<lower=0,upper=1>[A] avail_mask;

  array[E] int<lower=0,upper=A> demo_action;
  vector[E] demo_outcome_obs;
  array[E] int<lower=0,upper=1> has_demo_outcome;

  real<lower=1e-6> beta_lower;
  real<lower=1e-6> beta_upper;

  // hyperpriors
  int<lower=1,upper=8> mu_alpha_o_prior_family; real mu_alpha_o_prior_p1; real mu_alpha_o_prior_p2; real mu_alpha_o_prior_p3;
  int<lower=1,upper=8> sd_alpha_o_prior_family; real sd_alpha_o_prior_p1; real sd_alpha_o_prior_p2; real sd_alpha_o_prior_p3;

  int<lower=1,upper=8> mu_alpha_a_prior_family; real mu_alpha_a_prior_p1; real mu_alpha_a_prior_p2; real mu_alpha_a_prior_p3;
  int<lower=1,upper=8> sd_alpha_a_prior_family; real sd_alpha_a_prior_p1; real sd_alpha_a_prior_p2; real sd_alpha_a_prior_p3;

  int<lower=1,upper=8> mu_beta_prior_family; real mu_beta_prior_p1; real mu_beta_prior_p2; real mu_beta_prior_p3;
  int<lower=1,upper=8> sd_beta_prior_family; real sd_beta_prior_p1; real sd_beta_prior_p2; real sd_beta_prior_p3;

  int<lower=1,upper=8> mu_w_prior_family; real mu_w_prior_p1; real mu_w_prior_p2; real mu_w_prior_p3;
  int<lower=1,upper=8> sd_w_prior_family; real sd_w_prior_p1; real sd_w_prior_p2; real sd_w_prior_p3;
}
parameters {
  real mu_alpha_o; real<lower=0> sd_alpha_o; vector[N] z_alpha_o;
  real mu_alpha_a; real<lower=0> sd_alpha_a; vector[N] z_alpha_a;
  real mu_beta; real<lower=0> sd_beta; vector[N] z_beta;
  real mu_w; real<lower=0> sd_w; vector[N] z_w;
}
transformed parameters {
  vector<lower=0,upper=1>[N] alpha_o = inv_logit(mu_alpha_o + sd_alpha_o * z_alpha_o);
  vector<lower=0,upper=1>[N] alpha_a = inv_logit(mu_alpha_a + sd_alpha_a * z_alpha_a);

  vector<lower=beta_lower,upper=beta_upper>[N] beta =
    beta_lower + (beta_upper - beta_lower) * (tanh(mu_beta + sd_beta * z_beta) + 1) * 0.5;

  vector<lower=0,upper=1>[N] w = inv_logit(mu_w + sd_w * z_w);
}
model {
  z_alpha_o ~ normal(0,1);
  z_alpha_a ~ normal(0,1);
  z_beta ~ normal(0,1);
  z_w ~ normal(0,1);

  target += prior_lpdf(mu_alpha_o | mu_alpha_o_prior_family, mu_alpha_o_prior_p1, mu_alpha_o_prior_p2, mu_alpha_o_prior_p3);
  target += prior_lpdf(sd_alpha_o | sd_alpha_o_prior_family, sd_alpha_o_prior_p1, sd_alpha_o_prior_p2, sd_alpha_o_prior_p3);

  target += prior_lpdf(mu_alpha_a | mu_alpha_a_prior_family, mu_alpha_a_prior_p1, mu_alpha_a_prior_p2, mu_alpha_a_prior_p3);
  target += prior_lpdf(sd_alpha_a | sd_alpha_a_prior_family, sd_alpha_a_prior_p1, sd_alpha_a_prior_p2, sd_alpha_a_prior_p3);

  target += prior_lpdf(mu_beta | mu_beta_prior_family, mu_beta_prior_p1, mu_beta_prior_p2, mu_beta_prior_p3);
  target += prior_lpdf(sd_beta | sd_beta_prior_family, sd_beta_prior_p1, sd_beta_prior_p2, sd_beta_prior_p3);

  target += prior_lpdf(mu_w | mu_w_prior_family, mu_w_prior_p1, mu_w_prior_p2, mu_w_prior_p3);
  target += prior_lpdf(sd_w | sd_w_prior_family, sd_w_prior_p1, sd_w_prior_p2, sd_w_prior_p3);

  array[N] matrix[S, A] Q;
  array[N] vector[A] demo_pi;

  for (n in 1:N) {
    Q[n] = rep_matrix(0.0, S, A);
    demo_pi[n] = rep_vector(1.0 / A, A);
  }

  for (e in 1:E) {
    int n = subj[e];
    int s = state[e];

    if (etype[e] == 1) {
      Q[n] = rep_matrix(0.0, S, A);
      demo_pi[n] = rep_vector(1.0 / A, A);

    } else if (etype[e] == 2) {
      if (demo_action[e] > 0) {
        int a = demo_action[e];
        vector[A] onehot = rep_vector(0.0, A);
        onehot[a] = 1.0;

        demo_pi[n] = demo_pi[n] + alpha_a[n] * (onehot - demo_pi[n]);

        if (has_demo_outcome[e] == 1) {
          real r = demo_outcome_obs[e];
          Q[n][s,a] = Q[n][s,a] + alpha_o[n] * (r - Q[n][s,a]);
        }
      }

    } else if (etype[e] == 3) {
      if (choice[e] > 0) {
        vector[A] g = (demo_pi[n] - (1.0 / A)) / (1.0 - (1.0 / A));
        vector[A] social_drive = w[n] * to_vector(Q[n][s]') + (1.0 - w[n]) * g;
        vector[A] u = beta[n] * social_drive;
        for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
        target += categorical_logit_lpmf(choice[e] | u);
      }
    }
  }
}

generated quantities {
  vector[E] log_lik = rep_vector(0.0, E);
  {
    array[N] matrix[S, A] Q;
    array[N] vector[A] demo_pi;

    for (n in 1:N) {
      Q[n] = rep_matrix(0.0, S, A);
      demo_pi[n] = rep_vector(1.0 / A, A);
    }

    for (e in 1:E) {
      int n = subj[e];
      int s = state[e];

      if (etype[e] == 1) {
        Q[n] = rep_matrix(0.0, S, A);
        demo_pi[n] = rep_vector(1.0 / A, A);

      } else if (etype[e] == 2) {
        if (demo_action[e] > 0) {
          int a = demo_action[e];
          vector[A] onehot = rep_vector(0.0, A);
          onehot[a] = 1.0;

          demo_pi[n] = demo_pi[n] + alpha_a[n] * (onehot - demo_pi[n]);

          if (has_demo_outcome[e] == 1) {
            real r = demo_outcome_obs[e];
            Q[n][s,a] = Q[n][s,a] + alpha_o[n] * (r - Q[n][s,a]);
          }
        }

      } else if (etype[e] == 3) {
        if (choice[e] > 0) {
          vector[A] g = (demo_pi[n] - (1.0 / A)) / (1.0 - (1.0 / A));
          vector[A] social_drive = w[n] * to_vector(Q[n][s]') + (1.0 - w[n]) * g;
          vector[A] u = beta[n] * social_drive;
          for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
          log_lik[e] = categorical_logit_lpmf(choice[e] | u);
        }
      }
    }
  }

  real alpha_o_pop = inv_logit(mu_alpha_o);
  real alpha_a_pop = inv_logit(mu_alpha_a);

  real beta_pop =
    beta_lower + (beta_upper - beta_lower) * (tanh(mu_beta) + 1) * 0.5;

  real w_pop = inv_logit(mu_w);

  real mu_alpha_o_hat = mu_alpha_o;
  real sd_alpha_o_hat = sd_alpha_o;
  real mu_alpha_a_hat = mu_alpha_a;
  real sd_alpha_a_hat = sd_alpha_a;
  real mu_beta_hat = mu_beta;
  real sd_beta_hat = sd_beta;
  real mu_w_hat = mu_w;
  real sd_w_hat = sd_w;
}
