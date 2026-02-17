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
  real<lower=0> kappa_abs_max;
  real<lower=0> demo_bias_rel_abs_max;
  real<lower=1e-6> demo_dirichlet_prior;

  // hyperpriors
  int<lower=1,upper=8> mu_alpha_o_prior_family; real mu_alpha_o_prior_p1; real mu_alpha_o_prior_p2; real mu_alpha_o_prior_p3;
  int<lower=1,upper=8> sd_alpha_o_prior_family; real sd_alpha_o_prior_p1; real sd_alpha_o_prior_p2; real sd_alpha_o_prior_p3;

  int<lower=1,upper=8> mu_demo_bias_rel_prior_family; real mu_demo_bias_rel_prior_p1; real mu_demo_bias_rel_prior_p2; real mu_demo_bias_rel_prior_p3;
  int<lower=1,upper=8> sd_demo_bias_rel_prior_family; real sd_demo_bias_rel_prior_p1; real sd_demo_bias_rel_prior_p2; real sd_demo_bias_rel_prior_p3;

  int<lower=1,upper=8> mu_beta_prior_family;  real mu_beta_prior_p1;  real mu_beta_prior_p2;  real mu_beta_prior_p3;
  int<lower=1,upper=8> sd_beta_prior_family;  real sd_beta_prior_p1;  real sd_beta_prior_p2;  real sd_beta_prior_p3;

  int<lower=1,upper=8> mu_kappa_prior_family;  real mu_kappa_prior_p1;  real mu_kappa_prior_p2;  real mu_kappa_prior_p3;
  int<lower=1,upper=8> sd_kappa_prior_family;  real sd_kappa_prior_p1;  real sd_kappa_prior_p2;  real sd_kappa_prior_p3;
}
parameters {
  real mu_alpha_o; real<lower=0> sd_alpha_o; vector[N] z_alpha_o;
  real mu_demo_bias_rel; real<lower=0> sd_demo_bias_rel; vector[N] z_demo_bias_rel;
  real mu_beta;  real<lower=0> sd_beta;  vector[N] z_beta;
  real mu_kappa; real<lower=0> sd_kappa; vector[N] z_kappa;
}
transformed parameters {
  vector<lower=0,upper=1>[N] alpha_o = inv_logit(mu_alpha_o + sd_alpha_o * z_alpha_o);

  vector<lower=-demo_bias_rel_abs_max,upper=demo_bias_rel_abs_max>[N] demo_bias_rel =
    demo_bias_rel_abs_max * tanh(mu_demo_bias_rel + sd_demo_bias_rel * z_demo_bias_rel);

  vector<lower=beta_lower>[N] beta =
    beta_lower + exp(mu_beta + sd_beta * z_beta);

  vector<lower=-kappa_abs_max,upper=kappa_abs_max>[N] kappa =
    kappa_abs_max * tanh(mu_kappa + sd_kappa * z_kappa);
}
model {
  z_alpha_o ~ normal(0,1);
  z_demo_bias_rel ~ normal(0,1);
  z_beta ~ normal(0,1);
  z_kappa ~ normal(0,1);

  target += prior_lpdf(mu_alpha_o | mu_alpha_o_prior_family, mu_alpha_o_prior_p1, mu_alpha_o_prior_p2, mu_alpha_o_prior_p3);
  target += prior_lpdf(sd_alpha_o | sd_alpha_o_prior_family, sd_alpha_o_prior_p1, sd_alpha_o_prior_p2, sd_alpha_o_prior_p3);

  target += prior_lpdf(
    mu_demo_bias_rel | mu_demo_bias_rel_prior_family, mu_demo_bias_rel_prior_p1, mu_demo_bias_rel_prior_p2, mu_demo_bias_rel_prior_p3
  );
  target += prior_lpdf(
    sd_demo_bias_rel | sd_demo_bias_rel_prior_family, sd_demo_bias_rel_prior_p1, sd_demo_bias_rel_prior_p2, sd_demo_bias_rel_prior_p3
  );

  target += prior_lpdf(mu_beta | mu_beta_prior_family, mu_beta_prior_p1, mu_beta_prior_p2, mu_beta_prior_p3);
  target += prior_lpdf(sd_beta | sd_beta_prior_family, sd_beta_prior_p1, sd_beta_prior_p2, sd_beta_prior_p3);

  target += prior_lpdf(mu_kappa | mu_kappa_prior_family, mu_kappa_prior_p1, mu_kappa_prior_p2, mu_kappa_prior_p3);
  target += prior_lpdf(sd_kappa | sd_kappa_prior_family, sd_kappa_prior_p1, sd_kappa_prior_p2, sd_kappa_prior_p3);

  array[N] matrix[S, A] Q;
  array[N, S] int last_choice;
  array[N] vector[A] demo_counts;
  array[N] int recent_demo_choice;

  for (n in 1:N) {
    Q[n] = rep_matrix(0.0, S, A);
    for (s in 1:S) last_choice[n, s] = 0;
    demo_counts[n] = rep_vector(demo_dirichlet_prior, A);
    recent_demo_choice[n] = 0;
  }

  for (e in 1:E) {
    int n = subj[e];
    int s = state[e];

    if (etype[e] == 1) {
      Q[n] = rep_matrix(0.0, S, A);
      for (s2 in 1:S) last_choice[n, s2] = 0;
      demo_counts[n] = rep_vector(demo_dirichlet_prior, A);
      recent_demo_choice[n] = 0;

    } else if (etype[e] == 2) {
      if (demo_action[e] > 0) {
        int a = demo_action[e];
        recent_demo_choice[n] = a;
        demo_counts[n][a] = demo_counts[n][a] + 1.0;

        if (has_demo_outcome[e] == 1) {
          real r = demo_outcome_obs[e];
          Q[n][s,a] = Q[n][s,a] + alpha_o[n] * (r - Q[n][s,a]);
        }
      }

    } else if (etype[e] == 3) {
      if (choice[e] > 0) {
        vector[A] demo_pi = demo_counts[n] / sum(demo_counts[n]);
        real maxp = max(demo_pi);
        real rel = (maxp - 1.0 / A) / (1.0 - 1.0 / A);
        rel = fmin(1.0, fmax(0.0, rel));

        vector[A] u = beta[n] * to_vector(Q[n][s]');
        if (last_choice[n, s] > 0) u[last_choice[n, s]] += kappa[n];
        if (recent_demo_choice[n] > 0) u[recent_demo_choice[n]] += demo_bias_rel[n] * rel;
        for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
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
  vector[E] log_lik = rep_vector(0.0, E);
  {
    array[N] matrix[S, A] Q;
    array[N, S] int last_choice;
    array[N] vector[A] demo_counts;
    array[N] int recent_demo_choice;
  
    for (n in 1:N) {
      Q[n] = rep_matrix(0.0, S, A);
      for (s in 1:S) last_choice[n, s] = 0;
      demo_counts[n] = rep_vector(demo_dirichlet_prior, A);
      recent_demo_choice[n] = 0;
    }
  
    for (e in 1:E) {
      int n = subj[e];
      int s = state[e];
  
      if (etype[e] == 1) {
        Q[n] = rep_matrix(0.0, S, A);
        for (s2 in 1:S) last_choice[n, s2] = 0;
        demo_counts[n] = rep_vector(demo_dirichlet_prior, A);
        recent_demo_choice[n] = 0;
  
      } else if (etype[e] == 2) {
        if (demo_action[e] > 0) {
          int a = demo_action[e];
          recent_demo_choice[n] = a;
          demo_counts[n][a] = demo_counts[n][a] + 1.0;
  
          if (has_demo_outcome[e] == 1) {
            real r = demo_outcome_obs[e];
            Q[n][s,a] = Q[n][s,a] + alpha_o[n] * (r - Q[n][s,a]);
          }
        }
  
      } else if (etype[e] == 3) {
        if (choice[e] > 0) {
          vector[A] demo_pi = demo_counts[n] / sum(demo_counts[n]);
          real maxp = max(demo_pi);
          real rel = (maxp - 1.0 / A) / (1.0 - 1.0 / A);
          rel = fmin(1.0, fmax(0.0, rel));
  
          vector[A] u = beta[n] * to_vector(Q[n][s]');
          if (last_choice[n, s] > 0) u[last_choice[n, s]] += kappa[n];
          if (recent_demo_choice[n] > 0) u[recent_demo_choice[n]] += demo_bias_rel[n] * rel;
          for (a in 1:A) if (avail_mask[e][a] == 0) u[a] = negative_infinity();
          log_lik[e] = categorical_logit_lpmf(choice[e] | u);
        }
  
      } else if (etype[e] == 4) {
        if (action[e] > 0) {
          last_choice[n, s] = action[e];
        }
      }
    }
  }
  real alpha_o_pop = inv_logit(mu_alpha_o);

  real demo_bias_rel_pop = demo_bias_rel_abs_max * tanh(mu_demo_bias_rel);

  real beta_pop =
    beta_lower + exp(mu_beta);

  real kappa_pop =
    kappa_abs_max * tanh(mu_kappa);

  real mu_alpha_o_hat = mu_alpha_o;
  real sd_alpha_o_hat = sd_alpha_o;
  real mu_demo_bias_rel_hat = mu_demo_bias_rel;
  real sd_demo_bias_rel_hat = sd_demo_bias_rel;
  real mu_beta_hat = mu_beta;
  real sd_beta_hat = sd_beta;
  real mu_kappa_hat = mu_kappa;
  real sd_kappa_hat = sd_kappa;
}
