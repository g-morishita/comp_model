data {
  int<lower=1> B;
  int<lower=1> K;
  int<lower=1> S;
  int<lower=1> A;
  int<lower=1> T_max;
  array[B] int<lower=1, upper=T_max> T;
  array[B, T_max] int<lower=1, upper=S> state_idx;
  array[B, T_max] int<lower=1, upper=A> action_idx;
  array[B, T_max] real reward;
  array[B, T_max, A] int<lower=0, upper=1> is_available;
  array[K] int<lower=1, upper=5> param_codes;
  array[K] int<lower=0, upper=2> transform_codes;
  real fixed_alpha_1;
  real fixed_alpha_2;
  real fixed_beta;
  real fixed_kappa;
  real fixed_initial_value;
  vector[K] mu_prior_mean;
  vector<lower=0>[K] mu_prior_std;
  vector[K] log_sigma_prior_mean;
  vector<lower=0>[K] log_sigma_prior_std;
}

parameters {
  vector[K] group_loc_z;
  vector[K] group_log_scale;
  array[B] vector[K] block_z;
}

transformed parameters {
  array[B] vector[K] block_param;
  for (b in 1:B) {
    for (k in 1:K) {
      if (transform_codes[k] == 0) {
        block_param[b][k] = block_z[b][k];
      } else if (transform_codes[k] == 1) {
        block_param[b][k] = inv_logit(block_z[b][k]);
      } else {
        block_param[b][k] = exp(block_z[b][k]);
      }
    }
  }
}

model {
  for (k in 1:K) {
    target += normal_lpdf(group_loc_z[k] | mu_prior_mean[k], mu_prior_std[k]);
    target += normal_lpdf(group_log_scale[k] | log_sigma_prior_mean[k], log_sigma_prior_std[k]);
  }

  for (b in 1:B) {
    for (k in 1:K) {
      target += normal_lpdf(block_z[b][k] | group_loc_z[k], exp(group_log_scale[k]));
    }
  }

  for (b in 1:B) {
    real alpha_1 = fixed_alpha_1;
    real alpha_2 = fixed_alpha_2;
    real beta = fixed_beta;
    real kappa = fixed_kappa;
    real initial_value = fixed_initial_value;

    for (k in 1:K) {
      if (param_codes[k] == 1) {
        alpha_1 = block_param[b][k];
      } else if (param_codes[k] == 2) {
        alpha_2 = block_param[b][k];
      } else if (param_codes[k] == 3) {
        beta = block_param[b][k];
      } else if (param_codes[k] == 4) {
        kappa = block_param[b][k];
      } else if (param_codes[k] == 5) {
        initial_value = block_param[b][k];
      }
    }
    real alpha = alpha_1 + alpha_2;

    array[S] vector[A] q;
    array[S] int last_choice;
    for (s in 1:S) {
      for (a in 1:A) {
        q[s][a] = initial_value;
      }
      last_choice[s] = 0;
    }

    for (t in 1:T[b]) {
      int s = state_idx[b, t];
      int a_obs = action_idx[b, t];
      vector[A] logits;
      for (a in 1:A) {
        if (is_available[b, t, a] == 1) {
          real stay = 0;
          if (last_choice[s] == a) {
            stay = kappa;
          }
          logits[a] = beta * q[s][a] + stay;
        } else {
          logits[a] = -1e12;
        }
      }

      target += categorical_logit_lpmf(a_obs | logits);
      q[s][a_obs] = q[s][a_obs] + alpha * (reward[b, t] - q[s][a_obs]);
      last_choice[s] = a_obs;
    }
  }
}

generated quantities {
  real log_prior_total = 0;
  real log_likelihood_total = 0;
  real log_posterior_total;

  for (k in 1:K) {
    log_prior_total += normal_lpdf(group_loc_z[k] | mu_prior_mean[k], mu_prior_std[k]);
    log_prior_total += normal_lpdf(group_log_scale[k] | log_sigma_prior_mean[k], log_sigma_prior_std[k]);
  }

  for (b in 1:B) {
    for (k in 1:K) {
      log_prior_total += normal_lpdf(block_z[b][k] | group_loc_z[k], exp(group_log_scale[k]));
    }
  }

  for (b in 1:B) {
    real alpha_1 = fixed_alpha_1;
    real alpha_2 = fixed_alpha_2;
    real beta = fixed_beta;
    real kappa = fixed_kappa;
    real initial_value = fixed_initial_value;

    for (k in 1:K) {
      if (param_codes[k] == 1) {
        alpha_1 = block_param[b][k];
      } else if (param_codes[k] == 2) {
        alpha_2 = block_param[b][k];
      } else if (param_codes[k] == 3) {
        beta = block_param[b][k];
      } else if (param_codes[k] == 4) {
        kappa = block_param[b][k];
      } else if (param_codes[k] == 5) {
        initial_value = block_param[b][k];
      }
    }
    real alpha = alpha_1 + alpha_2;

    array[S] vector[A] q;
    array[S] int last_choice;
    for (s in 1:S) {
      for (a in 1:A) {
        q[s][a] = initial_value;
      }
      last_choice[s] = 0;
    }

    for (t in 1:T[b]) {
      int s = state_idx[b, t];
      int a_obs = action_idx[b, t];
      vector[A] logits;
      for (a in 1:A) {
        if (is_available[b, t, a] == 1) {
          real stay = 0;
          if (last_choice[s] == a) {
            stay = kappa;
          }
          logits[a] = beta * q[s][a] + stay;
        } else {
          logits[a] = -1e12;
        }
      }

      log_likelihood_total += categorical_logit_lpmf(a_obs | logits);
      q[s][a_obs] = q[s][a_obs] + alpha * (reward[b, t] - q[s][a_obs]);
      last_choice[s] = a_obs;
    }
  }

  log_posterior_total = log_prior_total + log_likelihood_total;
}
