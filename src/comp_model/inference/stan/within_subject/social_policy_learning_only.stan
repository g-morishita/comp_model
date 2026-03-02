// Stan program for social model component: social_policy_learning_only
// This file is intentionally per-model for researcher-facing traceability.

functions {
  real determinism_reliability(vector probabilities, int n_actions) {
    if (n_actions <= 1) {
      return 1.0;
    }
    real uniform_prob = 1.0 / n_actions;
    real rel = (max(probabilities) - uniform_prob) / (1.0 - uniform_prob);
    if (rel < 0.0) {
      return 0.0;
    }
    if (rel > 1.0) {
      return 1.0;
    }
    return rel;
  }

  vector normalized_policy_drive(vector probabilities, int n_actions) {
    vector[n_actions] out;
    if (n_actions <= 1) {
      out = rep_vector(0.0, n_actions);
      return out;
    }
    real uniform_prob = 1.0 / n_actions;
    out = (probabilities - uniform_prob) / (1.0 - uniform_prob);
    return out;
  }
}

data {
  int<lower=1> B;
  int<lower=1> K;
  int<lower=1> S;
  int<lower=1> A;
  int<lower=1> T_max;
  array[B] int<lower=1, upper=T_max> T;
  array[B, T_max] int<lower=1, upper=S> state_idx;
  array[B, T_max] int<lower=1, upper=A> action_idx;
  array[B, T_max] int<lower=1, upper=2> actor_code;
  array[B, T_max] real reward;
  array[B, T_max] int<lower=0, upper=1> has_reward;
  array[B, T_max, A] int<lower=0, upper=1> is_available;
  array[B, T_max] int<lower=0, upper=A> obs_demo_action_idx;
  array[K] int<lower=1, upper=15> param_codes;
  array[K] int<lower=0, upper=2> transform_codes;

  real fixed_alpha_self;
  real fixed_alpha_observed;
  real fixed_alpha_social;
  real fixed_alpha_policy;
  real fixed_alpha_social_base;
  real fixed_beta;
  real fixed_beta_q;
  real fixed_beta_policy;
  real fixed_kappa;
  real fixed_mix_weight;
  real fixed_demo_bias;
  real fixed_demo_bias_rel;
  real fixed_demo_dirichlet_prior;
  real fixed_initial_value;
  real fixed_pseudo_reward;

  int<lower=0, upper=1> flag_subject_outcome_learning;
  int<lower=0, upper=1> flag_demo_outcome_learning;
  int<lower=0, upper=1> flag_social_shaping_on_demo;
  int<lower=0, upper=1> flag_social_shaping_from_subject_observation;
  int<lower=0, upper=2> reliability_for_social_shaping;
  int<lower=0, upper=2> reliability_for_demo_bias;
  int<lower=0, upper=1> flag_policy_learning;
  int<lower=0, upper=1> flag_last_choice;
  int<lower=0, upper=1> flag_recent_demo_choice;
  int<lower=0, upper=1> flag_use_shared_mix;
  int<lower=0, upper=1> flag_use_independent_mix;
  int<lower=0, upper=1> flag_include_q_in_decision;
  int<lower=0, upper=1> flag_include_policy_in_decision;

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
    real alpha_self = fixed_alpha_self;
    real alpha_observed = fixed_alpha_observed;
    real alpha_social = fixed_alpha_social;
    real alpha_policy = fixed_alpha_policy;
    real alpha_social_base = fixed_alpha_social_base;
    real beta = fixed_beta;
    real beta_q = fixed_beta_q;
    real beta_policy = fixed_beta_policy;
    real kappa = fixed_kappa;
    real mix_weight = fixed_mix_weight;
    real demo_bias = fixed_demo_bias;
    real demo_bias_rel = fixed_demo_bias_rel;
    real demo_dirichlet_prior = fixed_demo_dirichlet_prior;
    real initial_value = fixed_initial_value;
    real pseudo_reward = fixed_pseudo_reward;

    for (k in 1:K) {
      if (param_codes[k] == 1) {
        alpha_self = block_param[b][k];
      } else if (param_codes[k] == 2) {
        alpha_observed = block_param[b][k];
      } else if (param_codes[k] == 3) {
        alpha_social = block_param[b][k];
      } else if (param_codes[k] == 4) {
        alpha_policy = block_param[b][k];
      } else if (param_codes[k] == 5) {
        alpha_social_base = block_param[b][k];
      } else if (param_codes[k] == 6) {
        beta = block_param[b][k];
      } else if (param_codes[k] == 7) {
        beta_q = block_param[b][k];
      } else if (param_codes[k] == 8) {
        beta_policy = block_param[b][k];
      } else if (param_codes[k] == 9) {
        kappa = block_param[b][k];
      } else if (param_codes[k] == 10) {
        mix_weight = block_param[b][k];
      } else if (param_codes[k] == 11) {
        demo_bias = block_param[b][k];
      } else if (param_codes[k] == 12) {
        demo_bias_rel = block_param[b][k];
      } else if (param_codes[k] == 13) {
        demo_dirichlet_prior = block_param[b][k];
      } else if (param_codes[k] == 14) {
        initial_value = block_param[b][k];
      } else if (param_codes[k] == 15) {
        pseudo_reward = block_param[b][k];
      }
    }

    array[S] vector[A] q;
    array[S] vector[A] policy_pi;
    array[S] vector[A] count_vals;
    array[S] int last_self_choice;
    array[S] int recent_demo_choice;

    for (s in 1:S) {
      for (a in 1:A) {
        q[s][a] = initial_value;
        policy_pi[s][a] = 1.0 / A;
        count_vals[s][a] = demo_dirichlet_prior;
      }
      last_self_choice[s] = 0;
      recent_demo_choice[s] = 0;
    }

    for (t in 1:T[b]) {
      int s = state_idx[b, t];
      int a_obs = action_idx[b, t];
      int actor = actor_code[b, t];

      if (flag_social_shaping_from_subject_observation == 1 && actor == 1 && obs_demo_action_idx[b, t] > 0) {
        int demo_a = obs_demo_action_idx[b, t];
        real rel = 1.0;
        if (reliability_for_social_shaping == 1) {
          rel = determinism_reliability(policy_pi[s], A);
        } else if (reliability_for_social_shaping == 2) {
          vector[A] count_probs = count_vals[s] / sum(count_vals[s]);
          rel = determinism_reliability(count_probs, A);
        }

        real alpha_social_effective = alpha_social;
        if (reliability_for_social_shaping != 0) {
          alpha_social_effective = alpha_social_base * rel;
        }
        q[s][demo_a] = q[s][demo_a] + alpha_social_effective * (pseudo_reward - q[s][demo_a]);
      }

      if (actor == 1) {
        vector[A] utilities = rep_vector(0.0, A);
        vector[A] q_vector = q[s];
        vector[A] g_vector = rep_vector(0.0, A);

        if (flag_include_policy_in_decision == 1 || flag_use_shared_mix == 1 || flag_use_independent_mix == 1) {
          g_vector = normalized_policy_drive(policy_pi[s], A);
        }

        if (flag_use_shared_mix == 1) {
          vector[A] drive = mix_weight * q_vector + (1.0 - mix_weight) * g_vector;
          utilities = beta * drive;
        } else if (flag_use_independent_mix == 1) {
          utilities = beta_q * q_vector + beta_policy * g_vector;
        } else {
          if (flag_include_q_in_decision == 1) {
            utilities += beta * q_vector;
          }
          if (flag_include_policy_in_decision == 1) {
            utilities += beta * g_vector;
          }
        }

        if (flag_last_choice == 1 && last_self_choice[s] > 0) {
          utilities[last_self_choice[s]] += kappa;
        }

        if (flag_recent_demo_choice == 1 && recent_demo_choice[s] > 0) {
          real demo_bonus = demo_bias;
          if (reliability_for_demo_bias == 1) {
            demo_bonus = demo_bias_rel * determinism_reliability(policy_pi[s], A);
          } else if (reliability_for_demo_bias == 2) {
            vector[A] count_probs = count_vals[s] / sum(count_vals[s]);
            demo_bonus = demo_bias_rel * determinism_reliability(count_probs, A);
          }
          utilities[recent_demo_choice[s]] += demo_bonus;
        }

        for (a in 1:A) {
          if (is_available[b, t, a] == 0) {
            utilities[a] = -1e12;
          }
        }

        target += categorical_logit_lpmf(a_obs | utilities);

        if (flag_last_choice == 1) {
          last_self_choice[s] = a_obs;
        }
        if (flag_subject_outcome_learning == 1 && has_reward[b, t] == 1) {
          q[s][a_obs] = q[s][a_obs] + alpha_self * (reward[b, t] - q[s][a_obs]);
        }
      } else {
        if (flag_recent_demo_choice == 1) {
          recent_demo_choice[s] = a_obs;
        }

        if (flag_policy_learning == 1) {
          for (a in 1:A) {
            real target_prob = 0.0;
            if (a == a_obs) {
              target_prob = 1.0;
            }
            policy_pi[s][a] = policy_pi[s][a] + alpha_policy * (target_prob - policy_pi[s][a]);
          }

          {
            real total = sum(policy_pi[s]);
            if (total <= 1e-12) {
              policy_pi[s] = rep_vector(1.0 / A, A);
            } else {
              policy_pi[s] = policy_pi[s] / total;
            }
          }
        }

        if (reliability_for_social_shaping == 2 || reliability_for_demo_bias == 2) {
          count_vals[s][a_obs] = count_vals[s][a_obs] + 1.0;
        }

        if (flag_social_shaping_on_demo == 1) {
          real rel = 1.0;
          if (reliability_for_social_shaping == 1) {
            rel = determinism_reliability(policy_pi[s], A);
          } else if (reliability_for_social_shaping == 2) {
            vector[A] count_probs = count_vals[s] / sum(count_vals[s]);
            rel = determinism_reliability(count_probs, A);
          }

          real alpha_social_effective = alpha_social;
          if (reliability_for_social_shaping != 0) {
            alpha_social_effective = alpha_social_base * rel;
          }
          q[s][a_obs] = q[s][a_obs] + alpha_social_effective * (pseudo_reward - q[s][a_obs]);
        }

        if (flag_demo_outcome_learning == 1 && has_reward[b, t] == 1) {
          q[s][a_obs] = q[s][a_obs] + alpha_observed * (reward[b, t] - q[s][a_obs]);
        }
      }
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
    real alpha_self = fixed_alpha_self;
    real alpha_observed = fixed_alpha_observed;
    real alpha_social = fixed_alpha_social;
    real alpha_policy = fixed_alpha_policy;
    real alpha_social_base = fixed_alpha_social_base;
    real beta = fixed_beta;
    real beta_q = fixed_beta_q;
    real beta_policy = fixed_beta_policy;
    real kappa = fixed_kappa;
    real mix_weight = fixed_mix_weight;
    real demo_bias = fixed_demo_bias;
    real demo_bias_rel = fixed_demo_bias_rel;
    real demo_dirichlet_prior = fixed_demo_dirichlet_prior;
    real initial_value = fixed_initial_value;
    real pseudo_reward = fixed_pseudo_reward;

    for (k in 1:K) {
      if (param_codes[k] == 1) {
        alpha_self = block_param[b][k];
      } else if (param_codes[k] == 2) {
        alpha_observed = block_param[b][k];
      } else if (param_codes[k] == 3) {
        alpha_social = block_param[b][k];
      } else if (param_codes[k] == 4) {
        alpha_policy = block_param[b][k];
      } else if (param_codes[k] == 5) {
        alpha_social_base = block_param[b][k];
      } else if (param_codes[k] == 6) {
        beta = block_param[b][k];
      } else if (param_codes[k] == 7) {
        beta_q = block_param[b][k];
      } else if (param_codes[k] == 8) {
        beta_policy = block_param[b][k];
      } else if (param_codes[k] == 9) {
        kappa = block_param[b][k];
      } else if (param_codes[k] == 10) {
        mix_weight = block_param[b][k];
      } else if (param_codes[k] == 11) {
        demo_bias = block_param[b][k];
      } else if (param_codes[k] == 12) {
        demo_bias_rel = block_param[b][k];
      } else if (param_codes[k] == 13) {
        demo_dirichlet_prior = block_param[b][k];
      } else if (param_codes[k] == 14) {
        initial_value = block_param[b][k];
      } else if (param_codes[k] == 15) {
        pseudo_reward = block_param[b][k];
      }
    }

    array[S] vector[A] q;
    array[S] vector[A] policy_pi;
    array[S] vector[A] count_vals;
    array[S] int last_self_choice;
    array[S] int recent_demo_choice;

    for (s in 1:S) {
      for (a in 1:A) {
        q[s][a] = initial_value;
        policy_pi[s][a] = 1.0 / A;
        count_vals[s][a] = demo_dirichlet_prior;
      }
      last_self_choice[s] = 0;
      recent_demo_choice[s] = 0;
    }

    for (t in 1:T[b]) {
      int s = state_idx[b, t];
      int a_obs = action_idx[b, t];
      int actor = actor_code[b, t];

      if (flag_social_shaping_from_subject_observation == 1 && actor == 1 && obs_demo_action_idx[b, t] > 0) {
        int demo_a = obs_demo_action_idx[b, t];
        real rel = 1.0;
        if (reliability_for_social_shaping == 1) {
          rel = determinism_reliability(policy_pi[s], A);
        } else if (reliability_for_social_shaping == 2) {
          vector[A] count_probs = count_vals[s] / sum(count_vals[s]);
          rel = determinism_reliability(count_probs, A);
        }

        real alpha_social_effective = alpha_social;
        if (reliability_for_social_shaping != 0) {
          alpha_social_effective = alpha_social_base * rel;
        }
        q[s][demo_a] = q[s][demo_a] + alpha_social_effective * (pseudo_reward - q[s][demo_a]);
      }

      if (actor == 1) {
        vector[A] utilities = rep_vector(0.0, A);
        vector[A] q_vector = q[s];
        vector[A] g_vector = rep_vector(0.0, A);

        if (flag_include_policy_in_decision == 1 || flag_use_shared_mix == 1 || flag_use_independent_mix == 1) {
          g_vector = normalized_policy_drive(policy_pi[s], A);
        }

        if (flag_use_shared_mix == 1) {
          vector[A] drive = mix_weight * q_vector + (1.0 - mix_weight) * g_vector;
          utilities = beta * drive;
        } else if (flag_use_independent_mix == 1) {
          utilities = beta_q * q_vector + beta_policy * g_vector;
        } else {
          if (flag_include_q_in_decision == 1) {
            utilities += beta * q_vector;
          }
          if (flag_include_policy_in_decision == 1) {
            utilities += beta * g_vector;
          }
        }

        if (flag_last_choice == 1 && last_self_choice[s] > 0) {
          utilities[last_self_choice[s]] += kappa;
        }

        if (flag_recent_demo_choice == 1 && recent_demo_choice[s] > 0) {
          real demo_bonus = demo_bias;
          if (reliability_for_demo_bias == 1) {
            demo_bonus = demo_bias_rel * determinism_reliability(policy_pi[s], A);
          } else if (reliability_for_demo_bias == 2) {
            vector[A] count_probs = count_vals[s] / sum(count_vals[s]);
            demo_bonus = demo_bias_rel * determinism_reliability(count_probs, A);
          }
          utilities[recent_demo_choice[s]] += demo_bonus;
        }

        for (a in 1:A) {
          if (is_available[b, t, a] == 0) {
            utilities[a] = -1e12;
          }
        }

        log_likelihood_total += categorical_logit_lpmf(a_obs | utilities);

        if (flag_last_choice == 1) {
          last_self_choice[s] = a_obs;
        }
        if (flag_subject_outcome_learning == 1 && has_reward[b, t] == 1) {
          q[s][a_obs] = q[s][a_obs] + alpha_self * (reward[b, t] - q[s][a_obs]);
        }
      } else {
        if (flag_recent_demo_choice == 1) {
          recent_demo_choice[s] = a_obs;
        }

        if (flag_policy_learning == 1) {
          for (a in 1:A) {
            real target_prob = 0.0;
            if (a == a_obs) {
              target_prob = 1.0;
            }
            policy_pi[s][a] = policy_pi[s][a] + alpha_policy * (target_prob - policy_pi[s][a]);
          }

          {
            real total = sum(policy_pi[s]);
            if (total <= 1e-12) {
              policy_pi[s] = rep_vector(1.0 / A, A);
            } else {
              policy_pi[s] = policy_pi[s] / total;
            }
          }
        }

        if (reliability_for_social_shaping == 2 || reliability_for_demo_bias == 2) {
          count_vals[s][a_obs] = count_vals[s][a_obs] + 1.0;
        }

        if (flag_social_shaping_on_demo == 1) {
          real rel = 1.0;
          if (reliability_for_social_shaping == 1) {
            rel = determinism_reliability(policy_pi[s], A);
          } else if (reliability_for_social_shaping == 2) {
            vector[A] count_probs = count_vals[s] / sum(count_vals[s]);
            rel = determinism_reliability(count_probs, A);
          }

          real alpha_social_effective = alpha_social;
          if (reliability_for_social_shaping != 0) {
            alpha_social_effective = alpha_social_base * rel;
          }
          q[s][a_obs] = q[s][a_obs] + alpha_social_effective * (pseudo_reward - q[s][a_obs]);
        }

        if (flag_demo_outcome_learning == 1 && has_reward[b, t] == 1) {
          q[s][a_obs] = q[s][a_obs] + alpha_observed * (reward[b, t] - q[s][a_obs]);
        }
      }
    }
  }

  log_posterior_total = log_prior_total + log_likelihood_total;
}
