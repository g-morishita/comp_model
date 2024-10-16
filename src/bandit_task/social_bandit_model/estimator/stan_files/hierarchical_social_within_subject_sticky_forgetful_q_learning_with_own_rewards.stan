data {
  int<lower=1> N; // Number of participants
  int<lower=1> S; // Number of sessions per participant
  int<lower=1> T; // Number of trials per session
  int<lower=1> NC; // Number of unique choices/actions

  // Multi-dimensional array declarations using your working syntax
  array[N, S, T] int<lower=1, upper=NC> C;   // Your own choices
  array[N, S, T] int R;                      // Your own rewards
  array[N, S, T] int<lower=1, upper=NC> PC;  // Partner's choices
  array[N, S, T] int PR;                     // Partner's rewards

  array[N, S] int<lower=1, upper=2> condition; // Condition indicator (1 = A, 2 = B)
}

parameters {
  // Individual-level latent parameters for each condition (A=1, B=2)
  array[N, 2] real alpha_own_nd;          // [N, Condition]
  array[N, 2] real alpha_partner_nd;      // [N, Condition]
  array[N, 2] real beta_nd;               // [N, Condition]
  array[N, 2] real forgetful_own_nd;      // [N, Condition]
  array[N, 2] real forgetful_partner_nd;  // [N, Condition]
  array[N, 2] real s_own_nd;              // [N, Condition]
  array[N, 2] real s_partner_nd;          // [N, Condition]

  // Group-level parameters (means and standard deviations)
  real mu_alpha_own_nd;
  real<lower=0> sigma_alpha_own_nd;
  real mu_alpha_partner_nd;
  real<lower=0> sigma_alpha_partner_nd;
  real mu_beta_nd;
  real<lower=0> sigma_beta_nd;
  real mu_forgetful_own_nd;
  real<lower=0> sigma_forgetful_own_nd;
  real mu_forgetful_partner_nd;
  real<lower=0> sigma_forgetful_partner_nd;
  real mu_s_own;
  real<lower=0> sigma_s_own;
  real mu_s_partner;
  real<lower=0> sigma_s_partner;
}

transformed parameters {
  // Transformed individual-level parameters for each condition
  array[N, 2] real alpha_own;          // [N, Condition] in [0,1]
  array[N, 2] real alpha_partner;      // [N, Condition] in [0,1]
  array[N, 2] real beta;               // [N, Condition] >0
  array[N, 2] real forgetful_own;      // [N, Condition] in [0,1]
  array[N, 2] real forgetful_partner;  // [N, Condition] in [0,1]
  array[N, 2] real s_own;              // [N, Condition] unbounded
  array[N, 2] real s_partner;          // [N, Condition] unbounded

  // Transform individual-level parameters to appropriate scales for each condition
  for (i in 1:N) {
    for (c in 1:2) { // c=1 for Condition A, c=2 for Condition B
      alpha_own[i, c] = inv_logit(mu_alpha_own_nd + sigma_alpha_own_nd * alpha_own_nd[i, c]);
      alpha_partner[i, c] = inv_logit(mu_alpha_partner_nd + sigma_alpha_partner_nd * alpha_partner_nd[i, c]);
      beta[i, c] = exp(mu_beta_nd + sigma_beta_nd * beta_nd[i, c]);
      forgetful_own[i, c] = inv_logit(mu_forgetful_own_nd + sigma_forgetful_own_nd * forgetful_own_nd[i, c]);
      forgetful_partner[i, c] = inv_logit(mu_forgetful_partner_nd + sigma_forgetful_partner_nd * forgetful_partner_nd[i, c]);
      s_own[i, c] = mu_s_own + sigma_s_own * s_own_nd[i, c];
      s_partner[i, c] = mu_s_partner + sigma_s_partner * s_partner_nd[i, c];
    }
  }
}

model {
  // Priors for individual-level latent parameters
  for (i in 1:N) {
    for (c in 1:2) { // c=1 for Condition A, c=2 for Condition B
      alpha_own_nd[i, c] ~ normal(0, 1);
      alpha_partner_nd[i, c] ~ normal(0, 1);
      beta_nd[i, c] ~ normal(0, 1);
      forgetful_own_nd[i, c] ~ normal(0, 1);
      forgetful_partner_nd[i, c] ~ normal(0, 1);
      s_own_nd[i, c] ~ normal(0, 1);
      s_partner_nd[i, c] ~ normal(0, 1);
    }
  }

  // Priors for group-level parameters
  mu_alpha_own_nd ~ normal(0, 1);
  sigma_alpha_own_nd ~ cauchy(0, 2.5);

  mu_alpha_partner_nd ~ normal(0, 1);
  sigma_alpha_partner_nd ~ cauchy(0, 2.5);

  mu_beta_nd ~ normal(0, 1);
  sigma_beta_nd ~ cauchy(0, 2.5);

  mu_forgetful_own_nd ~ normal(0, 1);
  sigma_forgetful_own_nd ~ cauchy(0, 2.5);

  mu_forgetful_partner_nd ~ normal(0, 1);
  sigma_forgetful_partner_nd ~ cauchy(0, 2.5);

  mu_s_own ~ normal(0, 1);
  sigma_s_own ~ cauchy(0, 2.5);

  mu_s_partner ~ normal(0, 1);
  sigma_s_partner ~ cauchy(0, 2.5);

  // Likelihood
  for (i in 1:N) { // Participant loop
    for (j in 1:S) { // Session loop
      vector[NC] Q; // Q-values for each action
      vector[NC] initial_values; // Initial Q-values
      int prev_own_choice = 0; // Initialize previous own choice
      int prev_partner_choice = 0; // Initialize previous partner choice

      // Initialize Q-values to 0.5
      Q = rep_vector(0.5, NC);
      initial_values = rep_vector(0.5, NC);

      for (t in 1:T) { // Trial loop
        vector[NC] action_values;
        vector[NC] log_probs;

        // Determine the current condition
        int current_condition = condition[i, j]; // 1 = A, 2 = B

        // Copy current Q-values
        action_values = Q;

        // Add stickiness effects based on previous choices
        if (prev_own_choice > 0) {
          action_values[prev_own_choice] += s_own[i, current_condition];
        }
        if (prev_partner_choice > 0) {
          action_values[prev_partner_choice] += s_partner[i, current_condition];
        }

        // Compute log probabilities using softmax
        log_probs = log_softmax(beta[i, current_condition] * action_values);

        // Update the target log likelihood with own choice
        target += log_probs[C[i, j, t]];

        // Update previous choices
        prev_own_choice = C[i, j, t];
        prev_partner_choice = PC[i, j, t];

        // Copy current Q-values for updates
        vector[NC] Q_next = Q;

        // Update Q-values for own choice
        Q_next[C[i, j, t]] += alpha_own[i, current_condition] * (R[i, j, t] - Q[C[i, j, t]]);

        // Apply forgetting to other actions after own choice
        for (k in 1:NC) {
          if (k != C[i, j, t]) {
            Q_next[k] = forgetful_own[i, current_condition] * initial_values[k] +
                        (1 - forgetful_own[i, current_condition]) * Q_next[k];
          }
        }

        // Update Q-values for partner's choice
        Q_next[PC[i, j, t]] += alpha_partner[i, current_condition] * (PR[i, j, t] - Q_next[PC[i, j, t]]);

        // Apply forgetting to other actions after partner's choice
        for (k in 1:NC) {
          if (k != PC[i, j, t]) {
            Q_next[k] = forgetful_partner[i, current_condition] * initial_values[k] +
                        (1 - forgetful_partner[i, current_condition]) * Q_next[k];
          }
        }

        // Update Q-values for the next trial
        Q = Q_next;
      }
    }
  }
}

generated quantities {
  // Transformed group-level parameters for interpretation
  real<lower=0, upper=1> mu_alpha_own = inv_logit(mu_alpha_own_nd);
  real<lower=0, upper=1> mu_alpha_partner = inv_logit(mu_alpha_partner_nd);
  real<lower=0> mu_beta = exp(mu_beta_nd);
  real<lower=0, upper=1> mu_forgetful_own = inv_logit(mu_forgetful_own_nd);
  real<lower=0, upper=1> mu_forgetful_partner = inv_logit(mu_forgetful_partner_nd);

  // Compute log-likelihood for each trial for model diagnostics
  vector[N * S * T] log_lik; // Log-likelihood for each trial
  int trial_count = 0;

  for (i in 1:N) {
    for (j in 1:S) {
      vector[NC] Q; // Q-values for each action
      vector[NC] initial_values;
      int prev_own_choice = 0;
      int prev_partner_choice = 0;

      // Initialize Q-values and initial values
      Q = rep_vector(0.5, NC);
      initial_values = rep_vector(0.5, NC);

      for (t in 1:T) {
        vector[NC] action_values;
        vector[NC] log_probs;

        // Determine the current condition
        int current_condition = condition[i, j]; // 1 = A, 2 = B

        // Copy current Q-values
        action_values = Q;

        // Add stickiness effects based on previous choices
        if (prev_own_choice > 0) {
          action_values[prev_own_choice] += s_own[i, current_condition];
        }
        if (prev_partner_choice > 0) {
          action_values[prev_partner_choice] += s_partner[i, current_condition];
        }

        // Compute log probabilities using softmax
        log_probs = log_softmax(beta[i, current_condition] * action_values);

        // Store log-likelihood for each trial
        trial_count += 1;
        log_lik[trial_count] = log_probs[C[i, j, t]];

        // Update previous choices
        prev_own_choice = C[i, j, t];
        prev_partner_choice = PC[i, j, t];

        // Copy current Q-values for updates
        vector[NC] Q_next = Q;

        // Update Q-values for own choice
        Q_next[C[i, j, t]] += alpha_own[i, current_condition] * (R[i, j, t] - Q[C[i, j, t]]);

        // Apply forgetting to other actions after own choice
        for (k in 1:NC) {
          if (k != C[i, j, t]) {
            Q_next[k] = forgetful_own[i, current_condition] * initial_values[k] +
                        (1 - forgetful_own[i, current_condition]) * Q_next[k];
          }
        }

        // Update Q-values for partner's choice
        Q_next[PC[i, j, t]] += alpha_partner[i, current_condition] * (PR[i, j, t] - Q_next[PC[i, j, t]]);

        // Apply forgetting to other actions after partner's choice
        for (k in 1:NC) {
          if (k != PC[i, j, t]) {
            Q_next[k] = forgetful_partner[i, current_condition] * initial_values[k] +
                        (1 - forgetful_partner[i, current_condition]) * Q_next[k];
          }
        }

        // Update Q-values for the next trial
        Q = Q_next;
      }
    }
  }
}
