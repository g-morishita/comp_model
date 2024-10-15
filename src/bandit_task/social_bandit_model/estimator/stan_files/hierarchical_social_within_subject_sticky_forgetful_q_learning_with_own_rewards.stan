data {
  int<lower=1> N;          // Number of participants
  int<lower=1> S;          // Number of sessions per participant
  int<lower=1> T;          // Number of trials per session
  int<lower=1> NC;         // Number of unique choices/actions
  int<lower=1, upper=NC> C[N, S, T];          // Your own choices
  real R[N, S, T];         // Your own rewards
  int<lower=1, upper=NC> PC[N, S, T];         // Partner's choices
  real PR[N, S, T];        // Partner's rewards
  int<lower=1, upper=2> condition[N, S];      // Condition indicator (1 = A, 2 = B)
}

parameters {
  // Individual-level latent parameters (standard normal)
  vector[N] alpha_own_nd;
  vector[N] alpha_partner_nd;
  vector[N] beta_nd;
  vector[N] forgetful_own_nd;
  vector[N] forgetful_partner_nd;
  vector[N] s_own_nd;
  vector[N] s_partner_nd;

  // Delta parameters (differences between conditions) on transformed scales
  vector[N] delta_alpha_own;        // On logit scale for [0,1] parameters
  vector[N] delta_alpha_partner;    // On logit scale for [0,1] parameters
  vector[N] delta_beta;             // On log scale for >0 parameters
  vector[N] delta_forgetful_own;    // On logit scale for [0,1] parameters
  vector[N] delta_forgetful_partner;// On logit scale for [0,1] parameters
  vector[N] delta_s_own;            // Unbounded
  vector[N] delta_s_partner;        // Unbounded

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

  // Group-level parameters for delta terms
  real mu_delta_alpha_own;
  real<lower=0> sigma_delta_alpha_own;
  real mu_delta_alpha_partner;
  real<lower=0> sigma_delta_alpha_partner;
  real mu_delta_beta;
  real<lower=0> sigma_delta_beta;
  real mu_delta_forgetful_own;
  real<lower=0> sigma_delta_forgetful_own;
  real mu_delta_forgetful_partner;
  real<lower=0> sigma_delta_forgetful_partner;
  real mu_delta_s_own;
  real<lower=0> sigma_delta_s_own;
  real mu_delta_s_partner;
  real<lower=0> sigma_delta_s_partner;
}

transformed parameters {
  // Condition A Parameters
  vector<lower=0, upper=1>[N] alpha_own_A;
  vector<lower=0, upper=1>[N] alpha_partner_A;
  vector<lower=0>[N] beta_A;
  vector<lower=0, upper=1>[N] forgetful_own_A;
  vector<lower=0, upper=1>[N] forgetful_partner_A;
  vector[N] s_own_A;
  vector[N] s_partner_A;

  // Transform individual-level parameters to appropriate scales for Condition A
  alpha_own_A = inv_logit(mu_alpha_own_nd + sigma_alpha_own_nd * alpha_own_nd);
  alpha_partner_A = inv_logit(mu_alpha_partner_nd + sigma_alpha_partner_nd * alpha_partner_nd);
  beta_A = exp(mu_beta_nd + sigma_beta_nd * beta_nd);
  forgetful_own_A = inv_logit(mu_forgetful_own_nd + sigma_forgetful_own_nd * forgetful_own_nd);
  forgetful_partner_A = inv_logit(mu_forgetful_partner_nd + sigma_forgetful_partner_nd * forgetful_partner_nd);
  s_own_A = mu_s_own + sigma_s_own * s_own_nd;
  s_partner_A = mu_s_partner + sigma_s_partner * s_partner_nd;

  // Compute transformed Condition A parameters
  vector[N] logit_alpha_own_A = log(alpha_own_A ./ (1 - alpha_own_A));
  vector[N] logit_alpha_partner_A = log(alpha_partner_A ./ (1 - alpha_partner_A));
  vector[N] log_beta_A = log(beta_A);
  vector[N] logit_forgetful_own_A = log(forgetful_own_A ./ (1 - forgetful_own_A));
  vector[N] logit_forgetful_partner_A = log(forgetful_partner_A ./ (1 - forgetful_partner_A));

  // Condition B Parameters: add deltas on transformed scales
  vector[N] logit_alpha_own_B = logit_alpha_own_A + delta_alpha_own;
  vector<lower=0, upper=1>[N] alpha_own_B = inv_logit(logit_alpha_own_B);

  vector[N] logit_alpha_partner_B = logit_alpha_partner_A + delta_alpha_partner;
  vector<lower=0, upper=1>[N] alpha_partner_B = inv_logit(logit_alpha_partner_B);

  vector[N] log_beta_B = log_beta_A + delta_beta;
  vector<lower=0>[N] beta_B = exp(log_beta_B);

  vector[N] logit_forgetful_own_B = logit_forgetful_own_A + delta_forgetful_own;
  vector<lower=0, upper=1>[N] forgetful_own_B = inv_logit(logit_forgetful_own_B);

  vector[N] logit_forgetful_partner_B = logit_forgetful_partner_A + delta_forgetful_partner;
  vector<lower=0, upper=1>[N] forgetful_partner_B = inv_logit(logit_forgetful_partner_B);

  // For unbounded parameters, add deltas directly
  vector[N] s_own_B = s_own_A + delta_s_own;
  vector[N] s_partner_B = s_partner_A + delta_s_partner;
}

model {
  // Priors for individual-level latent parameters
  alpha_own_nd ~ normal(0, 1);
  alpha_partner_nd ~ normal(0, 1);
  beta_nd ~ normal(0, 1);
  forgetful_own_nd ~ normal(0, 1);
  forgetful_partner_nd ~ normal(0, 1);
  s_own_nd ~ normal(0, 1);
  s_partner_nd ~ normal(0, 1);

  // Priors for delta parameters on transformed scales
  delta_alpha_own ~ normal(mu_delta_alpha_own, sigma_delta_alpha_own);
  delta_alpha_partner ~ normal(mu_delta_alpha_partner, sigma_delta_alpha_partner);
  delta_beta ~ normal(mu_delta_beta, sigma_delta_beta);
  delta_forgetful_own ~ normal(mu_delta_forgetful_own, sigma_delta_forgetful_own);
  delta_forgetful_partner ~ normal(mu_delta_forgetful_partner, sigma_delta_forgetful_partner);
  delta_s_own ~ normal(mu_delta_s_own, sigma_delta_s_own);
  delta_s_partner ~ normal(mu_delta_s_partner, sigma_delta_s_partner);

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

  // Priors for delta group-level parameters
  mu_delta_alpha_own ~ normal(0, 1);
  sigma_delta_alpha_own ~ cauchy(0, 2.5);
  mu_delta_alpha_partner ~ normal(0, 1);
  sigma_delta_alpha_partner ~ cauchy(0, 2.5);
  mu_delta_beta ~ normal(0, 1);
  sigma_delta_beta ~ cauchy(0, 2.5);
  mu_delta_forgetful_own ~ normal(0, 1);
  sigma_delta_forgetful_own ~ cauchy(0, 2.5);
  mu_delta_forgetful_partner ~ normal(0, 1);
  sigma_delta_forgetful_partner ~ cauchy(0, 2.5);
  mu_delta_s_own ~ normal(0, 1);
  sigma_delta_s_own ~ cauchy(0, 2.5);
  mu_delta_s_partner ~ normal(0, 1);
  sigma_delta_s_partner ~ cauchy(0, 2.5);

  // Likelihood
  for (i in 1:N) { // Participant loop
    for (j in 1:S) { // Session loop
      vector[NC] Q; // Q-values for each action
      vector[NC] initial_values; // Initial Q-values
      int current_condition = condition[i, j]; // Condition of the current session
      int prev_own_choice = 0; // Initialize previous own choice
      int prev_partner_choice = 0; // Initialize previous partner choice

      // Initialize Q-values to 0.5
      Q = rep_vector(0.5, NC);
      initial_values = rep_vector(0.5, NC);

      for (t in 1:T) { // Trial loop
        vector[NC] action_values;
        vector[NC] log_probs;

        // Declare scalar variables for condition-specific parameters
        real alpha_own_cond;
        real alpha_partner_cond;
        real beta_cond;
        real forgetful_own_cond;
        real forgetful_partner_cond;
        real s_own_cond;
        real s_partner_cond;

        // Assign condition-specific parameters based on the current condition
        if (current_condition == 1) { // Condition A
          alpha_own_cond = alpha_own_A[i];
          alpha_partner_cond = alpha_partner_A[i];
          beta_cond = beta_A[i];
          forgetful_own_cond = forgetful_own_A[i];
          forgetful_partner_cond = forgetful_partner_A[i];
          s_own_cond = s_own_A[i];
          s_partner_cond = s_partner_A[i];
        } else if (current_condition == 2) { // Condition B
          alpha_own_cond = alpha_own_B[i];
          alpha_partner_cond = alpha_partner_B[i];
          beta_cond = beta_B[i];
          forgetful_own_cond = forgetful_own_B[i];
          forgetful_partner_cond = forgetful_partner_B[i];
          s_own_cond = s_own_B[i];
          s_partner_cond = s_partner_B[i];
        }

        // Compute action values with information bonus
        // Adjust the bonus term as needed; here it's set to 1.0 for illustration
        action_values = Q + (1.0) ./ sqrt(rep_vector(1.0, NC)); // Example bonus term

        // Add stickiness effects if previous choices are available
        if (prev_own_choice > 0) {
          action_values[prev_own_choice] += s_own_cond;
        }
        if (prev_partner_choice > 0) {
          action_values[prev_partner_choice] += s_partner_cond;
        }

        // Compute log probabilities using softmax
        log_probs = log_softmax(beta_cond * action_values);

        // Update the target log likelihood with own choice
        target += log_probs[C[i, j, t]];

        // Update previous choices
        prev_own_choice = C[i, j, t];
        prev_partner_choice = PC[i, j, t];

        // Copy current Q-values for updates
        vector[NC] Q_next = Q;

        // Update Q-values for own choice
        Q_next[C[i, j, t]] += alpha_own_cond * (R[i, j, t] - Q[C[i, j, t]]);

        // Apply forgetting to other actions after own choice
        for (k in 1:NC) {
          if (k != C[i, j, t]) {
            Q_next[k] = forgetful_own_cond * initial_values[k] + (1 - forgetful_own_cond) * Q_next[k];
          }
        }

        // Update Q-values for partner's choice
        Q_next[PC[i, j, t]] += alpha_partner_cond * (PR[i, j, t] - Q_next[PC[i, j, t]]);

        // Apply forgetting to other actions after partner's choice
        for (k in 1:NC) {
          if (k != PC[i, j, t]) {
            Q_next[k] = forgetful_partner_cond * initial_values[k] + (1 - forgetful_partner_cond) * Q_next[k];
          }
        }

        // Update Q-values for the next trial
        Q = Q_next;
      }
    }
  }
}
