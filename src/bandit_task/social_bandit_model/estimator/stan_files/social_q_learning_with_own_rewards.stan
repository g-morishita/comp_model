data {
  int<lower=1> T; // The number of trials
  int<lower=1> NC; // The number of unique choices
  array[T] int C; // Choices
  array[T] int R; // Rewards
  array[T] int PC; // Partner's choice
  array[T] int PR; // Partner's rewards
}


parameters {
  real<lower=0.0,upper=1.0> alpha_own;
  real<lower=0.0,upper=1.0> alpha_other;
  real<lower=0.0> beta;
}

model {
  vector[NC] Q; // Q values
  alpha_own ~ normal(0, 10);
  alpha_other ~ normal(0, 10);
  beta ~ normal(0, 10);


  // Initialize Q values
  for (k in 1:NC) {
    Q[k] = 0.5;
  }

  for ( t in 1:T ) { // trial
    // Update Q value according to partner's choice and reward.
    Q[PC[t]] = Q[PC[t]] + alpha_other * (PR[t] - Q[PC[t]]);

    // Add the likelihood according to your own choice
    target += log_softmax(beta * Q)[C[t]];

    // Update Q value according to your own choice and reward.
    Q[C[t]] = Q[C[t]] + alpha_own * (R[t] - Q[C[t]]);
  }
}
