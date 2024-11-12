data {
  int<lower=1> T; // The number of trials
  int<lower=1> S; // The number of sessions
  int<lower=1> NC; // The number of unique choices
  array[S, T] int C; // Choices
  array[S, T] int PC; // Partner's choice
  array[S, T] int PR; // Partner's rewards
}
parameters {
  real<lower=0.0,upper=1.0> alpha;
  real<lower=0.0> beta;
}
model {
  vector[NC] Q; // Q values
  // Initialize Q values
  for (s in 1:S) { // session
      for (k in 1:NC) {
        Q[k] = 0.5;
      }
      for ( t in 1:T ) { // trial
        // Update Q value according to partner's choice and reward.
        Q[PC[s, t]] = Q[PC[s, t]] + alpha * (PR[s, t] - Q[PC[s, t]]);

        // Add the likelihood according to your own choice
        target += log_softmax(beta * Q)[C[s, t]];
      }
  }
}
