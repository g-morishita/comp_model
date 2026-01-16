# Value-Shaping (VS)

This directory contains the implementation of the Value-Shaping model used for social RL imitation.

## Model summary
- State: action values `Q[a]` for `a in {0..K-1}`
- Choice: softmax over `beta * Q + perseveration`
- Social shaping: demonstration acts as pseudo-reward

## Update rules (implementation)
### Private outcome update (chosen-only)
Q[a_t] <- Q[a_t] + alpha_p * (r_t - Q[a_t])

### Social shaping update (chosen-only)
Q[d_t] <- Q[d_t] + alpha_i * (pseudo_reward - Q[d_t])

## Parameters
- alpha_p in (0,1)
- alpha_i in (0,1)
- beta > 0
- kappa in R

## Notes on K-armed generalization
The original paper presents the model in a 2-action form. This implementation generalizes to K actions using chosen-only updates.

## Data requirements
Trials should provide:
- choice `a_t` and reward `r_t`
- optionally `others_choices` (demonstrator action), and `others_rewards` if observed

## Likelihood/replay timing
This library’s replay uses:
1) (optional) social_update
2) choice likelihood
3) private update (if reward present)

## Estimation
- Recommended bounds for MLE
- Typical optimizer settings
- Identifiability tips for recovery (probabilistic bandit, demonstrator noise, trials/block)

## References
Najar et al. (2020) ...
