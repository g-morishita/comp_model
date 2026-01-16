# Dynamic Value Shape Model

This model is a variant of the Value Shape (VS) model proposed by (Najar et al., PLOS Biology, 2020).
The action value is updates through vicarious learning.

## Task
A social version of multi-armed bandit task in which an individual learn from a demonstrator to make the optimal choice.
On each trial, the individual observes the demonstrator's action and reward, and then make their own choice.
Importantly, they do not observe the reward feedback of their own choice.

## Computational modeling
The model assigns the value to each option $V(X)$.
When it observes the demonstrator's action $X$, it updates the value $V(X) \leftarrow V(X) + \tau (1 - V(X))$ as if the observed option is rewarded.
When it observes the demonstrator's outcome $R$, it updates the value $V(X) \leftarrow V(X) + \alpha (R - V(X))$.

The model makes a choice according to a choice probability using the softmax function:

$$P(X) \sim \beta * \exp(V(X))$$.

## Reference
Najar, Anis, et al. "The actions of others act as a pseudo-reward to drive imitation in the context of social reinforcement learning." PLoS biology 18.12 (2020): e3001028.