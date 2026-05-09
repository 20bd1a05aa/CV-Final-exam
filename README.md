# CV-Final-exam

# REINFORCE Small Language Model Credit Assignment

## Overview

This assignment implements the REINFORCE policy-gradient algorithm from scratch on a small autoregressive character-level language model.

The objective is to understand:

* Reinforcement learning for sequence generation
* Policy gradient optimization
* Delayed rewards
* Temporal credit assignment
* Variance reduction using baselines

The model learns to generate sequences containing the substring:

```text
abc
```

using only a sparse terminal reward.

---

# Environment Setup

## Dependencies

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
```

---

# Vocabulary

The language model operates on a very small vocabulary:

```python
vocab = ['<bos>', 'a', 'b', 'c', '<eos>']
```

Where:

* `<bos>` = beginning of sequence
* `<eos>` = end of sequence

Mappings:

```python
stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for c, i in stoi.items()}
```

---

# Reward Function

The environment provides reward only at the end of the generated trajectory.

## Reward Definition

[
R(\tau) =
\begin{cases}
1 & \text{if the substring abc appears in the sampled sequence} \
0 & \text{otherwise}
\end{cases}
]

Example:

| Sequence                   | Reward |
| -------------------------- | ------ |
| `['a', 'b', 'c', '<eos>']` | 1      |
| `['a', 'c', 'b']`          | 0      |
| `['b', 'a', 'c']`          | 0      |

This creates a sparse delayed-reward reinforcement learning problem.

---

# Task 1 — Policy Network

## Objective

Implement a memoryless character-level policy:

[
\pi_\theta(a_t | a_{t-1})
]

## Architecture

The model consists of:

1. Embedding layer
2. Linear projection layer

Implementation:

```python
class CharPolicy(nn.Module):
    def __init__(self, embed_dim: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        logits = self.linear(emb)
        return logits
```

## Why Memorylessness Works

The optimal policy is deterministic:

```text
<bos> -> a
a -> b
b -> c
c -> <eos>
```

Each state uniquely determines the next token, so recurrent memory is unnecessary.

---

# Task 2 — Trajectory Sampling

## Objective

Sample sequences autoregressively from the policy.

At every timestep:

1. Compute logits
2. Construct categorical distribution
3. Sample next token
4. Store log probability
5. Stop at `<eos>` or `max_len`

Implementation:

```python
def sample(model: CharPolicy, max_len: int = 8):
    seq = []
    log_probs = []

    current = torch.tensor([stoi['<bos>']], dtype=torch.long)

    for _ in range(max_len):
        logits = model(current)
        dist = torch.distributions.Categorical(logits=logits.squeeze(0))

        token = dist.sample()
        log_prob = dist.log_prob(token)

        token_str = itos[token.item()]

        seq.append(token_str)
        log_probs.append(log_prob)

        current = token.view(1)

        if token_str == '<eos>':
            break

    return seq, torch.stack(log_probs)
```

---

# Task 3 — Reward Function

## Objective

Detect whether the generated sequence contains the substring `abc`.

Implementation:

```python
def reward(seq: list[str]) -> float:
    s = ''.join(tok for tok in seq if tok != '<eos>')
    return 1.0 if 'abc' in s else 0.0
```

---

# Task 4 — REINFORCE Without Baseline

## Objective

Train the policy using the REINFORCE policy gradient algorithm.

## Policy Gradient Objective

[
\nabla_\theta J(\theta)
=======================

\mathbb{E}*\tau
\left[
\sum*{t=1}^{T}
R(\tau)
\nabla_\theta
\log \pi_\theta(a_t | a_{<t})
\right]
]

## Loss Function

[
\mathcal{L}(\theta)
===================

-R(\tau)
\sum_t
\log \pi_\theta(a_t | a_{<t})
]

## Training Procedure

For each iteration:

1. Sample trajectory
2. Compute reward
3. Compute policy-gradient loss
4. Backpropagate
5. Update parameters

Implementation:

```python
for step in range(1500):
    seq, log_probs = sample(model_no_baseline)

    R = reward(seq)

    loss = -R * log_probs.sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Observation

Learning is unstable because rewards are sparse.

Many sampled trajectories receive:

```text
reward = 0
```

which produces zero gradient updates.

---

# Task 4b — REINFORCE With Baseline

## Objective

Reduce gradient variance using a running-mean baseline.

## Advantage Function

Instead of using:

[
R(\tau)
]

use:

[
A(\tau) = R(\tau) - b
]

where:

[
b = 0.9b + 0.1R
]

## Updated Loss

[
\mathcal{L}(\theta)
===================

-(R(\tau)-b)
\sum_t
\log \pi_\theta(a_t | a_{<t})
]

Implementation:

```python
baseline = 0.0

for step in range(1500):
    seq, log_probs = sample(model_with_baseline)

    R = reward(seq)

    advantage = R - baseline

    loss = -advantage * log_probs.sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    baseline = 0.9 * baseline + 0.1 * R
```

## Result

The baseline significantly stabilizes training and accelerates convergence.

Expected outcome:

```text
Final 100-step average reward ≈ 1.0
```

---

# Task 5 — Credit Assignment Analysis

## Learned Conditional Distributions

After training, the model learns approximately:

```text
π(a | <bos>) ≈ 1
π(b | a) ≈ 1
π(c | b) ≈ 1
π(<eos> | c) ≈ 1
```

This corresponds to the optimal sequence:

```text
a -> b -> c -> <eos>
```

---

# Temporal Credit Assignment

## Key Idea

The reward is observed only at the end of the sequence, but earlier actions still receive gradient signal.

The REINFORCE loss contains:

[
-R(\tau) \log \pi(a_1 | <bos>)
]

Therefore:

* successful trajectories increase the probability of all sampled actions
* earlier actions are reinforced even though reward arrives later

Example successful trajectory:

```text
a b c <eos>
```

The algorithm increases:

* (\log \pi(a|<bos>))
* (\log \pi(b|a))
* (\log \pi(c|b))
* (\log \pi(<eos>|c))

simultaneously.

This mechanism is called temporal credit assignment.

---

# Final Results

## Achieved Results

* Successful implementation of REINFORCE
* Successful implementation of running-mean baseline
* Final average reward reached approximately 1.0
* Model learned optimal token transitions
* Temporal credit assignment demonstrated successfully

---

# Key Concepts Learned

This assignment demonstrates:

* Policy-gradient reinforcement learning
* Sequence generation with RL
* Sparse delayed rewards
* Monte Carlo policy gradients
* Variance reduction using baselines
* Credit assignment across time
* Autoregressive language modeling

---

# Conclusion

This project provides a minimal but complete demonstration of reinforcement learning for language models.

Although the vocabulary and reward function are extremely small, the same core principles are used in larger RL systems such as:

* RLHF (Reinforcement Learning from Human Feedback)
* Preference optimization
* Language-model fine-tuning with rewards
* Sequence-level reinforcement learning

The assignment illustrates how a scalar terminal reward can propagate learning signal backward through an entire generated trajectory using policy gradients.
