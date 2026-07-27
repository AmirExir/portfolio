---
name: reinforcement-learning
description: Design, implement, and evaluate reinforcement-learning environments and agents for grid control, operations, planning, and educational projects.
---

# Reinforcement Learning

## Use this skill when

Use this skill for MDP design, value functions, policy optimization, TD learning, Monte Carlo methods, function approximation, control environments, or grid-focused RL experiments.

## Required workflow

1. Define state, action, transition, reward, discount, and termination conditions.
2. Explain why RL is appropriate instead of optimization, control, or supervised learning.
3. Build a deterministic or simplified baseline environment first.
4. Validate environment dynamics independently of the agent.
5. Establish random, heuristic, and optimization-based baselines where possible.
6. Train across multiple seeds.
7. Evaluate on unseen scenarios.
8. Report constraint violations, not just reward.
9. Inspect learned behavior for physical plausibility.

## Grid-specific rules

- Encode hard safety constraints explicitly where possible.
- Do not rely on reward penalties alone for critical reliability limits.
- Preserve units and operating bounds.
- Distinguish simulation reward from real engineering value.
- Test N-1, load variation, renewable uncertainty, and topology sensitivity when relevant.
- Avoid claiming deployability from simulation-only performance.

## Evaluation

Track:

- episodic return
- success rate
- constraint violations
- sample efficiency
- robustness across seeds
- out-of-distribution performance
- action smoothness or switching frequency

## Implementation rules

- Keep environment, agent, training loop, and evaluation separate.
- Seed all stochastic components.
- Save checkpoints and configuration.
- Add unit tests for transitions, rewards, and termination.
- Detect NaNs, divergence, and invalid actions.

## Completion format

Report:

- MDP definition
- Baselines
- Training setup
- Evaluation
- Constraint performance
- Limitations
