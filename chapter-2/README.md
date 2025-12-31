# Chapter 2: Multi-armed Bandits

## Contents

| File | Description |
|------|-------------|
| `epsilon_greedy_bandit.py` | ε-greedy comparison across action value distributions |
| `exercise_2_5_nonstationary.py` | Sample-average vs constant step-size on nonstationary bandit |
| `exercise_2_8_ucb_analysis.py` | UCB action selection simulation (1000 runs) |
| `exercise_2_10_associative_search.py` | Contextual bandit / associative search |
| `exercise_2_11_parameter_study.py` | Parameter study for nonstationary bandit (vectorized) |
| `EXERCISE.md` | Exercise solutions and notes |

---

## Epsilon-Greedy Bandit

Compares ε-greedy (ε = 0, 0.01, 0.05, 0.1, 0.2, 0.5) on 10-armed bandit with different q*(a) distributions.

```bash
python3 epsilon_greedy_bandit.py
```

| Distribution | q*(a) | Result |
|--------------|-------|--------|
| Gaussian | N(0, 1) | ![](epsilon_greedy_gaussian.png) |
| Uniform | evenly spaced [-2, 2] | ![](epsilon_greedy_uniform.png) |
| Random | Uniform(-2, 2) | ![](epsilon_greedy_random.png) |

---

## Exercise 2.5: Nonstationary Bandit

Demonstrates sample-average failure on nonstationary problems where q*(a) takes random walks.

```bash
python3 exercise_2_5_nonstationary.py
```

![](exercise_2_5_nonstationary.png)

**Key insight:** Constant step-size (α=0.1) tracks changes; sample-average cannot adapt.

---

## Exercise 2.8: UCB Analysis

UCB action selection: `A = argmax[Q(a) + c√(ln(t)/N(a))]`

```bash
python3 exercise_2_8_ucb_analysis.py
```

![](exercise_2_8_ucb_analysis.png)

**Key insight:** UCB explores all arms first (steps 1-10), spikes at step 11, then gradually improves.

---

## Exercise 2.10: Associative Search

2-armed bandit with context (Case A or B). Shows value of contextual information.

```bash
python3 exercise_2_10_associative_search.py
```

![](exercise_2_10_associative_search.png)

| Setting | Best Strategy | Expected Reward |
|---------|---------------|-----------------|
| Non-associative | Any action | 0.50 |
| Associative | Case A→action 2, Case B→action 1 | 0.55 |

---

## Exercise 2.11: Parameter Study

Figure 2.6 analog for nonstationary bandit. Vectorized for ~50x speedup.

```bash
python3 exercise_2_11_parameter_study.py
```

![](exercise_2_11_parameter_study.png)

**Key insight:** Constant step-size methods dominate in nonstationary environments.
