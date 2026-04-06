## Probabilistic Classification

A binary alert system — *incident* or *no incident* — throws away valuable information.
We need to know not just *if* something is wrong, but *what* is wrong and *how confident* we are.

The solution: replace binary outputs with a **probability distribution.**

---

## What Is It? The Softmax Function

**Softmax** converts the raw outputs of the neural network (called logits)
into a probability distribution across all possible incident types.

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}$$

- Every output is between 0 and 1
- All outputs sum to exactly 1.0
- The model expresses *confidence*, not just a yes/no decision

---

## What I Learned: Using Confidence Scores

The probability score itself carries meaning:

- Model returns **0.51** → flag as "Uncertain" — escalate for human review
- Model returns **0.99** → flag as "Critical Outage" — trigger automated response

This nuance dramatically reduces false positives compared to binary classification.

---

## How We Implemented: Custom Output Layer

```python
exp_scores = np.exp(logits)
softmax = exp_scores / np.sum(exp_scores)
```

Built as a custom `OutputLayer` class in pure NumPy —
producing a probability distribution across **12 incident types.**