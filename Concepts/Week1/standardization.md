## Standardizing the Signal

Before feeding data into a neural network, we need all metrics on the **same scale.**
A CPU reading of 80 and a memory reading of 80 may mean very different things.

---

## What Is It? The Z-Score

The **Z-Score** answers: *"How far is this value from normal, in standard deviations?"*

$$z = \frac{x - \mu}{\sigma}$$

- $z = 0$ → perfectly average
- $z = 3$ → three standard deviations above normal, likely an anomaly
- Works across any metric, any unit, any scale

---

## What I Learned: Preventing Gradient Explosion

Without normalization, large raw values cause the model's weight updates to spiral out of control during training — known as **Gradient Explosion.**

Z-Score normalization centers the data at 0, keeping gradients stable and training reliable.

---

## How We Implemented: Real-Time Z-Score in NumPy

```python
z = (x - np.mean(trace)) / np.std(trace)
```

Calculated on every incoming trace in real-time —
no stored global mean, no batch processing required.