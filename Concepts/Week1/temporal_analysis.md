## Temporal Analysis & Signal Processing

---

## The System as a Signal

Most people treat server logs as plain text. We treat them as something more powerful:
**Stochastic Process** : a sequence of random variables evolving over time.

---

## What Is It? The Stochastic Process

A **Stochastic Process** models a system as a stream of values that change randomly over time — not randomly in the sense of "meaningless," but in the sense of "probabilistic and measurable."

- Each metric (CPU, memory, error rate) is a random variable $X_t$
- Together they form a sequence: $X_1, X_2, X_3 \ldots X_t$
- We analyze how those values evolve not just what they are

---

## What I Learned: Weight Drift

> "Normal" is a moving target.

A CPU spike at **3 AM** is an anomaly. The same spike at **3 PM** (peak hours) is expected behavior.

Static thresholds like *"alert if CPU > 80%"* are mathematically insufficient for dynamic systems.
The model must continuously redefine its baseline as the system changes.

---

## How We Implemented: Sliding Traces

To give the model memory, we used a **Sliding Trace** approach.

Instead of one value at a time, the model receives a chunk of consecutive values —
letting it see **10 seconds of history at once.**

```python
traces = numpy.lib.stride_tricks.sliding_window_view(stream, trace_size)
```

A 1D telemetry stream → a 2D matrix of time traces → fed into the MLP.