## Probability & Information Theory

How do you measure how *weird* a system state is?
**Information Theory** gives us the math to do exactly that —
turning the vague feeling of "something seems off" into a precise, calculable number.

---

## What Is It? Shannon Entropy

**Shannon Entropy** measures the average uncertainty — or "chaos" — in a stream of data.

$$H(X) = -\sum P(x) \log P(x)$$

- Low entropy → system is predictable, behaving normally
- High entropy → system is chaotic, logs are unpredictable
- A sudden **spike in entropy** is a mathematical signature of an incident

---

## What I Learned: An Incident Is a Spike in Chaos

When a system crashes, its logs stop following familiar patterns.
Error types multiply, sequences break down, and the data becomes highly unpredictable.

That unpredictability *is* the incident — and entropy measures it directly.
We're not looking for specific error messages; we're measuring **disorder itself.**

---

## How We Implemented: Entropy Scoring

For each incoming trace, we calculate the probability distribution of log types.
If the entropy of that distribution exceeds a dynamic threshold:

```
H(trace) > threshold  →  MLP flags a high-priority incident
```

No hardcoded rules. No keyword matching. Pure math.