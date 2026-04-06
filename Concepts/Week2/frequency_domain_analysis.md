## Frequency Domain Analysis

Some patterns are completely invisible when you look at raw values over time.
They only appear when you ask a different question:

> *"What frequencies are hidden inside this signal?"*

---

## What Is It? Fourier Transforms

A **Fourier Transform** decomposes a signal into its constituent frequencies —
like splitting white light through a prism to reveal the colors inside.

Applied to system telemetry, it can expose:
- **Periodic background tasks** (a heartbeat of regular activity)
- **Cyclic load patterns** (daily or hourly rhythms)
- **Hidden anomalies** buried under surface-level noise

---

## What I Learned: Separating Noise from Signal

Not all unusual activity is an incident. Systems produce constant low-level noise —
scheduled jobs, health checks, background scans.

Frequency analysis taught me to distinguish:

- **System Noise** — high-frequency, low-impact background chatter
- **Signal** — low-frequency, high-impact state changes that indicate real incidents

---

## How We Implemented: NumPy FFT

```python
frequencies = np.fft.fft(trace)
```

By analyzing log-frequency spikes in the frequency domain,
the model learned to **ignore high-frequency chatter**
and focus only on the low-frequency shifts that signal a real incident.

---

## Summary: Temporal Feature Pipeline

| Stage | What Happens |
|---|---|
| **Raw Data** | Unprocessed telemetry stream |
| **Tracing** | Sliding trace via `stride_tricks` |
| **Scaling** | Z-score normalization |
| **FFT** | Frequency domain transformation |
| **Feature Vector** | MLP-ready input |

**Pipeline:** Raw Data → Tracing → Scaling → FFT → Feature Vector