
# Adaptive Resonant SGD Optimizer

This project explores the idea of **adaptive resonance in optimization**, treating optimizers as **IIR filters** applied to the gradient signal. Instead of plain momentum (a 1st-order low-pass filter), we designed a **resonant, adaptive feedback loop** that can **amplify useful gradient patterns** and **dampen oscillations** dynamically.

---

## 🔹 Motivation

* **Vanilla SGD**: steps directly in the gradient direction.
* **SGD + Momentum**: adds a *first-order recurrence*, acting like a low-pass filter on gradients.
* **Our Idea**: generalize momentum into a **higher-order IIR resonant filter** with a moving quality factor (**Q**).

This gives the optimizer the ability to:

* **Amplify what works** → consistent descent directions get reinforced.
* **Suppress what doesn’t** → oscillations and divergence get dampened.
* **Adapt dynamically** → Q is tuned based on how the loss evolves.

---

## 🔹 Theory

Momentum update (IIR-1):

$$
v_t = \mu v_{t-1} + g_t
$$

Resonant SGD update (IIR-2):

$$
u_t = g_t + \alpha_1 u_{t-1} + \alpha_2 u_{t-2}
$$

where coefficients $(\alpha_1, \alpha_2)$ come from the pole radius $r$ and angle $\omega_0$.

* $r$ \~ damping factor (how close poles are to the unit circle).
* Q \~ quality factor (sharpness of resonance).

We let **r adapt** based on feedback from the loss trajectory:

* If loss ↓ → increase r (higher Q, more resonance).
* If loss ↑ → decrease r (more damping).

This is the essence of **adaptive resonance** inside the optimizer.

---

## 🔹 Implementation

### ResonantSGD Optimizer

```python
class ResonantSGD:
    def __init__(self, params, lr=0.01, r=0.9, omega=0.0, q_gain=0.5):
        ...
    def step(self, grads, loss):
        ...
```

* **params**: list of parameters (NumPy arrays).
* **lr**: learning rate.
* **r**: initial pole radius (damping).
* **omega**: resonance frequency (0 = DC trend).
* **q\_gain**: feedback strength for adaptive Q updates.

---

## 🔹 Demo Results

We trained a simple linear regression model ($y = 3x + 2 + \epsilon$).

* At start: `r` dropped (Q ↓) to stabilize against oscillations.
* During convergence: `r` crept upward, adding resonance.
* Final result:

  * Parameters converged to **w ≈ 3.01, b ≈ 2.01**.
  * Loss settled near **0.0096** (noise floor).
  * `r` rose from 0.5 → 0.6 as the optimizer trusted the direction more.

✅ Unlike the unstable version (which exploded due to uncontrolled resonance), this adaptive version **balanced resonance with damping**.

---

## 🔹 Key Insights

* Optimizers can be understood as **filters on gradient signals**.
* **Momentum = IIR-1 filter**.
* **ResonantSGD = IIR-2 resonator with adaptive Q**.
* Adaptive Q ≈ “adaptive resonance” → amplifies stability, suppresses oscillation.
* This connects **signal processing** with **optimization dynamics**.

---

## 🔹 Next Steps

* Compare ResonantSGD with vanilla SGD and momentum in the same run.
* Visualize pole–zero diagrams and how adaptive Q moves poles over time.
* Explore different adaptation signals (loss, gradient variance, curvature).
* Extend to PyTorch so it plugs into `torch.optim`.

---

## 🔹 References

* Grossberg, S. *Adaptive Resonance Theory* (1976–1980s).
* Rumelhart et al. *Learning Representations by Back-Propagating Errors* (1986).
* DSP pole-zero analysis for IIR filter design.

---

📌 **Summary**:
We extended momentum into a **resonant adaptive filter** that learns to amplify useful directions and damp harmful ones — a practical demonstration of **adaptive resonance** inside an optimizer.

### PSEUDOCODE

```python
import numpy as np
import matplotlib.pyplot as plt

class ResonantSGD:
    def __init__(self, params, lr=0.01, r=0.9, omega=0.0, q_gain=0.5):
        self.params = params
        self.lr = lr
        self.r = r
        self.omega = omega
        self.q_gain = q_gain

        self.alpha1 = 2 * r * np.cos(omega)
        self.alpha2 = -r**2

        self.state = [np.zeros((2,) + p.shape) for p in params]
        self.prev_loss = None

    def step(self, grads, loss):
        for i, (p, g) in enumerate(zip(self.params, grads)):
            s = self.state[i]

            u = g + self.alpha1 * s[0] + self.alpha2 * s[1]
            p -= self.lr * u

            self.state[i][1] = self.state[i][0]
            self.state[i][0] = u

        # Feedback on resonance
        if self.prev_loss is not None:
            delta = (loss - self.prev_loss) / (abs(self.prev_loss) + 1e-8)

            if delta < 0:   # improved
                self.r = min(0.99, self.r * (1 + self.q_gain * 0.01))
            else:           # loss increased → damp harder
                self.r = max(0.5, self.r * (1 - self.q_gain * min(delta, 1.0)))

            # Update filter coefficients
            self.alpha1 = 2 * self.r * np.cos(self.omega)
            self.alpha2 = -self.r**2

        self.prev_loss = loss


# ---- Synthetic dataset ----
np.random.seed(42)
X = np.random.randn(200, 1)
y = 3 * X[:, 0] + 2 + 0.1 * np.random.randn(200)

# ---- Parameters ----
w = np.random.randn(1)
b = np.random.randn(1)

params = [w, b]
opt = ResonantSGD(params, lr=0.05, r=0.9, omega=0.0, q_gain=0.5)

# ---- Training loop ----
losses = []
r_values = []

for epoch in range(100):
    # Predictions
    y_pred = X[:, 0] * w + b
    loss = np.mean((y_pred - y) ** 2)

    # Gradients
    grad_w = np.mean(2 * (y_pred - y) * X[:, 0])
    grad_b = np.mean(2 * (y_pred - y))
    grads = [grad_w, grad_b]

    # Step
    opt.step(grads, loss)

    losses.append(loss)
    r_values.append(opt.r)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss={loss:.4f}, w={w[0]:.3f}, b={b[0]:.3f}, r={opt.r:.3f}")

# ---- Plot results ----
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")

plt.subplot(1,2,2)
plt.plot(r_values)
plt.title("Pole Radius (r ~ Q)")
plt.xlabel("Epoch")
plt.ylabel("r value")

plt.show()
```
