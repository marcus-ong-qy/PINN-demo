# Physics-Informed Neural Network (PINN) — Parabola demo (educational)

This small, hands-on project demonstrates the core idea behind Physics-Informed Neural Networks (PINNs) using a simple, familiar physics problem: a free-falling object described by a parabola. It is written for learners with basic machine learning and calculus knowledge who want a friendly introduction to PINNs and how to combine data and physical laws in a neural-network training loop.

Why this demo?

- The governing equation is simple and analytic: d²y/dt² = −g. That makes it easy to verify results.
- The problem is low-dimensional (1D time → position), so you can focus on PINN concepts (physics residuals, boundary conditions, loss weighting) rather than engineering complexity.
- You can experiment interactively in the provided Jupyter notebook and immediately see how physics constraints change model behaviour.

Project structure

- src/models.py — implements:
  - Regular_NN: a plain feed-forward neural network and a training helper.
  - PINN: the same architecture plus:
    - physics_loss(t): enforces d²y/dt² = −G via automatic differentiation.
    - bc_loss(t_start, t_end): enforces start/end position constraints (y(start)=0, y(end)=0).
    - train_model(...): convenience wrapper that combines data, physics and BC losses.
  - G: gravity constant (9.81 m/s²).
- pinn.ipynb — interactive demo: data generation, noise injection, training three models, and plotting.
- README.md — this file (you’re reading it).

Quick conceptual overview

- Ground truth: y(t) = −0.5 _ G _ t² + v0 \* t. The demo uses v0 = 10 m/s.
- Regular_NN: learns y(t) purely from (noisy) samples. Tends to overfit noise or produce physically inconsistent trajectories.
- PINN: augments the data loss with a physics residual loss r(t) = d²y/dt² + G. The network is trained so r(t) ≈ 0 in addition to fitting data.
- Boundary constraints: add extra loss terms to pin y(t) at the start and end times (e.g., ground-level y = 0). These help the PINN match realistic landing behavior.

How to run (Windows)

1. Create a virtual env and install dependencies:
   pip install numpy pandas matplotlib torch
2. Open `pinn.ipynb` in Jupyter or VS Code and run cells top-to-bottom.

Practical tips and learning experiments

- Loss balancing: physics and BC losses often have different magnitudes than data loss. Try lambda_phys and lambda_bc values across orders of magnitude (0.01 → 100) and observe effects.
- Diagnostics: track data_loss, phys_loss, and bc_loss separately during training to understand which objective dominates.
- Input scaling: normalize time t to [0,1] or to zero mean / unit variance. This often stabilizes training for PINNs.
- Enforce velocities: to enforce initial velocity, add derivative BC (use automatic differentiation to compute dy/dt at t=0).
- Network capacity: experiment with hidden size/depth — too large can overfit noise; too small may not express the parabola well.

Suggested exercises (for learners)

1. Change v0 in the data generator and re-run experiments. Can the PINN recover the new parabola?
2. Remove noisy offsets and see how each model behaves.
3. Replace MSE with Huber loss for data_loss — does robustness to outliers improve?
4. Add velocity BC at t=0 (dy/dt = v0) and observe training speed and accuracy.
5. Modify physics residual to include air drag (simple proportional damping) and adapt the ground-truth to match.

Implementation notes & suggested improvements

- Currently training helpers accept criterion and optimizer inputs — this keeps model classes focused on architecture and losses.
- Add device handling (cpu/gpu) and tensor .to(device) inside train_model for larger experiments.
- Consider passing gravity G as an argument instead of relying on the global constant.
- Add loss-curve plotting and checkpointing for longer experiments.

References and next steps

- PINNs unify data and differential equations in a single loss. For more depth, read the original PINN papers and tutorials that show PDE examples (Poisson, Navier–Stokes).
- Try extending this demo to a 2D projectile (x(t), y(t)) with coupled ODEs and wind/drag terms.

Have fun experimenting — PINNs are a practical bridge between physics knowledge and data-driven models.
