# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational Physics-Informed Neural Network (PINN) demonstrating how to combine data-driven learning with physical laws. The demo models free-fall motion (parabola) governed by d²y/dt² = -g.

## Development Commands

```bash
# Setup
pip install -r requirements.txt

# Run the demo
# Open pinn.ipynb in Jupyter or VS Code and run cells top-to-bottom
```

## Architecture

### Core Components

**src/models.py** - Contains two neural network classes:
- `Regular_NN`: Standard feed-forward network (1→20→20→1 with Tanh activations)
- `PINN`: Same architecture plus:
  - `physics_loss(t)`: Enforces d²y/dt² = -G using autograd for second derivatives
  - `bc_loss(t_start, t_end, ...)`: Enforces boundary conditions (position and optionally velocity)
  - `train_model(...)`: Combined loss = data_loss + λ_phys × physics_loss + λ_bc × bc_loss
- `G = 9.81`: Gravity constant (m/s²)

**pinn.ipynb** - Interactive demo comparing three approaches:
1. Regular NN (overfits noisy data)
2. PINN with physics loss only (learns parabola but wrong boundary positions)
3. PINN with physics + boundary losses (correct trajectory)

### Key PINN Concepts

- **Physics residual**: r(t) = d²y/dt² + G, minimized to enforce the ODE
- **Boundary conditions**: Pin y(t) at start/end times via MSE loss terms
- **Loss balancing**: `lambda_phys` and `lambda_bc` weight the physics/BC losses relative to data loss
- **Automatic differentiation**: PyTorch autograd computes dy/dt and d²y/dt² for physics loss

### Training Parameters

- `lambda_phys`: Weight for physics loss (try 0.01 → 100)
- `lambda_bc`: Weight for boundary condition loss
- `enforce_v0`: Optional initial velocity constraint (adds dy/dt term to BC loss)
