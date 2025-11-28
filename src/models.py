import torch
import torch.nn as nn

# define gravity
G = 9.81  # m/s^2

class Regular_NN(nn.Module):
    def __init__(self):
        super(Regular_NN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )
    
    def forward(self, t):
        return self.net(t)
    
    def train_model(self, t: torch.Tensor, y: torch.Tensor, 
                    criterion, optimizer,
                    num_epochs=1000,
                    device: str = None):

        if optimizer is None or criterion is None:
            raise ValueError("Please provide optimizer and criterion to train_model")

        # infer device from model if not provided
        device = torch.device(device) if device else next(self.parameters()).device
        self.to(device)
        t = t.to(device)
        y = y.to(device)

        history = {'loss': []}
        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()
            y_pred = self(t)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            
            history['loss'].append(loss.item())
            if (epoch+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        return history

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )
    
    def forward(self, t):
        return self.net(t)
    
    def physics_loss(self, t):
        # Create a version of t that requires gradients for autograd
        # Note: t should already be on the correct device from train_model()
        t_ = t.detach().requires_grad_(True)
        y = self.forward(t_)
        dy_dt = torch.autograd.grad(y, t_, grad_outputs=torch.ones_like(y), create_graph=True)[0]
        d2y_dt2 = torch.autograd.grad(dy_dt, t_, grad_outputs=torch.ones_like(dy_dt), create_graph=True)[0]

        # Physics: d2y/dt2 + g = 0  => d2y/dt2 = -g
        physics_residual = d2y_dt2 + G
        return torch.mean(physics_residual**2)

    def bc_loss(self, t_start, t_end, y_start=0.0, y_end=0.0, v_start=None):
        """
        Enforce y(t_start)=y_start and y(t_end)=y_end.
        Optionally enforce initial velocity v_start (dy/dt at t_start).
        """
        device = next(self.parameters()).device
        t0 = torch.tensor([[float(t_start)]], dtype=torch.float32, device=device).requires_grad_(True)
        tT = torch.tensor([[float(t_end)]], dtype=torch.float32, device=device)
        y0 = self.forward(t0)
        yT = self.forward(tT)
        loss_pos = torch.mean((y0 - float(y_start))**2) + torch.mean((yT - float(y_end))**2)

        loss_vel = torch.tensor(0.0, device=device)
        if v_start is not None:
            dy_dt = torch.autograd.grad(y0, t0, grad_outputs=torch.ones_like(y0), create_graph=True)[0]
            loss_vel = torch.mean((dy_dt - float(v_start))**2)

        return loss_pos + loss_vel

    def train_model(self, t: torch.Tensor, y: torch.Tensor, 
                    criterion, optimizer,
                    num_epochs=1000,
                    lambda_phys=1, 
                    lambda_bc=0,
                    device: str = None,
                    enforce_v0: float = None):

        if optimizer is None or criterion is None:
            raise ValueError("Please provide optimizer and criterion to train_model")

        # move model and data to device (infer from model if not given)
        device = torch.device(device) if device else next(self.parameters()).device
        self.to(device)
        t = t.to(device)
        y = y.to(device)

        t_start = float(t.min().cpu().item())
        t_end = float(t.max().cpu().item())

        history = {'loss': []}
        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()
            y_pred = self(t)
            data_loss = criterion(y_pred, y)

            loss = data_loss
            # physics informed losses
            if lambda_phys > 0:
                phys_loss = self.physics_loss(t)
                loss += lambda_phys * phys_loss

            if lambda_bc > 0:
                bc_loss = self.bc_loss(t_start, t_end, y_start=0.0, y_end=0.0, v_start=enforce_v0)
                loss += lambda_bc * bc_loss

            loss.backward()
            optimizer.step()
            
            history['loss'].append(loss.item())
            if (epoch+1) % 100 == 0:
                log_msg = f'Epoch [{epoch+1}/{num_epochs}], Total: {loss.item():.6f}, data: {data_loss.item():.6f}'
                if lambda_phys > 0:
                    log_msg += f', phys: {phys_loss.item():.6f}'
                if lambda_bc > 0:
                    log_msg += f', bc: {bc_loss.item():.6f}'
                print(log_msg)
        return history
