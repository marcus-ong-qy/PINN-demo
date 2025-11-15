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
                    num_epochs=1000):

        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()
            y_pred = self(t)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

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
        # avoid modifying the caller's tensor in-place
        t_ = t.clone().detach().requires_grad_(True)
        y = self.forward(t_)
        dy_dt = torch.autograd.grad(y, t_, grad_outputs=torch.ones_like(y), create_graph=True)[0]
        d2y_dt2 = torch.autograd.grad(dy_dt, t_, grad_outputs=torch.ones_like(dy_dt), create_graph=True)[0]

        # Physics: d2y/dt2 + g = 0  => d2y/dt2 = -g
        physics_residual = d2y_dt2 + G
        return torch.mean(physics_residual**2)

    def bc_loss(self, t_start, t_end, y_start=0.0, y_end=0.0):
        # enforce y(t_start)=y_start and y(t_end)=y_end
        t0 = torch.tensor([[float(t_start)]], dtype=torch.float32)
        tT = torch.tensor([[float(t_end)]], dtype=torch.float32)
        y0 = self.forward(t0)
        yT = self.forward(tT)
        return torch.mean((y0 - float(y_start))**2) + torch.mean((yT - float(y_end))**2)
    
    def train_model(self, t: torch.Tensor, y: torch.Tensor, 
                    criterion, optimizer,
                    num_epochs=1000,
                    lambda_phys=1, 
                    lambda_bc=0):

        t_start = float(t.min())   # expects `t` from earlier (numpy array)
        t_end = float(t.max())

        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()
            y_pred = self(t)
            data_loss = criterion(y_pred, y)

            # physics informed losses
            phys_loss = self.physics_loss(t)
            bc_loss = self.bc_loss(t_start, t_end, y_start=0.0, y_end=0.0)

            loss = data_loss + lambda_phys * phys_loss + lambda_bc * bc_loss
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

