import torch


def linear_beta_schedule(timesteps):
    return torch.linspace(1e-4, 0.02, timesteps)


class DiffusionRefiner:

    def __init__(self, model, timesteps=50, device="cuda"):

        self.model = model
        self.timesteps = timesteps
        self.device = device

        self.betas = linear_beta_schedule(timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    def forward_diffusion(self, x0, t):

        noise = torch.randn_like(x0)
        a_hat = self.alpha_hat[t].view(-1, 1, 1, 1)

        xt = torch.sqrt(a_hat) * x0 + torch.sqrt(1 - a_hat) * noise
        return xt, noise

    def sample(self, x):

        for t in reversed(range(self.timesteps)):

            t_tensor = torch.tensor([t]).to(self.device)

            pred_noise = self.model(x, t_tensor)

            alpha = self.alphas[t]
            a_hat = self.alpha_hat[t]
            beta = self.betas[t]

            z = torch.randn_like(x) if t > 0 else 0

            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - a_hat)) * pred_noise
            ) + torch.sqrt(beta) * z

        return x
