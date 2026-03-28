import numpy as np

class EulerDiscreteScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012):
        # Simplified Euler Discrete Scheduler Sigma schedule
        self.num_train_timesteps = num_train_timesteps
        self.betas = np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps)**2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
        self.sigmas = np.array(self.sigmas, dtype=np.float32)

    def set_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        timesteps = np.linspace(0, self.num_train_timesteps - 1, num_inference_steps)[::-1]
        self.timesteps = timesteps.astype(np.float32)
        
        # Interpolate sigmas to target steps
        self.step_sigmas = np.interp(self.timesteps, np.arange(self.num_train_timesteps), self.sigmas)
        self.step_sigmas = np.append(self.step_sigmas, 0.0)
        return self.timesteps

    def get_sigma(self, step_index):
        return self.step_sigmas[step_index]

    def step(self, model_output, step_index, latents):
        sigma = self.step_sigmas[step_index]
        sigma_next = self.step_sigmas[step_index + 1]
        
        dt = sigma_next - sigma
        latents = latents + model_output * dt
        return latents
