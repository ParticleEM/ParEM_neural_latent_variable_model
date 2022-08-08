import torch
from torchtyping import TensorType
from torch.nn.utils import clip_grad_norm_

optimisers = {'sgd': torch.optim.SGD,
              'adagrad': torch.optim.Adagrad,
              'rmsprop': torch.optim.RMSprop,
              }


class PGA:
    def __init__(self,
                 model,
                 n_images: int,
                 dl,
                 q_step_size: float = 1e-2,
                 theta_step_size: float = 1e-2,
                 n_particles: int = 30,
                 device="cpu",
                 clip_grad=False,
                 theta_opt='sgd',):

        self.n_particles = n_particles
        self.model = model
        self.q_step_size = q_step_size
        self.device = device
        self.clip_grad = clip_grad
        self.dl = dl
        self.n_images = n_images

        # Initialize samples
        self._particles = model.init_x([n_images, n_particles],
                                       device=self.device)

        # Declare theta optimiser
        if type(theta_opt) == str:
            self.theta_opt = optimisers[theta_opt](model.parameters(),
                                                   lr=theta_step_size)
        elif isinstance(theta_opt, torch.optim.Optimizer):
            self.theta_opt = theta_opt

    def loss(self,
             images,#: TensorType["n_batch", "image_dimensions": ...],
             particles: TensorType["n_batch", "n_particles","x_dim"]
             ) -> TensorType[()]:
        """
        \frac{M}{N|images|}\sum_{n=1}^N\sum_{m in images}p_{\theta_k}(X_k^{n,m}, y^m)
        """
        log_p = self.model.log_p_v(images, particles)
        assert not log_p.isnan().any(), "log_p is nan."
        return - (self.n_images / images.shape[0]) * log_p.mean()

    def step(self,
             img_batch,#: TensorType["n_batch", "image_dimensions":...],
             idx: TensorType["n_batch"]):

        # Compute theta gradients:
        self.model.train()  # ??
        self.theta_opt.zero_grad()  # Zero theta gradients
        self.model = self.model.requires_grad_(True)  # ??

        # Evaluate loss function:
        loss = self.loss(img_batch, self._particles[idx].to(img_batch.device))

        # Backpropagate theta gradients:
        loss.backward()

        # Clip theta gradients if clipping requested:
        if self.clip_grad:
            clip_grad_norm_(self.model.parameters(), 100)

        # Update particles batch by batch (s.t. device memory is not exceeded):
        self.model.eval()
        self.model = self.model.requires_grad_(False)
        for imgs, idx in self.dl:
            # Select particles to be updated in this iteration:
            sub_particles = (self._particles[idx].detach().clone()
                                 .to(self.device).requires_grad_(True))
            # Send relevant images to device:
            imgs = imgs.to(self.device)

            # Compute x gradients:
            log_p_v = self.model.log_p_v(imgs, sub_particles).sum()
            x_grad = torch.autograd.grad(log_p_v, sub_particles)[0]

            # Take a gradient step for this batch's particles:
            self._particles[idx] += (self.q_step_size
                                     * x_grad.to(self._particles.device))

        # Add noise to all particles:
        self._particles += ((2 * self.q_step_size) ** 0.5
                            * torch.randn_like(self._particles))

        # Update theta:
        self.theta_opt.step()

        # Return value of loss function:
        return loss.item()

    def sample_posterior(self, n_samples):

        n = torch.randint(low=0, high=self.n_particles, size=(n_samples))
        m = torch.randint(low=0, high=self.n_images, size=(n_samples))
        return self._particles[m, n]
    