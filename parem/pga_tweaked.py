import torch
from torchtyping import TensorType
from torch.nn.utils import clip_grad_norm_

optimisers = {'sgd': torch.optim.SGD,
              'adagrad': torch.optim.Adagrad,
              'rmsprop': torch.optim.RMSprop,
              }


class PGA_tweaked:
    def __init__(self,
                 model,
                 n_images: int,
                 dl,
                 q_step_size: float = 1e-2,
                 theta_step_size: float = 1e-2,
                 n_particles: int = 30,
                 device="cpu", # Specifies where self._particle resides.
                 clip_grad=False,
                 theta_opt='sgd',
                 image_batches=None):

        self.n_particles = n_particles
        self.model = model
        self.q_step_size = q_step_size
        self.device = device
        self.clip_grad = clip_grad
        self.dl = dl
        self.n_images = n_images

        # Initialize samples
        self._particles, self.init_losses = model.init_x([n_images, n_particles],
                                       device=device,
                                       image_batches=image_batches)

        # Declare theta optimiser
        self.theta_opt = optimisers[theta_opt](model.parameters(),
                                               lr=theta_step_size)

    def loss(self,
             images,#: TensorType["n_batch", "image_dimensions": ...],
             particles: TensorType["n_batch", "n_particles","x_dim"]
             ) -> TensorType[()]:
        """
        \frac{M}{N|images|}\sum_{n=1}^N\sum_{m in images}p_{\theta_k}(X_k^{n,m}, y^m)
        """
        log_p = self.model.log_p_v(images, particles)
        assert not log_p.isnan().any(), "log_p is nan."
        scale = (self.n_images
                 / images.shape[0])
        return - scale * log_p.mean()

    def step(self,
             img_batch,#: TensorType["n_batch", "image_dimensions":...],
             idx: TensorType["n_batch"]):

        # Compute theta gradients:
        self.model.train()  # ??
        self.theta_opt.zero_grad()  # Zero theta gradients
        self.model = self.model.requires_grad_(True)  # ??

        ## Turn to leaf variable.
        sub_particles = self._particles[idx].to(self.device).requires_grad_(True)

        # Evaluate loss function:
        loss = self.loss(img_batch, sub_particles)

        # Backpropagate theta gradients:
        loss.backward()

        # Clip theta gradients if clipping requested:
        if self.clip_grad:
            clip_grad_norm_(self.model.parameters(), 100)

        # Fudge q_step_size
        scale = sub_particles.shape[1] * img_batch.shape[0] / self.n_images
        # Take a gradient step for this batch's particles:
        self._particles[idx] -= (self.q_step_size * scale
                                 * sub_particles._grad.to(self._particles.device))

        # Add noise to sub_articles:
        self._particles[idx] += ((2 * self.q_step_size) ** 0.5
                                 * torch.randn_like(self._particles[idx]).to(self._particles.device))

        # Update theta:
        self.theta_opt.step()

        # Return value of loss function:
        return loss.item()
