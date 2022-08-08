import torch
from torchtyping import TensorType

class PGA:
    def __init__(self, model, dataset, h: float, lambd: float, 
                 n_particles: int, device="cpu"):
        
        self.model = model
        self.dataset = dataset
        self.n_particles = n_particles
        self.h = h
        self.device = device
        
        # Initialize samples:
        self._particles = model.init_x([len(dataset), n_particles])

        # Declare theta optimizer for theta:
        self.theta_opt = torch.optim.RMSprop(model.parameters(),
                                             lr=h*len(dataset)*lambd,
                                             alpha=0.1)

    def loss(self,
             images,#: TensorType["n_batch", "image_dimensions": ...],
             particles: TensorType["n_batch", "n_particles","x_dim"]
             ) -> TensorType[()]:
        """
        Returns 
        \frac{1}{N|images|}\sum_{n=1}^N\sum_{m in images}
                                            log(p_{\theta_k}(X_k^{n,m}, y^m)).
        """
        log_p = self.model.log_p_v(images, particles)
        return - (1. / images.shape[0]) * log_p.mean()

    def step(self,
             img_batch,#: TensorType["n_batch", "image_dimensions":...],
             idx: TensorType["n_batch"]):

        ## Compute theta gradients ##
        
        # Turn on theta gradients: 
        self.model.train()  
        self.model = self.model.requires_grad_(True)  
        
        # Evaluate loss function:
        loss = self.loss(img_batch, self._particles[idx].to(img_batch.device))
        
        # Backpropagate theta gradients:
        self.theta_opt.zero_grad()  
        loss.backward()  

        ## Update particles ##
        
        # Turn off theta gradients:
        self.model.eval() 
        self.model = self.model.requires_grad_(False)
        
        # To avoid exceeding the device's memory, we update the particles
        # using minibatches:
        batches = torch.utils.data.DataLoader(self.dataset, batch_size=750, 
                                              pin_memory=True)
        for imgs, idx in batches:
            # Select particle components to be updated in this iteration:
            sub_particles = (self._particles[idx].detach().clone()
                                 .to(img_batch.device).requires_grad_(True))
            
            # Send relevant images to device:
            imgs = imgs.to(img_batch.device)

            # Compute x gradients:
            log_p_v = self.model.log_p_v(imgs, sub_particles).sum()
            x_grad = torch.autograd.grad(log_p_v, sub_particles)[0]

            # Take a gradient step for this batch's particles:
            self._particles[idx] += self.h * x_grad.to(self._particles.device)

        # Add noise to all particles:
        self._particles += ((2*self.h)**0.5 *torch.randn_like(self._particles))

        # Update theta:
        self.theta_opt.step()

        # Return value of loss function:
        return loss.item()
