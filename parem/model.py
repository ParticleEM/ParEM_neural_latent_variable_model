import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

class Deterministic(nn.Module):
    def __init__(self, in_dim, out_dim, activation=F.gelu):
        super(Deterministic, self).__init__()
        
        self.activation = activation

        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=5, stride=1,
                              padding=2)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1,
                               padding=1)

        self.bn = nn.BatchNorm2d(out_dim)
        self.bn2 = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = out + x  # Skip connection
        return out

class Projection(nn.Module):
    def __init__(self, in_dim, ngf=16, coef=4, activation=F.gelu):
        super(Projection, self).__init__()

        self.activation = activation
        self.ngf = 16
        self.coef = 4

        self.linear = nn.Linear(in_dim, coef * ngf * ngf)
        self.deconv1 = nn.ConvTranspose2d(coef, ngf * coef, kernel_size=5,
                                          stride=1, padding=2, bias=False)
        self.linear_bn = nn.BatchNorm1d(coef * ngf * ngf)
        self.deconv1_bn = nn.BatchNorm2d(ngf * coef)

    def forward(self, x):
        out = self.linear(x.view(x.size(0), -1))
        out = self.linear_bn(out)
        out = self.activation(out)
        out = out.view(out.size(0), self.coef, self.ngf, self.ngf).contiguous()
        out = self.deconv1(out)
        out = self.deconv1_bn(out)
        out = self.activation(out)
        return out

class Output(nn.Module):
    def __init__(self, x_in, nc, activation=torch.tanh):
        super(Output, self).__init__()
        self.output_layer = nn.ConvTranspose2d(x_in, nc, kernel_size=4,
                                               stride=2, padding=1)

    def forward(self, x):
        out = self.output_layer(x)
        out = torch.tanh(out)
        return out

def AnyBatchSize3D(f):
    def wrapper(self, x):
        not_batch_shape = x.shape[-3:]
        batch_shape = x.shape[:-3]
        # Flatten
        x = x.view(-1, *not_batch_shape)
        out = f(self, x)
        return out.view(*batch_shape, *out.shape[-3:])
    return wrapper

class G(nn.Module):
    def __init__(self, x_dim, nc=3, ngf=16, coef=4, sigma2=1.):

        super(G, self).__init__()
        self.sigma2 = sigma2
        self.x_dim = x_dim
        self.ngf = ngf

        self.projection_layer = Projection(x_dim,  ngf=ngf, coef=coef)
        self.deterministic_layer_1 = Deterministic(ngf * coef, ngf * coef)
        self.deterministic_layer_2 = Deterministic(ngf * coef, ngf * coef)
        self.output_layer = Output(ngf * coef, nc)

    @AnyBatchSize3D
    def forward(self, x):
        out = self.projection_layer(x)
        out = self.deterministic_layer_1(out)
        out = self.deterministic_layer_2(out)
        out = self.output_layer(out)
        return out

    def init_x(self, shape):
        'Initializes particles by sampling the prior.'
        return torch.randn(*shape, self.x_dim, 1, 1, device='cpu')

    def log_p(self,
               image: TensorType["n_batch", "n_channels", "height", "width"],
               x: TensorType["n_batch", "x_dim", 1, 1]) -> TensorType[()]:
        # Log prior
        log_prior = - 0.5 * (x ** 2).sum([])

        # Log likelihood
        x_decoded = self.forward(x)
        log_likelihood = - 0.5 * ((image - x_decoded) ** 2 / self.sigma2).sum()
        return log_prior + log_likelihood

    def log_p_v(self,
                image: TensorType["n_batch", "n_channels", "height", "width"],
                x: TensorType["n_batch", "n_particles", "x_dim", 1, 1]) -> TensorType["n_particles"]:
        # Log prior
        log_prior = - 0.5 * (x ** 2).sum([-1, -2, -3, 0])

        # Log likelihood
        x_decoded = self.forward(x)
        log_likelihood = - 0.5 * ((image.unsqueeze(1) - x_decoded) ** 2 / self.sigma2).sum([-1, -2, -3, 0])
        return log_prior + log_likelihood