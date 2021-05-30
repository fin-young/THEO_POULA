import torch
import math
from torch.optim.optimizer import Optimizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class THEOPOULA(Optimizer):

    def __init__(self, params, eta, lr=1e-1, beta=1e10, r=3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, beta=beta, eta=eta, r=r, betas=betas, eps=eps)
        super(THEOPOULA, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(THEOPOULA, self).__setstate__(state)


    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()


        for group in self.param_groups:
            pnorm = 0
            eta = group['eta']
            if eta > 0:
                for p in group['params']:
                    pnorm += torch.sum(torch.pow(p.data, exponent=2))
            r = group['r']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0

                eta, beta, lr = group['eta'], group['beta'], group['lr']

                noise = math.sqrt(2 * lr / beta) * torch.randn(size=p.size(), device=device)
                numer = grad * ( 1 + math.sqrt(lr)/ (group['eps'] + torch.abs(grad)))
                denom = 1 + math.sqrt(lr) * torch.abs(grad)

                p.data.addcdiv_(value=-lr, tensor1=numer, tensor2=denom).add_(noise)

        return loss

