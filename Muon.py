import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.cpp_extension import load
from functools import partial
from typing import Dict, List, Optional, Tuple, Callable, Union

torch.set_default_device('cuda')

muon_cpp = load(name='muon_cpp', sources=['muon_cpp.cpp'], verbose=False)

class Muon(optim.Optimizer):
    r"""MomentUm Orthogonalized by Newton-schulz optimizer.
    """
    def __init__(
        self,
        params,
        lr: Union[float, Callable[[torch.Tensor], torch.Tensor]] = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 1e-2,
        nesterov: bool = True,
        backend: str = "newtonschulz5",
        backend_steps: int = 5,
    ):
        if backend not in ["newtonschulz5"]:
             raise ValueError(f"Unknown backend: {backend}. Choose 'newtonschulz5'.")
             
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                       nesterov=nesterov, backend=backend,
                       backend_steps=backend_steps)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', True)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            muon_bufs = []

            lr = group['lr']
            decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is not None:
                    if p.grad.dim() == 0:
                        continue
                        
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    state = self.state[p]
                    if 'muon_buf' not in state:
                        state['muon_buf'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    muon_bufs.append(state['muon_buf'])

            if not params_with_grad:
                continue

            torch._foreach_mul_(muon_bufs, group['momentum'])
            torch._foreach_add_(muon_bufs, grads, alpha=1)

            if group['nesterov']:
                torch._foreach_add_(grads, muon_bufs, alpha=group['momentum'])

            grads = muon_cpp.process_gradients(
                grads, 
                group['backend'], 
                group['backend_steps'],
                1e-8,
            )

            torch._foreach_mul_(grads, -0.2*lr)
            torch._foreach_mul_(params_with_grad, 1 - lr*decay)
            torch._foreach_add_(params_with_grad, grads)

        return loss
