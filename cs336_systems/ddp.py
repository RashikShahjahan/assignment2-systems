import torch
import torch.distributed as dist

class DDP(torch.nn.Module):
    def __init__(self,module: torch.nn.Module):
        super().__init__()
        self.module = module
        for param in module.parameters():
            dist.broadcast(param, 0)
        for buffer in module.buffers():
            dist.broadcast(buffer, 0)

    def forward(self, *args, **kwargs):
        return self.module(*args,**kwargs)

    def finish_gradient_synchronization(self):
        for param in self.module.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)


