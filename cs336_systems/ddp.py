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
        grads = []
        params = list(self.module.parameters())
        for param in params:
            if param.grad is not None:
                grads.append(param.grad)
                

        flattened_grads = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flattened_grads, op=dist.ReduceOp.AVG)

        synced_grads = torch._utils._unflatten_dense_tensors(
            flattened_grads,
            grads,
        )

        for param, synced_grad in zip(params, synced_grads):
            param.grad.copy_(synced_grad)
