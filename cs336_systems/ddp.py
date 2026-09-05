import torch
import torch.distributed as dist

class DDP(torch.nn.Module):
    def __init__(self,module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.pending_work = []
        for param in module.parameters():
            dist.broadcast(param, 0)
        for buffer in module.buffers():
            dist.broadcast(buffer, 0)

        for param in module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self.hook)



    def forward(self, *args, **kwargs):
        return self.module(*args,**kwargs)

    def hook(self,param: torch.Tensor) -> None:
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
        self.pending_work.append(handle)

    def finish_gradient_synchronization(self):
        for handle in self.pending_work:
            handle.wait()
        self.pending_work.clear()
        