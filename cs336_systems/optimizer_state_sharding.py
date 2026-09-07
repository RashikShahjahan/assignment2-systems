import torch
from typing import Any
import torch.distributed as dist

class OptimizerStateSharding(torch.optim.Optimizer):

    def __init__(self, params, optimizer_cls: type[torch.optim.Optimizer], **kwargs: Any):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = dict(kwargs)
        self.ownerships = []
        self.local_groups = []
        self.param_index = 0
        self.optimizer = None

        super().__init__(params, defaults=dict(kwargs))
        self.optimizer=optimizer_cls(self.local_groups, **kwargs)




    def step(self, closure=None, **kwargs):
        res = self.optimizer.step(closure,**kwargs)
        for param, owner in self.ownerships:
            dist.broadcast(param, owner)

        return res


    def add_param_group(self, param_group: dict[str, Any]):
        super().add_param_group(param_group)
        params = list(param_group["params"])
        local_params = []
        for param in params:
            owner = self.param_index%self.world_size
            self.ownerships.append((param, owner))
            if owner == self.rank:
                local_params.append(param)

            self.param_index+=1
            
        param_group_copy = param_group.copy()
        param_group_copy["params"] = local_params

        if self.optimizer is None:
            self.local_groups.append(param_group_copy)
        else:
            self.optimizer.add_param_group(param_group_copy)






