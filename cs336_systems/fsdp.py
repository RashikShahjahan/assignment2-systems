import torch
import torch.distributed as dist
from cs336_basics.model import Linear, Embedding


class FSDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.local_shards = torch.nn.ParameterList()
        self.shard_metadata = []
        seen_weights = {}
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        for module_name, submodule in module.named_modules():
            if isinstance(submodule, (Linear, Embedding)):
                weight = submodule.weight
                weight_id = id(weight)
                if weight_id in seen_weights:
                    existing = seen_weights[weight_id]
                    existing["consumers"].append((module_name, submodule))
                    delattr(submodule, "weight")
                    continue


                flat_weight = weight.detach().flatten().float()
                shard_numel = (
                                weight.numel() + self.world_size - 1
                                ) // self.world_size
                padded_numel = shard_numel * self.world_size
                padding = padded_numel - flat_weight.numel()

                if padding > 0:
                    padded_weight = torch.cat(
                        [flat_weight, flat_weight.new_zeros(padding)]
                    )
                else:
                    padded_weight = flat_weight
                start = self.rank * shard_numel
                end = start + shard_numel

                local_shard = torch.nn.Parameter(
                        padded_weight[start:end].clone(),
                        requires_grad=weight.requires_grad,
                    )


                metadata = {
                    "module_name": module_name,
                    "module": submodule,
                    "original_shape": tuple(weight.shape),
                    "original_numel": weight.numel(),
                    "dtype": weight.dtype,
                    "device": weight.device,
                    "requires_grad": weight.requires_grad,
                    "weight_id": weight_id,
                    "consumers": [(module_name, submodule)],
                    "shard_numel": shard_numel,
                    "padded_numel":padded_numel,
                    "padding":padding,
                    "local_shard_index": len(self.local_shards)
                }

                self.shard_metadata.append(metadata)

                seen_weights[weight_id] = metadata
                self.local_shards.append(local_shard)
                delattr(submodule, "weight")


    def _gather_full_weight(self,metadata):
        local_shard = self.local_shards[metadata["local_shard_index"]]      
        comm_dtype = self.compute_dtype or local_shard.dtype
        comm_shard = local_shard.to(dtype=comm_dtype).contiguous()

        tensor_list = [
            torch.empty_like(comm_shard)
            for _ in range(self.world_size)
        ]

        dist.all_gather(tensor_list, comm_shard)

        full_flat = torch.cat(tensor_list)
        full_flat = full_flat[:metadata["original_numel"]]
        full_weight = full_flat.reshape(metadata["original_shape"])
        full_weight = full_weight.detach()
        full_weight.requires_grad_(metadata["requires_grad"])
        return full_weight


    def forward(self, *inputs, **kwargs):
        for metadata in self.shard_metadata:
            full_weight = self._gather_full_weight(metadata)
            for module_name, submodule in metadata["consumers"]:
                submodule.weight = full_weight
            metadata["full_weight"] = full_weight
        return self.module( *inputs, **kwargs)


    def finish_gradient_synchronization(self):
        for metadata in self.shard_metadata:
             local_shard = self.local_shards[metadata["local_shard_index"]]
             full_grad = metadata["full_weight"].grad
             flat_grad = full_grad.detach().flatten().float()
             padding = metadata["padding"]
             if padding > 0:
                    padded_grad = torch.cat(
                        [flat_grad, flat_grad.new_zeros(padding)]
                    )
             else:
                    padded_grad = flat_grad

             out = torch.empty_like(local_shard)
             dist.reduce_scatter_tensor(out,padded_grad.contiguous(),dist.ReduceOp.SUM)

             out.div_(self.world_size)
             local_shard.grad = out
             for _, submodule in metadata["consumers"]:
                delattr(submodule, "weight")
             del metadata["full_weight"]

        for parameter in self.module.parameters():
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                parameter.grad.div_(self.world_size)
                                
