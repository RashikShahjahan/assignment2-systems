import torch
from einops import einsum
import math

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, is_causal= False)->torch.Tensor:
        B, Nq, d = Q.shape
        _, Nk, _ = K.shape

        Q_TILE_SIZE = 16
        K_TILE_SIZE= 16
        Tq = (Nq + Q_TILE_SIZE - 1) // Q_TILE_SIZE
        Tk = (Nk + K_TILE_SIZE - 1) // K_TILE_SIZE

        O = torch.zeros((B, Nq, d ), device=Q.device, dtype=Q.dtype)
        L = torch.zeros((B, Nq), device=Q.device, dtype=torch.float32)

        for i in range(Tq):
            q_start = i*Q_TILE_SIZE
            q_end = q_start+Q_TILE_SIZE
            Q_i = Q[:, q_start:q_end, :]
            num_rows = Q_i.shape[1]
            m_i = torch.full((B, num_rows), float("-inf"), device=Q.device,dtype=torch.float32)
            l_i = torch.zeros((B, num_rows),device=Q.device, dtype=torch.float32)
            accum = torch.zeros((B, num_rows, d),device=Q.device, dtype=torch.float32)
            for j in range(Tk):
                k_start = j*K_TILE_SIZE
                k_end = k_start+K_TILE_SIZE
                K_j = K[:, k_start:k_end, :]
                V_j = V[:, k_start:k_end, :]
                scores = einsum(
                    Q_i.float(),
                    K_j.float(),
                    "batch query d, batch key d -> batch query key",
                ) / math.sqrt(d)

                m_tile = torch.max(scores, dim=-1).values
                m_new = torch.maximum(m_i, m_tile)
                alpha = torch.exp(m_i - m_new)
                p_tilde = torch.exp(scores-m_new[:,:,None])
                l_new = alpha*l_i + torch.sum(p_tilde, dim=-1)
                O_new = (
                    alpha[:, :, None] * accum
                    + einsum(
                        p_tilde,
                        V_j.float(),
                        "batch query key, batch key d -> batch query d",
                    )
                )

                m_i = m_new
                l_i = l_new
                accum = O_new
            O_i = accum/l_i[:,:,None]
            O[:, q_start:q_end, :] = O_i.to(dtype=O.dtype)
            L[:, q_start:q_end] = m_i + torch.log(l_i)
            ctx.save_for_backward(L,Q,K,V,O)

        return O

