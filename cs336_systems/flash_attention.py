import torch
from einops import einsum
import math
import triton
import triton.language as tl

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
                query_indices = torch.arange(q_start, q_end, device=Q.device)
                key_indices = torch.arange(k_start, k_end, device=Q.device)
                causal_mask = key_indices[None, :] <= query_indices[:, None]
                if is_causal:
                    scores = scores.masked_fill(
                        ~causal_mask[None, :, :],
                        float("-inf"),
                    )
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
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, grad_out):
        L,Q,K,V,O = ctx.saved_tensors
        is_causal = ctx.is_causal
        D = torch.sum(
        grad_out.float() * O.float(),
        dim=-1,
        )
        dQ = torch.zeros_like(Q, dtype=torch.float32)
        dK = torch.zeros_like(K, dtype=torch.float32)
        dV = torch.zeros_like(V, dtype=torch.float32)
        _, Nk, _ = K.shape
        B, Nq, d = Q.shape

        Q_TILE_SIZE = 16
        K_TILE_SIZE= 16
        Tq = (Nq + Q_TILE_SIZE - 1) // Q_TILE_SIZE
        Tk = (Nk + K_TILE_SIZE - 1) // K_TILE_SIZE
        for i in range(Tq):        
            q_start = i*Q_TILE_SIZE
            q_end = min(q_start+Q_TILE_SIZE, Nq)
            L_i = L[:, q_start:q_end]
            dO_i = grad_out[:, q_start:q_end, :]

            D_i = D[:, q_start:q_end]
            Q_i = Q[:, q_start:q_end, :]
            dQ_i = torch.zeros_like(Q_i)
            for j in range(Tk):
                k_start = j*K_TILE_SIZE
                k_end = min(k_start+K_TILE_SIZE,Nk)
                K_j = K[:, k_start:k_end, :]
                V_j = V[:, k_start:k_end, :]
                scores = einsum(
                    Q_i,
                    K_j,
                    "batch query d, batch key d -> batch query key",
                ) / math.sqrt(d)
                query_indices = torch.arange(q_start, q_end, device=Q.device)
                key_indices = torch.arange(k_start, k_end, device=Q.device)
                causal_mask = key_indices[None, :] <= query_indices[:, None]
                if is_causal:
                    scores = scores.masked_fill(
                        ~causal_mask[None, :, :],
                        float("-inf"),
                    )
                probs = torch.exp(scores-L_i[:,:,None])
                grad_probs = einsum(
                    dO_i,
                    V_j,
                    "batch query d, batch key d -> batch query key",
                )

                grad_scores = probs*(grad_probs-D_i[:,:,None])

                dQ_contribution = einsum(
                grad_scores,
                K_j,
                "batch query key, batch key d -> batch query d",
            ) / math.sqrt(d)


                dK_contribution = einsum(
                grad_scores,
                Q_i,
                "batch query key, batch query d -> batch key d",
            ) / math.sqrt(d)
                
                dV_contribution = einsum(
                probs,
                dO_i,
                "batch query key, batch query d -> batch key d",
            )
                dQ_i+=dQ_contribution
                dK[:, k_start:k_end, :]+=dK_contribution
                dV[:, k_start:k_end, :]+=dV_contribution
            dQ[:, q_start:q_end, :] = dQ_i
        return dQ, dK, dV,None


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(D,N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(0, 1),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    m = tl.full((Q_TILE_SIZE,), -float("inf"), tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), tl.float32)
    accum = tl.zeros((Q_TILE_SIZE, D), tl.float32)

    q = tl.load(
    Q_block_ptr,
    boundary_check=(0, 1),
    padding_option="zero",
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    for k_start in tl.range(0, N_KEYS, K_TILE_SIZE):
        k = tl.load(
        K_block_ptr,
        boundary_check=(1,),
        padding_option="zero",
        )
        v  = tl.load(
        V_block_ptr,
        boundary_check=(0,),
        padding_option="zero",
        )



        key_indices = k_start + tl.arange(0, K_TILE_SIZE)
        query_indices = (
            query_tile_index * Q_TILE_SIZE
            + tl.arange(0, Q_TILE_SIZE)
        )

        if IS_CAUSAL:
            valid_scores = (
                (key_indices[None, :] <= query_indices[:, None])
                & (key_indices[None, :] < N_KEYS)
            )
        else:
            valid_scores = key_indices[None, :] < N_KEYS

        S = tl.dot(q, k) * scale
        S = tl.where(valid_scores, S, -float("inf"))
        m_tile = tl.max(S,axis=1)
        m_new = tl.maximum(m,m_tile)
        p_tilde = tl.exp(S-m_new[:,None])
        alpha = tl.exp(m-m_new)
        l_new = alpha*l + tl.sum(p_tilde, axis = 1)
        accum_new = alpha[:,None]*accum+tl.dot(p_tilde,v)

        m = m_new
        l = l_new
        accum = accum_new
        K_block_ptr = tl.advance(
            K_block_ptr,
            (0, K_TILE_SIZE),
        )
        V_block_ptr = tl.advance(
            V_block_ptr,
            (K_TILE_SIZE, 0),
        )


    O_tile = accum/l[:,None]
    L_tile = m+tl.log(l)
    tl.store(O_block_ptr, O_tile,boundary_check=(0, 1),
)

    tl.store(L_block_ptr, L_tile,boundary_check=(0,),)

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, Nq, D = Q.shape
        _, Nk, _ = K.shape

        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16

        O = torch.empty_like(Q)
        L = torch.empty(
            (B, Nq),
            device=Q.device,
            dtype=torch.float32,
        )

        grid = (
            triton.cdiv(Nq, Q_TILE_SIZE),
            B,
        )

        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,

            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),

            Nq, Nk,
            1.0 / math.sqrt(D),

            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            IS_CAUSAL = is_causal
        )

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O
    @staticmethod
    def backward(ctx, grad_out):
        L,Q,K,V,O = ctx.saved_tensors
        is_causal = ctx.is_causal
        D = torch.sum(
        grad_out.float() * O.float(),
        dim=-1,
        )
        dQ = torch.zeros_like(Q, dtype=torch.float32)
        dK = torch.zeros_like(K, dtype=torch.float32)
        dV = torch.zeros_like(V, dtype=torch.float32)
        _, Nk, _ = K.shape
        B, Nq, d = Q.shape

        Q_TILE_SIZE = 16
        K_TILE_SIZE= 16
        Tq = (Nq + Q_TILE_SIZE - 1) // Q_TILE_SIZE
        Tk = (Nk + K_TILE_SIZE - 1) // K_TILE_SIZE
        for i in range(Tq):        
            q_start = i*Q_TILE_SIZE
            q_end = min(q_start+Q_TILE_SIZE, Nq)
            L_i = L[:, q_start:q_end]
            dO_i = grad_out[:, q_start:q_end, :]

            D_i = D[:, q_start:q_end]
            Q_i = Q[:, q_start:q_end, :]
            dQ_i = torch.zeros_like(Q_i)
            for j in range(Tk):
                k_start = j*K_TILE_SIZE
                k_end = min(k_start+K_TILE_SIZE,Nk)
                K_j = K[:, k_start:k_end, :]
                V_j = V[:, k_start:k_end, :]
                scores = einsum(
                    Q_i,
                    K_j,
                    "batch query d, batch key d -> batch query key",
                ) / math.sqrt(d)
                query_indices = torch.arange(q_start, q_end, device=Q.device)
                key_indices = torch.arange(k_start, k_end, device=Q.device)
                causal_mask = key_indices[None, :] <= query_indices[:, None]
                if is_causal:
                    scores = scores.masked_fill(
                        ~causal_mask[None, :, :],
                        float("-inf"),
                    )
                probs = torch.exp(scores-L_i[:,:,None])
                grad_probs = einsum(
                    dO_i,
                    V_j,
                    "batch query d, batch key d -> batch query key",
                )

                grad_scores = probs*(grad_probs-D_i[:,:,None])

                dQ_contribution = einsum(
                grad_scores,
                K_j,
                "batch query key, batch key d -> batch query d",
            ) / math.sqrt(d)


                dK_contribution = einsum(
                grad_scores,
                Q_i,
                "batch query key, batch query d -> batch key d",
            ) / math.sqrt(d)
                
                dV_contribution = einsum(
                probs,
                dO_i,
                "batch query key, batch query d -> batch key d",
            )
                dQ_i+=dQ_contribution
                dK[:, k_start:k_end, :]+=dK_contribution
                dV[:, k_start:k_end, :]+=dV_contribution
            dQ[:, q_start:q_end, :] = dQ_i
        return dQ, dK, dV,None

