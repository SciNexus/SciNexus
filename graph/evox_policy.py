from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
def interaction_entropy(
    probs: torch.Tensor,
    num_nodes: int,
    num_relations: int,
    *,
    allow_terminate: bool = False,
    sample_terminate: bool = False,
) -> torch.Tensor:
    term_idx = num_nodes * num_nodes * num_relations
    if allow_terminate and sample_terminate:
        probs = probs.clamp(min=1e-12)
        probs = probs / probs.sum().clamp(min=1e-12)
        return torch.distributions.Categorical(probs=probs).entropy()
    interact = probs[:term_idx]
    interact = interact / interact.sum().clamp(min=1e-12)
    return torch.distributions.Categorical(probs=interact).entropy()
def action_log_prob(
    probs: torch.Tensor,
    action_idx: int,
    num_nodes: int,
    num_relations: int,
    *,
    allow_terminate: bool = True,
    sample_terminate: bool = False,
) -> torch.Tensor:
    n, k = num_nodes, num_relations
    term_idx = n * n * k
    probs = probs.clamp(min=1e-12)
    if action_idx == term_idx:
        return torch.log(probs[term_idx])
    if allow_terminate and sample_terminate:
        return torch.log(probs[action_idx])
    interact = probs[:term_idx]
    interact = interact / interact.sum().clamp(min=1e-12)
    return torch.log(interact[action_idx])
def pick_step_action_index(
    probs: torch.Tensor,
    num_nodes: int,
    num_relations: int,
    *,
    force_terminate: bool = False,
    sample_interaction: bool = True,
    allow_terminate: bool = True,
    sample_terminate: bool = False,
) -> tuple[int, torch.Tensor]:
    n, k = num_nodes, num_relations
    term_idx = n * n * k
    probs = probs.clamp(min=1e-12)
    if force_terminate:
        return term_idx, torch.log(probs[term_idx])
    if allow_terminate:
        if sample_terminate:
            if sample_interaction:
                idx = int(torch.multinomial(probs / probs.sum().clamp(min=1e-12), 1).item())
            else:
                idx = int(probs.argmax().item())
            return idx, torch.log(probs[idx])
        if probs.argmax().item() == term_idx:
            return term_idx, torch.log(probs[term_idx])
    interact = probs[:term_idx]
    interact = interact / interact.sum().clamp(min=1e-12)
    if sample_interaction:
        idx = int(torch.multinomial(interact, 1).item())
    else:
        idx = int(interact.argmax().item())
    return idx, torch.log(interact[idx])
_ACTION_MASK_INF = -1e9
def constrain_action_logits(
    logits: torch.Tensor,
    num_nodes: int,
    num_relations: int,
    *,
    last_action_idx: int | None = None,
    forbid_self_loop: bool = True,
    forbid_repeat: bool = True,
) -> torch.Tensor:
    adjusted = logits.clone()
    n, k = num_nodes, num_relations
    term_idx = n * n * k
    if forbid_self_loop:
        for node_idx in range(n):
            start = (node_idx * n + node_idx) * k
            end = start + k
            adjusted[start:end] = _ACTION_MASK_INF
    if (
        forbid_repeat
        and last_action_idx is not None
        and 0 <= last_action_idx < term_idx
    ):
        adjusted[last_action_idx] = _ACTION_MASK_INF
    return adjusted
class RelGNNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_relations: int):
        super().__init__()
        self.num_relations = num_relations
        self.W_rel = nn.ModuleList(
            [nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_relations)]
        )
        self.W_self = nn.Linear(in_dim, out_dim, bias=True)
        for lin in self.W_rel:
            nn.init.xavier_uniform_(lin.weight)
        nn.init.xavier_uniform_(self.W_self.weight)
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        out = self.W_self(x)
        for k in range(self.num_relations):
            a_k = adj[k]
            degree = a_k.sum(dim=1, keepdim=True).clamp(min=1.0)
            out = out + a_k @ self.W_rel[k](x) / degree
        return out
class MemoryFusion(nn.Module):
    NUM_SOURCES = 5                              
    def __init__(self, memory_dim: int, type_dim: int, task_dim: int, content_dim: int):
        super().__init__()
        dims = [memory_dim, memory_dim, type_dim, task_dim, content_dim]
        self.proj = nn.ModuleList([nn.Linear(d, memory_dim, bias=False) for d in dims])
        self.query = nn.Linear(memory_dim, memory_dim, bias=False)
        self.scale = math.sqrt(memory_dim)
    def forward(
        self,
        m_self: torch.Tensor,
        m_other: torch.Tensor,
        type_emb: torch.Tensor,
        task_emb: torch.Tensor,
        content_emb: torch.Tensor,
    ) -> torch.Tensor:
        sources = [m_self, m_other, type_emb, task_emb, content_emb]
        projected = torch.stack([p(x) for p, x in zip(self.proj, sources)], dim=0)
        query = self.query(m_self)
        scores = (projected * query.unsqueeze(0)).sum(dim=-1) / self.scale
        weights = torch.softmax(scores, dim=0)
        return (weights.unsqueeze(-1) * projected).sum(dim=0)
class EvoXGraphPolicy(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        role_embed_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 3,
        task_embed_dim: int = 384,
        content_embed_dim: int | None = None,
        dropout: float = 0.1,
        use_relgnn: bool = True,
        use_gru_memory: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.memory_dim = hidden_dim
        self.task_embed_dim = task_embed_dim
        self.role_embed_dim = role_embed_dim
        self.content_embed_dim = content_embed_dim or task_embed_dim
        self.type_dim = hidden_dim // 4
        self.use_relgnn = use_relgnn
        self.use_gru_memory = use_gru_memory
        self.type_emb = nn.Embedding(num_relations, self.type_dim)
        self.W0 = nn.Linear(role_embed_dim + task_embed_dim, hidden_dim)
        self.memory_fusion = MemoryFusion(
            hidden_dim, self.type_dim, task_embed_dim, self.content_embed_dim
        )
        if self.use_gru_memory:
            self.memory_updater = nn.GRUCell(hidden_dim, hidden_dim)
            self.memory_concat = None
        else:
            self.memory_updater = None
            self.memory_concat = nn.Linear(hidden_dim * 2, hidden_dim)
        self.Wm = nn.Linear(hidden_dim + task_embed_dim, hidden_dim)
        if self.use_relgnn:
            self.rel_gnn = nn.ModuleList(
                [RelGNNLayer(hidden_dim, hidden_dim, num_relations) for _ in range(num_layers)]
            )
            self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
            self.global_mlp = None
        else:
            self.rel_gnn = nn.ModuleList()
            self.layer_norms = nn.ModuleList()
            self.global_mlp = nn.Sequential(
                nn.Linear(hidden_dim + task_embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        self.dropout = nn.Dropout(dropout)
        global_in = hidden_dim + task_embed_dim
        self.Wg = nn.Linear(global_in, hidden_dim, bias=False)
        self.Wa = nn.Linear(hidden_dim * 2 + self.type_dim, hidden_dim, bias=False)
        self.term_head = nn.Sequential(
            nn.Linear(global_in, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(global_in, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._scale = math.sqrt(hidden_dim)
    def init_memory(
        self,
        task_embedding: torch.Tensor,
        role_embeddings: torch.Tensor,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        n = self.num_nodes
        q = task_embedding.unsqueeze(0).expand(n, -1)
        return self.W0(torch.cat([role_embeddings.to(q.device), q], dim=-1))
    def _compute_node_repr(
        self,
        memory: torch.Tensor,
        adj: torch.Tensor,
        task_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = memory.shape[0]
        q = task_embedding.unsqueeze(0).expand(n, -1)
        mq = torch.cat([memory, q], dim=-1)
        if self.use_relgnn:
            h = F.relu(self.Wm(mq))
            for layer, norm in zip(self.rel_gnn, self.layer_norms):
                h_new = norm(F.relu(layer(h, adj)))
                h = h + self.dropout(h_new)
        else:
            h = self.global_mlp(mq)
        g = h.mean(dim=0)
        return h, g
    def update_memory(
        self,
        memory: torch.Tensor,
        src_idx: int,
        dst_idx: int,
        rel_idx: int,
        task_emb: torch.Tensor,
        src_content_emb: torch.Tensor,
        dst_content_emb: torch.Tensor,
    ) -> torch.Tensor:
        dev = memory.device
        type_e = self.type_emb(torch.tensor(rel_idx, device=dev))
        new_mem = memory.clone()
        def _fuse_and_update(
            node_idx: int,
            partner_idx: int,
            content_emb: torch.Tensor,
        ) -> None:
            m_self = memory[node_idx]
            m_other = memory[partner_idx]
            z = self.memory_fusion(m_self, m_other, type_e, task_emb, content_emb)
            if self.use_gru_memory:
                new_mem[node_idx] = self.memory_updater(z, m_self)
            else:
                new_mem[node_idx] = self.memory_concat(torch.cat([z, m_self], dim=-1))
        if src_idx == dst_idx:
            _fuse_and_update(src_idx, src_idx, dst_content_emb)
        else:
            _fuse_and_update(src_idx, dst_idx, src_content_emb)
            _fuse_and_update(dst_idx, src_idx, dst_content_emb)
        return new_mem
    def forward(
        self,
        memory: torch.Tensor,
        adj: torch.Tensor,
        task_embedding: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        dst_mask: Optional[torch.Tensor] = None,
        forbid_self_loop: bool = True,
        forbidden_flat_idx: int | None = None,
        forbid_repeat: bool = True,
    ) -> dict:
        del active_mask                                                   
        h, g = self._compute_node_repr(memory, adj, task_embedding)
        global_feat = torch.cat([g, task_embedding])
        u = self.Wg(global_feat)
        term_logit = self.term_head(global_feat).squeeze(-1)
        value = self.value_head(global_feat).squeeze(-1)
        n, k = self.num_nodes, self.num_relations
        type_embs = self.type_emb.weight
        action_logits = torch.zeros(n, n, k, device=memory.device)
        for i in range(n):
            for j in range(n):
                for rel in range(k):
                    key = self.Wa(torch.cat([h[i], h[j], type_embs[rel]]))
                    action_logits[i, j, rel] = (u * key).sum() / self._scale
        if src_mask is not None:
            action_logits[~src_mask] = _ACTION_MASK_INF
        if dst_mask is not None:
            action_logits[:, ~dst_mask, :] = _ACTION_MASK_INF
        flat_action = action_logits.reshape(-1)
        all_logits = torch.cat([flat_action, term_logit.unsqueeze(0)])
        all_logits = constrain_action_logits(
            all_logits,
            n,
            k,
            last_action_idx=forbidden_flat_idx,
            forbid_self_loop=forbid_self_loop,
            forbid_repeat=forbid_repeat,
        )
        term_idx = n * n * k
        all_probs = F.softmax(all_logits, dim=0)
        return {
            "action_logits": action_logits,
            "flat_action_logits": all_logits[:term_idx],
            "all_logits": all_logits,
            "all_probs": all_probs,
            "term_logit": term_logit,
            "term_prob": all_probs[term_idx],
            "value": value,
            "node_embs": h,
            "graph_memory": g,
        }
    def select_action(
        self,
        memory: torch.Tensor,
        adj: torch.Tensor,
        task_embedding: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        dst_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        force_terminate: bool = False,
        forbidden_flat_idx: int | None = None,
        forbid_self_loop: bool = True,
        forbid_repeat: bool = True,
    ) -> tuple[dict, torch.Tensor, torch.Tensor]:
        out = self.forward(
            memory,
            adj,
            task_embedding,
            active_mask,
            src_mask,
            dst_mask,
            forbid_self_loop=forbid_self_loop,
            forbidden_flat_idx=forbidden_flat_idx,
            forbid_repeat=forbid_repeat,
        )
        n, k = self.num_nodes, self.num_relations
        term_idx = n * n * k
        idx, log_prob = pick_step_action_index(
            out["all_probs"],
            n,
            k,
            force_terminate=force_terminate,
            sample_interaction=not deterministic,
        )
        if idx == term_idx:
            action = {
                "src": -1,
                "dst": -1,
                "type": -1,
                "is_terminate": True,
                "flat_idx": term_idx,
            }
        else:
            rel = idx % k
            rem = idx // k
            dst = rem % n
            src = rem // n
            action = {
                "src": src,
                "dst": dst,
                "type": rel,
                "is_terminate": False,
                "flat_idx": idx,
            }
        return action, log_prob, out["value"]
    def evaluate_actions(
        self,
        memory: torch.Tensor,
        adj: torch.Tensor,
        task_embedding: torch.Tensor,
        action_flat_idx: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        dst_mask: Optional[torch.Tensor] = None,
        forbidden_flat_idx: int | None = None,
        forbid_self_loop: bool = True,
        forbid_repeat: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del active_mask
        out = self.forward(
            memory,
            adj,
            task_embedding,
            src_mask=src_mask,
            dst_mask=dst_mask,
            forbid_self_loop=forbid_self_loop,
            forbidden_flat_idx=forbidden_flat_idx,
            forbid_repeat=forbid_repeat,
        )
        n, k = self.num_nodes, self.num_relations
        lps = []
        for idx in action_flat_idx.tolist():
            lps.append(
                action_log_prob(out["all_probs"], idx, n, k)
            )
        lp = torch.stack(lps)
        entropy = interaction_entropy(out["all_probs"], n, k).expand(action_flat_idx.shape[0])
        values = out["value"].expand(action_flat_idx.shape[0])
        return lp, values, entropy
