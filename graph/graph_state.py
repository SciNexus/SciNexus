from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
@dataclass
class NodeState:
    role: str
    role_idx: int
    active: bool = False
    current_hypothesis: str = ""
    hypothesis_history: list[str] = field(default_factory=list)
    hypothesis_embedding: np.ndarray | None = None
@dataclass
class EdgeRecord:
    step: int
    src_role: str
    dst_role: str
    interaction_type: str
    src_message: str
    dst_response: str
class GraphState:
    def __init__(
        self,
        agent_roles: list[str],
        interaction_types: list[str],
        text_embed_dim: int = 384,
        role_embed_dim: int = 32,
    ):
        self.agent_roles = agent_roles
        self.interaction_types = interaction_types
        self.text_embed_dim = text_embed_dim
        self.role_embed_dim = role_embed_dim
        self.num_nodes = len(agent_roles)
        self.num_relations = len(interaction_types)
        self.role_to_idx = {r: i for i, r in enumerate(agent_roles)}
        self.type_to_idx = {t: i for i, t in enumerate(interaction_types)}
        torch.manual_seed(42)
        self._role_emb = nn.Embedding(self.num_nodes, role_embed_dim)
        nn.init.orthogonal_(self._role_emb.weight)
        self._role_emb.weight.requires_grad_(False)
        self.nodes: list[NodeState] = []
        self.adj = torch.zeros(0)
        self.edge_history: list[EdgeRecord] = []
        self.current_step = 0
        self.reset()
    def reset(self) -> None:
        self.nodes = [NodeState(role=r, role_idx=i) for i, r in enumerate(self.agent_roles)]
        self.adj = torch.zeros(self.num_relations, self.num_nodes, self.num_nodes)
        self.edge_history = []
        self.current_step = 0
    def activate_node(self, role: str, hypothesis: str, hypothesis_emb: np.ndarray) -> None:
        node = self.nodes[self.role_to_idx[role]]
        node.active = True
        node.current_hypothesis = hypothesis
        node.hypothesis_history.append(hypothesis)
        node.hypothesis_embedding = hypothesis_emb.copy()
    def add_edge(
        self,
        src_role: str,
        dst_role: str,
        interaction_type: str,
        src_message: str,
        dst_response: str,
        dst_hypothesis_emb: np.ndarray,
    ) -> None:
        src_idx = self.role_to_idx[src_role]
        dst_idx = self.role_to_idx[dst_role]
        type_idx = self.type_to_idx[interaction_type]
        self.adj[type_idx, dst_idx, src_idx] = 1.0
        src = self.nodes[src_idx]
        if not src.active:
            src.active = True
        dst = self.nodes[dst_idx]
        dst.active = True
        dst.current_hypothesis = dst_response
        dst.hypothesis_history.append(dst_response)
        dst.hypothesis_embedding = dst_hypothesis_emb.copy()
        self.edge_history.append(
            EdgeRecord(
                step=self.current_step,
                src_role=src_role,
                dst_role=dst_role,
                interaction_type=interaction_type,
                src_message=src_message,
                dst_response=dst_response,
            )
        )
        self.current_step += 1
    @property
    def node_feature_dim(self) -> int:
        return self.role_embed_dim + self.text_embed_dim + 1
    def get_node_features(self) -> torch.Tensor:
        feats = []
        for i, node in enumerate(self.nodes):
            r_emb = self._role_emb(torch.tensor(i)).detach()
            if node.hypothesis_embedding is not None:
                h_emb = torch.tensor(node.hypothesis_embedding, dtype=torch.float32)
            else:
                h_emb = torch.zeros(self.text_embed_dim)
            active_flag = torch.tensor([1.0 if node.active else 0.0])
            feats.append(torch.cat([r_emb, h_emb, active_flag]))
        return torch.stack(feats)
    def get_active_mask(self) -> torch.Tensor:
        return torch.tensor([n.active for n in self.nodes], dtype=torch.bool)
    def get_role_embedding(self, idx: int) -> torch.Tensor:
        return self._role_emb(torch.tensor(idx)).detach()
    def get_best_hypothesis(self) -> str:
        for edge in reversed(self.edge_history):
            dst = self.nodes[self.role_to_idx[edge.dst_role]]
            if dst.current_hypothesis:
                return dst.current_hypothesis
        for node in self.nodes:
            if node.active and node.current_hypothesis:
                return node.current_hypothesis
        return ""
    def get_all_hypotheses(self) -> list[dict]:
        return [
            {"role": n.role, "hypothesis": n.current_hypothesis}
            for n in self.nodes
            if n.active and n.current_hypothesis
        ]
