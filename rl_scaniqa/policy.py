from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScanPolicyGRU(nn.Module):
    def __init__(
        self,
        d_feat: int = 1024,
        d_hidden: int = 512,
        d_z: int = 256,
        num_layers: int = 6,
        dropout: float = 0.0,
        use_bias: bool = True,
        global_only_first_step: bool = True,
    ):
        super().__init__()
        self.d_feat = d_feat
        self.d_hidden = d_hidden
        self.d_z = d_z
        self.num_layers = num_layers
        self.global_only_first_step = global_only_first_step

        self.gru = nn.GRU(
            input_size=d_feat,
            hidden_size=d_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.W_h = nn.Linear(d_hidden, d_z, bias=False)  # [H] -> [Z]
        self.W_g = nn.Linear(d_feat,   d_z, bias=False)  # [D] -> [Z]
        self.W_f = nn.Linear(d_feat,   d_z, bias=False)  # [D] -> [Z]

        self.v = nn.Linear(d_z, 1, bias=False)

        if use_bias:
            self.b = nn.Parameter(torch.zeros(d_z))
        else:
            self.register_parameter('b', None)

        self.value_head = nn.Sequential(
            nn.Linear(d_hidden + d_feat, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.W_init = nn.Linear(d_feat, d_hidden)

    def init_hidden(self, g: torch.Tensor) -> torch.Tensor:
        B = g.size(0)
        h0_last = torch.tanh(self.W_init(g))               # [B, H]
        h0 = h0_last.unsqueeze(0).repeat(self.num_layers, 1, 1)  # [L, B, H]
        return h0

    def forward(
        self,
        g: torch.Tensor,
        f_candidates: torch.Tensor,
        h_prev: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_first_step: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Eq.(1) logits and softmax probabilities at a given step t (no GRU update here).

        Args:
            g:           [B, D]      global feature
            f_candidates:[B, X, D]   candidate viewport features at step t
            h_prev:      [L, B, H]   previous GRU hidden (top layer is h_{t-1})
            mask:        [B, X] or None  dynamic mask (0 for valid; large negative for invalid)
            is_first_step: whether this is t=1 (affects W_f term if global_only_first_step=True)
        Returns:
            logits:      [B, X]
            probs:       [B, X]
        """
        B, X, D = f_candidates.shape
        assert g.shape == (B, self.d_feat)
        assert h_prev.shape[0] == self.num_layers and h_prev.shape[2] == self.d_hidden

        h_last = h_prev[-1]

        Whh = self.W_h(h_last).unsqueeze(1)
        Wgg = self.W_g(g).unsqueeze(1)

        if self.global_only_first_step and is_first_step:
            Wff = torch.zeros(B, X, self.d_z, dtype=Whh.dtype, device=Whh.device)
        else:
            Wff = self.W_f(f_candidates)

        u = Whh + Wgg + Wff
        if self.b is not None:
            u = u + self.b.view(1, 1, -1)

        u = torch.tanh(u)
        logits = self.v(u).squeeze(-1)

        if mask is not None:
            logits = logits + mask

        probs = F.softmax(logits, dim=-1)
        return logits, probs

    @torch.no_grad()
    def act(
        self,
        g: torch.Tensor,
        f_candidates: torch.Tensor,
        h_prev: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_first_step: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample one action and advance the GRU to the next time step.
        """
        logits, probs = self.forward(g, f_candidates, h_prev, mask, is_first_step)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()                             # [B]
        logp = dist.log_prob(action)                       # [B]
        entropy = dist.entropy()                           # [B]

        # Gather the selected viewport feature to feed GRU input
        # f_selected: [B, 1, D]
        B, X, D = f_candidates.shape
        idx = action.view(B, 1, 1).expand(B, 1, D)         # [B,1,D]
        f_selected = f_candidates.gather(dim=1, index=idx) # [B,1,D]

        # Advance GRU hidden: input [B,1,D], h_prev [L,B,H] -> h_next [L,B,H]
        _, h_next = self.gru(f_selected, h_prev)
        return action, logp, entropy, h_next, logits

    def _unpack_obs(self, obs):
        """
        Unpack an observation structure `obs` into (g, f_candidates, h_prev, mask, is_first_step).
        """
        import torch
        mask = None
        is_first_step = False

        if isinstance(obs, (tuple, list)):
            if len(obs) < 3:
                raise ValueError(f"obs tuple/list expects at least 3 items (g, f_candidates, h_prev); got len={len(obs)}")
            g, f_candidates, h_prev = obs[0], obs[1], obs[2]
            if len(obs) >= 4:
                mask = obs[3]
            if len(obs) >= 5:
                is_first_step = bool(obs[4])
            return g, f_candidates, h_prev, mask, is_first_step

        if isinstance(obs, dict):
            def _get(d, keys, default=None):
                for k in keys:
                    if k in d:
                        return d[k]
                return default
            g = _get(obs, ["g", "global", "global_feat", "global_feature"])
            f_candidates = _get(obs, ["f_candidates", "f", "candidates", "f_t", "f_list"])
            h_prev = _get(obs, ["h_prev", "h", "hidden", "hidden_prev"])
            mask = _get(obs, ["mask", "dynamic_mask", "valid_mask"], None)
            is_first_step = bool(_get(obs, ["is_first_step", "first", "t0", "is_first"], False))
            missing = [name for name, val in [("g", g), ("f_candidates", f_candidates), ("h_prev", h_prev)] if val is None]
            if missing:
                raise KeyError(f"Missing keys in obs dict: {missing}")
            return g, f_candidates, h_prev, mask, is_first_step

        if torch.is_tensor(obs):
            raise TypeError(
                "Received a single Tensor for `obs`, but ScanPolicyGRU expects a structured obs: "
                "(g [B,D], f_candidates [B,X,D], h_prev [L,B,H]). "
                "Please pass a tuple/list or dict so the policy can unpack it."
            )
        raise TypeError(f"Unsupported obs type for ScanPolicyGRU: {type(obs)}")

    def _evaluate_actions_parts(
        self,
        g: torch.Tensor,
        f_candidates: torch.Tensor,
        h_prev: torch.Tensor,
        actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_first_step: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Same semantics as the original `evaluate_actions`, but also returns the value prediction.
        """
        logits, probs = self.forward(g, f_candidates, h_prev, mask, is_first_step)
        dist = torch.distributions.Categorical(probs=probs)
        logp = dist.log_prob(actions)                      # [B]
        entropy = dist.entropy()                           # [B]
        value = self.value(g, h_prev)                      # [B]
        return logp, entropy, value

    def evaluate_actions(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # evaluate_actions(obs, actions)
        if len(args) == 2 and not isinstance(args[0], torch.Tensor):
            obs, actions = args
            g, f_candidates, h_prev, mask, is_first_step = self._unpack_obs(obs)
            return self._evaluate_actions_parts(g, f_candidates, h_prev, actions, mask, is_first_step)

        # old explicit-args style
        if len(args) >= 4:
            return self._evaluate_actions_parts(*args, **kwargs)

        raise TypeError("ScanPolicyGRU.evaluate_actions expected either (obs, actions) or (g, f_candidates, h_prev, actions, [mask], [is_first_step])")
    def value(self, g: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        Compute V([h_{t-1}; g]) (critic on the policy state).
        """

        h_last = h_prev[-1]
        s = torch.cat([h_last, g], dim=-1)
        v = self.value_head(s).squeeze(-1)
        return v
