
from __future__ import annotations

from typing import Dict, Generator, Iterable, Optional, Tuple
import torch


class RolloutBuffer:
    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        obs_shape: Tuple[int, ...],
        action_shape: Optional[Tuple[int, ...]] = None,
        *,
        action_is_discrete: bool = False,
        device: torch.device | str = "cpu",
        obs_dtype: torch.dtype = torch.float32,
        storage_dtype: torch.dtype = torch.float32,
        # RL-ScanIQA options:
        group_index: Optional[torch.Tensor] = None,   # [N], image id per env/scanpath
        k_per_group: Optional[int] = None,            # K scanpaths per image (for Eq. (9))
    ) -> None:

        self.T = int(rollout_steps)
        self.N = int(num_envs)
        if self.T <= 0 or self.N <= 0:
            raise ValueError("rollout_steps and num_envs must be positive.")

        self.obs_shape = tuple(obs_shape)
        self.action_is_discrete = bool(action_is_discrete)
        if self.action_is_discrete:
            self.act_shape = tuple()  # discrete -> stored as [T, N]
        else:
            if action_shape is None:
                raise ValueError("action_shape must be provided for continuous actions.")
            self.act_shape = tuple(action_shape)

        self.device = torch.device(device)
        self.obs_dtype = obs_dtype
        self.storage_dtype = storage_dtype
        self.action_dtype = torch.long if self.action_is_discrete else torch.get_default_dtype()

        # grouping for K scanpaths per image (Eq. (9)) ---
        self.group_index: Optional[torch.Tensor] = None
        self.k_per_group: Optional[int] = None
        if group_index is not None:
            gi = torch.as_tensor(group_index, device=self.device, dtype=torch.long)
            if tuple(gi.shape) != (self.N,):
                raise ValueError(f"group_index must have shape [N], got {tuple(gi.shape)}")
            self.group_index = gi
            if k_per_group is not None:
                self.k_per_group = int(k_per_group)

        # --- pre-allocate time-major storage ---
        self.obs = torch.empty((self.T, self.N, *self.obs_shape), dtype=self.obs_dtype, device=self.device)
        if self.action_is_discrete:
            self.actions = torch.empty((self.T, self.N), dtype=torch.long, device=self.device)
        else:
            self.actions = torch.empty((self.T, self.N, *self.act_shape),
                                       dtype=torch.get_default_dtype(), device=self.device)
        self.logprobs  = torch.empty((self.T, self.N), dtype=self.storage_dtype, device=self.device)
        self.rewards   = torch.empty((self.T, self.N), dtype=self.storage_dtype, device=self.device)
        self.dones     = torch.empty((self.T, self.N), dtype=torch.bool,       device=self.device)
        self.values    = torch.empty((self.T, self.N), dtype=self.storage_dtype, device=self.device)

        # filled by GAE
        self.advantages = torch.empty((self.T, self.N), dtype=self.storage_dtype, device=self.device)
        self.returns    = torch.empty((self.T, self.N), dtype=self.storage_dtype, device=self.device)

        # write pointer
        self._t = 0

        # online episode stats per env
        self._ep_returns: list[float] = []
        self._ep_lengths: list[int]   = []
        self._running_ret = torch.zeros(self.N, dtype=self.storage_dtype, device=self.device)  # [N]
        self._running_len = torch.zeros(self.N, dtype=torch.long,        device=self.device)   # [N]

    # basic properties
    @property
    def is_full(self) -> bool:
        return self._t >= self.T

    def __len__(self) -> int:
        return self.T * self.N

    # write one step
    @torch.no_grad()
    def add(
        self,
        obs: torch.Tensor,        # [N, *obs_shape]
        action: torch.Tensor,     # [N] or [N, *act_shape]
        logprob: torch.Tensor,    # [N]
        reward: torch.Tensor,     # [N]
        done: torch.Tensor,       # [N] bool or {0,1}
        value: torch.Tensor,      # [N]
        *,
        check_shapes: bool = True,
    ) -> None:
        if self.is_full:
            raise RuntimeError("RolloutBuffer is full. Call reset() before adding more.")

        # move to device/dtype & ensure contiguity
        obs     = torch.as_tensor(obs,     device=self.device, dtype=self.obs_dtype).contiguous()
        logprob = torch.as_tensor(logprob, device=self.device, dtype=self.storage_dtype).contiguous()
        reward  = torch.as_tensor(reward,  device=self.device, dtype=self.storage_dtype).contiguous()
        done    = torch.as_tensor(done,    device=self.device).to(torch.bool).contiguous()
        value   = torch.as_tensor(value,   device=self.device, dtype=self.storage_dtype).contiguous()
        if self.action_is_discrete:
            action = torch.as_tensor(action, device=self.device, dtype=torch.long).contiguous()
        else:
            action = torch.as_tensor(action, device=self.device,
                                     dtype=torch.get_default_dtype()).contiguous()

        if check_shapes:
            if tuple(obs.shape) != (self.N, *self.obs_shape):
                raise ValueError(f"obs shape {tuple(obs.shape)} != {(self.N, *self.obs_shape)}")
            if self.action_is_discrete:
                if tuple(action.shape) != (self.N,):
                    raise ValueError(f"discrete action shape {tuple(action.shape)} != {(self.N,)}")
            else:
                if tuple(action.shape) != (self.N, *self.act_shape):
                    raise ValueError(f"continuous action shape {tuple(action.shape)} != {(self.N, *self.act_shape)}")
            for name, tensor, expected in [
                ("logprob", logprob, (self.N,)),
                ("reward",  reward,  (self.N,)),
                ("done",    done,    (self.N,)),
                ("value",   value,   (self.N,)),
            ]:
                if tuple(tensor.shape) != expected:
                    raise ValueError(f"{name} shape {tuple(tensor.shape)} != {expected}")

        t = self._t

        # time-major writes
        self.obs[t].copy_(obs)            # [N,*] -> [T,N,*]
        self.logprobs[t].copy_(logprob)   # [N]   -> [T,N]
        self.rewards[t].copy_(reward)     # [N]   -> [T,N]
        self.dones[t].copy_(done)         # [N]   -> [T,N]
        self.values[t].copy_(value)       # [N]   -> [T,N]
        if self.action_is_discrete:
            self.actions[t].copy_(action) # [N]   -> [T,N]
        else:
            self.actions[t].copy_(action) # [N,act_dim] -> [T,N,act_dim]

        # online episode stats per env
        self._running_ret += reward   # [N]
        self._running_len += 1        # [N]
        if done.any():
            finished = done.nonzero(as_tuple=False).squeeze(-1)  # [k]
            for idx in finished.tolist():
                self._ep_returns.append(float(self._running_ret[idx].item()))
                self._ep_lengths.append(int(self._running_len[idx].item()))
                self._running_ret[idx] = 0.0
                self._running_len[idx] = 0

        self._t += 1

    # RL-ScanIQA multi-level rewards (Eqs. (5)-(9))
    @torch.no_grad()
    def apply_multilevel_group_rewards(
        self,
        *,
        group_diversity: Optional[torch.Tensor] = None,  # [G], R_div per image (Eq. (6))
        group_mse: Optional[torch.Tensor] = None,        # [G], R_mse per image (Eq. (7), negative MSE)
        group_rank: Optional[torch.Tensor] = None,       # [G], R_rank per image (Eq. (8))
        lambda_mse: float = 0.0,                         # λ_mse in Eq. (9)
        lambda_rank: float = 0.0,                        # λ_rank in Eq. (9)
        average_step_rewards: bool = True,               # apply 1/K scaling to step-wise rt^(k) (Eq. (9))
        distribute_set_rewards_evenly: bool = True,      # add (R_div + λ*R) / K to each scanpath at t=T-1
    ) -> None:

        if self._t != self.T:
            raise RuntimeError("Call apply_multilevel_group_rewards() after filling T steps.")
        if self.group_index is None:
            raise RuntimeError("group_index is required to apply group-level rewards.")

        gi = self.group_index  # [N]
        uniq = torch.unique(gi, sorted=True)
        counts = torch.stack([(gi == g).sum() for g in uniq]).to(torch.long)  # [G]
        if (counts <= 0).any():
            raise RuntimeError("Empty group detected in group_index.")

        # determine K per group
        if self.k_per_group is not None:
            K = int(self.k_per_group)
        else:
            # infer; ensure all groups have the same K
            K = int(counts[0].item())
            if not torch.all(counts == counts[0]):
                raise RuntimeError("Inconsistent K per group; please provide k_per_group explicitly.")
        Kf = float(K)

        # (1/K) average over step-wise rewards if requested
        if average_step_rewards:
            # rewards: [T, N]; scale each group's columns by 1/K
            for g in uniq.tolist():
                idx = (gi == g).nonzero(as_tuple=False).squeeze(-1)  # [K]
                self.rewards[:, idx] /= Kf

        # set-level bonuses at terminal step (t = T-1)
        G = uniq.numel()
        # prepare bonuses per group: b_g = R_div + λ_mse*R_mse + λ_rank*R_rank
        def _zero_like(x: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x, dtype=self.storage_dtype, device=self.device)

        bonus = torch.zeros(G, dtype=self.storage_dtype, device=self.device)
        if group_diversity is not None:
            gd = torch.as_tensor(group_diversity, device=self.device, dtype=self.storage_dtype)
            if tuple(gd.shape) != (G,):
                raise ValueError(f"group_diversity must be shape [G], got {tuple(gd.shape)}")
            bonus += gd
        if group_mse is not None:
            gm = torch.as_tensor(group_mse, device=self.device, dtype=self.storage_dtype)
            if tuple(gm.shape) != (G,):
                raise ValueError(f"group_mse must be shape [G], got {tuple(gm.shape)}")
            bonus += float(lambda_mse) * gm
        if group_rank is not None:
            gr = torch.as_tensor(group_rank, device=self.device, dtype=self.storage_dtype)
            if tuple(gr.shape) != (G,):
                raise ValueError(f"group_rank must be shape [G], got {tuple(gr.shape)}")
            bonus += float(lambda_rank) * gr

        if bonus.abs().sum() > 0:
            last_t = self.T - 1
            for i, g in enumerate(uniq.tolist()):
                idx = (gi == g).nonzero(as_tuple=False).squeeze(-1)  # [K]
                add = bonus[i] / Kf if distribute_set_rewards_evenly else bonus[i]
                self.rewards[last_t, idx] += add  # inject at terminal step per scanpath

    # GAE (Eqs. (3)-(4))
    @torch.no_grad()
    def compute_gae(
        self,
        next_value: torch.Tensor,   # [N], V(s_T) after rollout
        gamma: float,
        gae_lambda: float,
        *,
        normalize_advantages: bool = False,
        eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self._t != self.T:
            raise RuntimeError(f"Buffer not full: filled steps = {self._t}, expected T = {self.T}.")
        next_value = torch.as_tensor(next_value, device=self.device, dtype=self.storage_dtype).contiguous()
        if tuple(next_value.shape) != (self.N,):
            raise ValueError(f"next_value shape {tuple(next_value.shape)} != {(self.N,)}")

        rewards = self.rewards   # [T,N]
        values  = self.values    # [T,N]
        dones   = self.dones     # [T,N] bool
        adv     = self.advantages

        last_gae = torch.zeros(self.N, dtype=self.storage_dtype, device=self.device)  # [N]
        next_val = next_value  # V(s_T) for t = T-1

        for t in reversed(range(self.T)):
            nonterminal = (~dones[t]).to(self.storage_dtype)                 # [N] in {0,1}
            delta = rewards[t] + gamma * next_val * nonterminal - values[t]  # [N]
            last_gae = delta + gamma * gae_lambda * nonterminal * last_gae   # [N]
            adv[t] = last_gae
            next_val = values[t]

        self.returns.copy_(adv + values)  # [T,N]

        if normalize_advantages:
            self.normalize_advantages_(eps=eps)

        return self.advantages, self.returns

    @torch.no_grad()
    def normalize_advantages_(self, eps: float = 1e-8) -> None:
        """Standardize advantages over flattened batch B = T * N."""
        a = self.advantages.view(-1)
        a.sub_(a.mean()).div_(a.std(unbiased=False).clamp_min(eps))

    # flatten & minibatches
    @torch.no_grad()
    def get_flat(self) -> Dict[str, torch.Tensor]:
        """Flatten [T,N,*] -> [B,*] without extra copies."""
        if self._t != self.T:
            raise RuntimeError(f"Buffer not full: filled steps = {self._t}, expected T = {self.T}.")
        B = self.T * self.N
        out: Dict[str, torch.Tensor] = {
            "obs":        self.obs.view(B, *self.obs_shape),
            "logprobs":   self.logprobs.view(B),
            "rewards":    self.rewards.view(B),
            "dones":      self.dones.view(B),
            "values":     self.values.view(B),
            "advantages": self.advantages.view(B),
            "returns":    self.returns.view(B),
        }
        if self.action_is_discrete:
            out["actions"] = self.actions.view(B)                  # [B]
        else:
            out["actions"] = self.actions.view(B, *self.act_shape) # [B, act_dim]
        return out

    @torch.no_grad()
    def iter_minibatches(
        self,
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        yield_dict: bool = False,
    ) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """
        Yield mini-batches for PPO update. Shapes per mini-batch M:
          obs[M,*], actions[M or M,act_dim], logprobs[M],
          advantages[M], returns[M], values[M]
        """
        data = self.get_flat()
        B = self.T * self.N
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if batch_size > B:
            raise ValueError(f"batch_size ({batch_size}) > total B ({B})")

        idx = torch.arange(B, device=self.device)
        if shuffle:
            idx = idx[torch.randperm(B, device=self.device)]

        num_full = B // batch_size
        remainder = B % batch_size
        num_batches = num_full if (drop_last or remainder == 0) else (num_full + 1)

        for i in range(num_batches):
            s = i * batch_size
            e = min(s + batch_size, B)
            if e <= s:
                break
            mb = idx[s:e]
            obs_mb = data["obs"][mb]
            act_mb = data["actions"][mb]
            log_mb = data["logprobs"][mb]
            adv_mb = data["advantages"][mb]
            ret_mb = data["returns"][mb]
            val_mb = data["values"][mb]
            if yield_dict:
                yield {"obs": obs_mb, "actions": act_mb, "logprobs": log_mb,
                       "advantages": adv_mb, "returns": ret_mb, "values": val_mb}
            else:
                yield obs_mb, act_mb, log_mb, adv_mb, ret_mb, val_mb

    def reset(self) -> None:
        """Reset write pointer and per-env running stats (tensors remain allocated)."""
        self._t = 0
        self._running_ret.zero_()
        self._running_len.zero_()

    def clear_episode_stats(self) -> None:
        self._ep_returns.clear()
        self._ep_lengths.clear()

    @property
    def episode_returns(self) -> Iterable[float]:
        return self._ep_returns

    @property
    def episode_lengths(self) -> Iterable[int]:
        return self._ep_lengths
