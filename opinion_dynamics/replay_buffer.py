import torch
from collections import deque
import numpy as np
import random
import pickle


class ReplayBuffer:
    def __init__(
        self, max_size, state_dim, action_dim=1, n_step=0, gamma=0.9, betas=None
    ):
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_step = n_step
        self.buffer = deque(maxlen=self.max_size)
        self.gamma = gamma
        self.betas = betas

    def __len__(self):

        return len(self.buffer)

    def append(self, state, action, reward, next_state, done):
        """
        Append a transition to the replay buffer.
        Normalize the state and next_state tensors before storing.
        """
        state = np.array(state).reshape(-1)  # Flatten to 1D
        next_state = np.array(next_state).reshape(-1)
        
        beta_idx, w = action
        beta_idx = int(np.asarray(beta_idx).reshape(-1)[0])
        
        # storage contract: w is (J, N)
        assert w.ndim == 2, f"Expected w to be (J, N), got {w.shape}"

        self.buffer.append((state, (beta_idx, w), reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the replay buffer.
        Ensure that the states and next_states are stacked into tensors.
        """
        if batch_size > len(self):
            raise ValueError("Not enough transitions to sample")

        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        # Split action into beta indices and w vectors
        beta_indices, ws = zip(*actions)

        states = torch.tensor(np.stack(states), dtype=torch.float32)
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32)
        beta_indices = torch.tensor(beta_indices, dtype=torch.long)
        ws = torch.tensor(np.stack(ws), dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        return states, (beta_indices, ws), rewards, next_states, dones

    def sample_n_step(self, batch_size):
        """
        Return a batch of n-step transitions:
        (s_t, (beta_idx_t, w_t), R_t^{(n)}, s_{t+n}, done_{t+n})
        where R_t^{(n)} = r_t + gamma r_{t+1} + ... + gamma^{n-1} r_{t+n-1},
        and we stop early if a 'done' is encountered.
        """
        # If n-step is disabled, just use the regular sampler.
        if not isinstance(self.n_step, int) or self.n_step <= 1:
            return self.sample(batch_size)

        if batch_size > len(self):
            raise ValueError("Not enough transitions to sample")

        buf = list(self.buffer)  # local indexable view
        N = len(buf)
        gamma = float(getattr(self, "gamma", 0.9))  # default if not set

        # Choose arbitrary starting indices; we will clamp by episode end (done) when rolling forward.
        idxs = random.sample(range(N), batch_size)

        S0, BIDX0, W0, Rn, S_next_n, Dn = [], [], [], [], [], []

        for start in idxs:
            s0, (beta_idx0, w0), r0, ns0, d0 = buf[start]

            # Accumulate n-step return forward from 'start'
            R = 0.0
            g = 1.0
            last_next_state = ns0
            done_n = bool(d0)

            # Walk forward up to n steps or until episode ends.
            for k in range(self.n_step):
                i = start + k
                if i >= N:
                    break  # ran out of buffer; keep what we have
                s_i, a_i, r_i, ns_i, d_i = buf[i]
                R += g * float(r_i)
                last_next_state = ns_i
                done_n = bool(d_i)
                g *= gamma
                if d_i:
                    break

            # Collect the n-step transition anchored at 'start'
            S0.append(np.asarray(s0, dtype=np.float32))
            BIDX0.append(int(beta_idx0))
            W0.append(np.asarray(w0, dtype=np.float32))
            Rn.append(R)
            S_next_n.append(np.asarray(last_next_state, dtype=np.float32))
            Dn.append(float(done_n))

        # Convert to tensors with shapes compatible with model_learn(...)
        states = torch.tensor(np.stack(S0, axis=0), dtype=torch.float32)
        next_states = torch.tensor(np.stack(S_next_n, axis=0), dtype=torch.float32)
        beta_indices = torch.tensor(BIDX0, dtype=torch.long)
        ws = torch.tensor(np.stack(W0, axis=0), dtype=torch.float32)
        rewards = torch.tensor(Rn, dtype=torch.float32).view(-1, 1)
        dones = torch.tensor(Dn, dtype=torch.float32).view(-1, 1)

        return states, (beta_indices, ws), rewards, next_states, dones

    def save(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self, file_name):
        with open(file_name, "rb") as f:
            self.buffer = pickle.load(f)
