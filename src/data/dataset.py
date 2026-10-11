from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset

class StateNormalizer:

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = np.where(std < 1e-06, 1.0, std).astype(np.float32)

    @classmethod
    def fit(cls, states: np.ndarray):
        flat = states.reshape(-1, states.shape[-1])
        return cls(flat.mean(0), flat.std(0))

    def __call__(self, x):
        return (x - self.mean) / self.std

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    @classmethod
    def load(cls, d):
        return cls(np.asarray(d['mean']), np.asarray(d['std']))

class DemoDataset(Dataset):

    def __init__(self, demos: dict, normalizer: StateNormalizer, *, lambda_jitter=0.0):
        self.state = demos['state']
        self.lam = demos['lam']
        self.a0 = demos['a0']
        self.objectives = None
        if 'f1' in demos and 'f2' in demos:
            self.objectives = np.stack([demos['f1'], demos['f2']], axis=-1).astype(np.float32)
        self.norm = normalizer
        self.lambda_jitter = float(lambda_jitter)
        if not 0.0 <= self.lambda_jitter <= 0.5:
            raise ValueError('lambda_jitter must be in [0, 0.5]')

    def __len__(self):
        return len(self.lam)

    def __getitem__(self, i):
        s = self.norm(self.state[i])
        lam = float(self.lam[i])
        if self.lambda_jitter:
            lam += float((2.0 * torch.rand(()) - 1.0) * self.lambda_jitter)
            lam = min(max(lam, 0.0), 1.0)
        lam_vec = np.array([lam, 1.0 - lam], dtype=np.float32)
        sample = (torch.from_numpy(s), torch.from_numpy(lam_vec), torch.from_numpy(self.a0[i]), torch.from_numpy(self.state[i]))
        if self.objectives is None:
            return sample
        return (*sample, torch.from_numpy(self.objectives[i]))
