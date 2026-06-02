"""从 Reconvla 复制的动作离散化工具。

来源：Reconvla/reconvla/action_tokenizer.py
说明：为确保 AuroraIG 独立可运行，此处保留兼容副本，不依赖 activegazevla。
"""

from typing import List, Tuple, Union

import numpy as np
from transformers import PreTrainedTokenizerBase


class ActionTokenizer:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        bins: int = 256,
        min_action: int = -1,
        max_action: int = 1,
        use_norm_bins: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.n_bins = bins
        self.min_action = min_action
        self.max_action = max_action
        self.bins = self.get_bins(min_action, max_action, self.n_bins, use_norm_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self._tokenizer_vocab_size = len(self.tokenizer)
        self.action_token_begin_idx = int(self._tokenizer_vocab_size - (self.n_bins + 1))

    def __call__(self, action: np.ndarray) -> Union[Tuple[List[int], str], List[str]]:
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)

        if len(discretized_action.shape) == 1:
            real_action_token = list(self._tokenizer_vocab_size - discretized_action)
            decode_action_token = self.tokenizer.decode(real_action_token)
            return real_action_token, decode_action_token

        return self.tokenizer.batch_decode((self._tokenizer_vocab_size - discretized_action).tolist())

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        discretized_actions = self._tokenizer_vocab_size - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        return self.n_bins

    def get_bins(self, min_action, max_action, n_bins, use_norm_bins=False):
        if use_norm_bins:
            a = np.linspace(0.5, 1.0, 15, endpoint=True)
            b = np.linspace(0.25, 0.5, 40, endpoint=False)
            c = np.linspace(0.1, 0.25, 42, endpoint=False)
            d = np.linspace(-0.1, 0.1, 62, endpoint=False)
            e = np.linspace(-0.25, -0.1, 42, endpoint=False)
            f = np.linspace(-0.5, -0.25, 40, endpoint=False)
            g = np.linspace(-1, -0.5, 15, endpoint=False)
            return np.concatenate((g, f, e, d, c, b, a), axis=0)

        return np.linspace(min_action, max_action, n_bins)
