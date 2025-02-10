import random

import numpy as np
import torch


def fix_seeds(seed: int) -> None:
    """Fix random seeds for experiments reproducibility.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
