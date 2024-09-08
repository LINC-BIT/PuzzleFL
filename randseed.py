import torch
import numpy as np
import random
def set_rand(seed=2024):

    # seed_value = 2024   # 设定随机数种子
    seed_value = seed
    
    np.random.seed(seed_value)
    random.seed(seed_value)

    torch.manual_seed(seed_value)     # 为CPU设置随机种子
    torch.cuda.manual_seed(seed_value)      # 为当前GPU设置随机种子（只用一块GPU）
    torch.cuda.manual_seed_all(seed_value)   # 为所有GPU设置随机种子（多块GPU）

    # ZUO: 
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    torch.backends.cudnn.benchmark = False