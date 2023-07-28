import numpy as np
import torch
# from model_builder_mulhead_128 import MultiheadAttention
# from model_builder_mulhead_conv import MultiheadAttention
from model_builder_mulhead_conv_2 import MultiheadAttention

input1 = torch.tensor(np.zeros((1,64,10,51)),dtype=torch.float)
input2 = torch.tensor(np.zeros((1,64,10,51)),dtype=torch.float)

mulhead = MultiheadAttention(in_channels=64, num_heads=2, head_channels=32)
out = mulhead(input1)
print(out.shape)
