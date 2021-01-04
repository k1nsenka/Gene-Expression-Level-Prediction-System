import numpy as np
import torch


import ge_loss

train_out = np.empty((50, 1024, 10), dtype=np.float32)
out = train_out
print(train_out, out)

train_out = torch.Tensor(train_out)
out = torch.Tensor(out)

acc = ge_loss.log_r2_score(out, train_out)

print(acc)