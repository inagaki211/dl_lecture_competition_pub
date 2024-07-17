# import torch
# print(torch.version.cuda)
# print(torch.version.cuda)
# print(torch.cuda.is_available())

import os
print(os.path.exists('data/train_X/00000.npy'))  # Trueを期待
print(os.path.exists('data/train_y/00000.npy'))  # Trueを期待
print(os.path.exists('data/train_subject_idxs/00000.npy'))  # Trueを期待