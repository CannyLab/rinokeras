import tqdm
import time
import torch

from rinokeras.torch.functional import multi_head_attention_map, attention_map

BATCH_SIZE = 8
SEQ_LEN = 2048
REPS = 300
MAT_SIZE = 2048

# Compute the attention map on CUDA
print('Testing allocation time...')
qx = torch.randn(BATCH_SIZE, SEQ_LEN, MAT_SIZE).cuda()
kx = torch.randn(BATCH_SIZE, SEQ_LEN, MAT_SIZE).cuda()
vx = torch.randn(BATCH_SIZE, SEQ_LEN, MAT_SIZE).cuda()

# a0 = time.time()
# for _ in tqdm.tqdm(range(REPS)):
#     qx = torch.randn(BATCH_SIZE, MAT_SIZE, MAT_SIZE).cuda()
#     kx = torch.randn(BATCH_SIZE, MAT_SIZE, MAT_SIZE).cuda()
#     vx = torch.randn(BATCH_SIZE, MAT_SIZE, MAT_SIZE).cuda()
# b0 = time.time()
# print('Time:', (b0-a0)/REPS)

print('Benchmarking code...')
output = torch.zeros(BATCH_SIZE, SEQ_LEN, MAT_SIZE).cuda()
s0 = time.time()
for _ in tqdm.tqdm(range(REPS)):
#     qx = torch.randn(BATCH_SIZE, MAT_SIZE, MAT_SIZE).cuda()
#     kx = torch.randn(BATCH_SIZE, MAT_SIZE, MAT_SIZE).cuda()
#     vx = torch.randn(BATCH_SIZE, MAT_SIZE, MAT_SIZE).cuda()
    # output += torch.bmm(qx, kx)
    # output += attention_map(qx,kx,vx,return_attention_weights=False)
    output += multi_head_attention_map(qx,kx,vx,8,return_attention_weights=False)
print(output)
s1 = time.time()

# BATCH_SIZE * (MAT_SIZE ** 3 * 2)
flops = BATCH_SIZE * ( (SEQ_LEN * SEQ_LEN * MAT_SIZE * 2) * 2)
atime = (s1-s0)/REPS #- (b0-a0)/REPS

print('Avg time: ', atime)
print('GFLOPS/second: ', (flops/atime)/1000000000)


