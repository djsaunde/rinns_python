import os
import tensorflow as tf
#import tensorflow.contrib.mpi as mpi

num_gpus = len(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(','))
gpus = ['/gpu:{}'.format(gpu) for gpu in range(num_gpus)]

mpi_rank = int(os.environ['PMI_RANK'])
mpi_local_rank = int(os.environ['SLURM_LOCALID'])
mpi_size = int(os.environ['PMI_SIZE'])

print("Rank:",mpi_rank)
print("Local Rank:", mpi_local_rank)
print("Mpi Size:", mpi_size)
