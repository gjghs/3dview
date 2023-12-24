import torch
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import argparse

parser = argparse.ArgumentParser(description = '3dview baseline')

args = parser.parse_args()


class Trainer(object):
	def __init__(self, rank=0, world_size=4, mixed_precision=False):
		self.device_count, self.rank, self.world_size, self.mixed_precision = torch.cuda.device_count(), rank, world_size, mixed_precision
		self.device = torch.device('cuda:{}'.format(self.rank)) if torch.cuda.is_available() else torch.device('cpu')
		self.__init_multi_gpu__()
		self._create_model()
		self._create_dataset()
		if self.rank==0:
			self.logger = SummaryWriter('logs')
		self.glb_iter = -1

	def __init_multi_gpu__(self):
		if self.device_count < 2:
			return
		torch.cuda.set_device(self.rank)
		dist.init_process_group(
			backend='nccl', 
			init_method='tcp://127.0.0.1:{}'.format(args.nccl_port), 
			world_size=self.world_size, 
			rank=self.rank
		)

	def _create_model(self):
		model = Model().to(self.device)



def worker(rank, world_size, *args, **kargs):
	trainer = Trainer(rank=rank, world_size=world_size, mixed_precision=False)
	trainer.do_train()

def main():
	print(args)
	world_size = torch.cuda.device_count()
	if world_size <= 1:
		worker(0, 1)
	else:
		mp.spawn(worker, nprocs=world_size, args=(world_size,))

if __name__ == '__main__':
	main()