from __future__ import print_function

import argparse
import json
import logging
import os
from tqdm import tqdm
import sys

import numpy as np 
import pandas as pd

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.profiler
import timm # PyTorch Image Models
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
from torchvision import transforms as T,datasets

import glob
import torch

import pickle
import random
import time
import copy
from torchsummary import  summary

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp


import torchvision.models as models

from torch.utils.tensorboard import SummaryWriter

## Set environment parameters

log_dir = '/home/ec2-user/logs/vit'
model_dir = '/home/ec2-user/output/ml'
model_file = 'CellPhenotypingViT.pt'
img_size = 75                          # Resize all the images to be 244 by 244
channels = 4
train_dir = '/home/ec2-user/input/data/train/'
test_dir = '/home/ec2-user/input/data/test/'

writer = SummaryWriter(log_dir=log_dir)

## Set up log configuration
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.getLevelName("INFO"),
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

#device = 'cuda' if torch.cuda.is_available() else 'cpu'

metrics = []

def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

class CustomDataset(Dataset):
    def __init__(self, root_dir, dim, channels, transform=None, TotalSamples=100):
        self.root_dir = root_dir
        self.transform = transform
        file_list = glob.glob(self.root_dir + "*")
        print(file_list)
        self.data = []
        self.datashape=(channels,dim,dim)
        self.TotalSamples=TotalSamples

        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for file_path in glob.glob(class_path + "/*.pickle"):
                #data.append([file_path, class_name])
                print(file_path)
                x = pickle.load(open(file_path,"rb"))
                x = tf.keras.utils.normalize(x)
                size = x.shape
                print(size)
                for image in range(size[0]):
                    im = [x[image].astype(float)]
                    im = np.array(im)
                    im = im.squeeze()
                    #channel first for PyTorch
                    im = np.moveaxis(im, source=-1, destination=0)                        
                    if im.shape == self.datashape:
                        self.data.append([im, class_name])
        self.data = self.format_data(True)
        #print(self.data)

        self.class_map = {'HCT-116': 0, 'HL60': 1, 'JURKAT': 2, 'LNCAP': 3, 'MCF7': 4, 'PC3': 5, 'THP-1': 6, 'U2OS': 7}
        self.img_dim = (dim, dim)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img, class_name = self.data[idx]
        class_id = self.class_map[class_name]
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        #class_id = torch.tensor([class_id])
        #return img_tensor, class_id
        return img_tensor.float(), class_id
    
    def class_to_idx(self):
        return print(self.class_map)
    
    #Balance the classes so that they are of equal lengths
    #dataset is a list of [image, label]
    def format_data(self, augment):
        dataset = self.data
        classes = dict([])
        class_index = []
        data = []
        X = []
        y = []
        dataset_new=[]
        reverse_class_map = {0:'HCT-116' , 1:'HL60', 2:'JURKAT', 3:'LNCAP', 4:'MCF7', 5:'PC3', 6:'THP-1', 7:'U2OS'}
        for x in dataset:
            # check if exists in unique_list or not 
            if x[1] not in list(classes.keys()):
                classes[x[1]] = 1
            else:
                classes[x[1]] = classes[x[1]] + 1
            class_index.append(x[1])
            data.append(x[0])
        print(classes.items())

        if augment == True:
            for item in list(classes.keys()):
                indicies = [i for i, x in enumerate(class_index) if x == item] 
                if len(indicies) >= self.TotalSamples:
                    indicies = random.sample(indicies, k = self.TotalSamples)
                    for i in indicies:
                        #X.append(data[i])
                        #y.append(class_index[i])
                        #class_name = reverse_class_map[class_index[i]]
                        dataset_new.append([data[i],class_index[i]])
                else:
                    aug = []
                    for i in indicies:
                        #class_name = reverse_class_map[class_index[i]]
                        dataset_new.append([data[i],class_index[i]])
                        #X.append(data[i])
                        #y.append(class_index[i])
                        aug.append(data[i])
                    new_data = self.data_augmentation(aug)
                    for i in range(len(new_data)):
                        #X.append(new_data[i])
                        #y.append(class_index[indicies[0]])
                        #class_name = reverse_class_map[class_index[i]]
                        dataset_new.append([data[i],class_index[i]])
        else:
             for item in list(classes.keys()):
                    indicies = [i for i, x in enumerate(class_index) if x == item]
                    for i in indicies:
                        #X.append(data[i])
                        #y.append(class_index[i])
                        #class_name = reverse_class_map[class_index[i]]
                        dataset_new.append([data[i],class_index[i]])
        return dataset_new
    ##Rotational data augmentation
    def data_augmentation(self, data):
        new_data = []

        for i in range(self.TotalSamples-len(data)):
            new_image = data[random.randint(1,len(data)-1)]
            for r in range(random.randint(1,3)):
                # channel first for PyTorch
                new_image = np.rot90(new_image, axes=(1,2))
            new_data.append(new_image)
        return new_data

class CellPhenotypingTrainer():
    
    def __init__(self,criterion = None,optimizer = None,schedular = None):
        
        self.criterion = criterion
        self.optimizer = optimizer
        self.schedular = schedular
    
    def train_batch_loop(self,model,trainloader,args,epoch):
        device = args.device        
        train_loss = 0.0
        train_acc = 0.0
        
        with torch.profiler.profile(
    activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True) as prof:
            for batch_idx, (images, labels) in enumerate(trainloader):
            
            #for images,labels in tqdm(trainloader): 
                
                # move the data to GPU
                images = images.to(device)
                labels = labels.to(device)
                self.optimizer.zero_grad()
                logits = model(images)
                loss = self.criterion(logits,labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                train_acc += accuracy(logits,labels)
                if batch_idx % args.log_interval == 0 and args.rank == 0:
                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]; Train Loss: {:.6f}; Train Acc: {:.6f};".format(
                            epoch,
                            batch_idx * len(images) * args.world_size,
                            len(trainloader.dataset),
                            100.0 * batch_idx / len(trainloader),
                            train_loss / len(trainloader),
                            train_acc / len(trainloader)
                        )
                    )
                if args.verbose:
                    print("Batch", batch_idx, "from rank", args.rank)            
                prof.step()  # Need to call this at the end of each step to notify profiler of steps' boundary.    

            prof.stop()

        return train_loss / len(trainloader), train_acc / len(trainloader) 

    
    def valid_batch_loop(self,model,validloader,args):
        device = args.device        
        valid_loss = 0.0
        valid_acc = 0.0
        
        with torch.no_grad():
            for images,labels in tqdm(validloader):

                # move the data to CPU
                images = images.to(device) 
                labels = labels.to(device)

                logits = model(images)
                loss = self.criterion(logits,labels)

                valid_loss += loss.item()
                valid_acc += accuracy(logits,labels)
            
        return valid_loss / len(validloader), valid_acc / len(validloader)
            
        
    def fit(self,model,trainloader,validloader,args,epochs):
        
        valid_min_loss = np.Inf 
        avg_valid_loss = 0.0
        avg_valid_acc = 0.0
        
        for i in range(epochs):
            
            model.train() # this turn on dropout
            avg_train_loss, avg_train_acc = self.train_batch_loop(model,trainloader,args,i) ###

            if args.rank == 0:
                model.eval()  # this turns off the dropout layer and batch norm
                avg_valid_loss, avg_valid_acc = self.valid_batch_loop(model,validloader,args) ###
                if avg_valid_loss <= valid_min_loss :
                    print("Valid_loss decreased {} --> {}".format(valid_min_loss,avg_valid_loss))
                    torch.save(model.state_dict(), os.path.join(model_dir, model_file))
                    valid_min_loss = avg_valid_loss
                print("Epoch : {} Valid Loss:{:.6f}; Valid Acc:{:.6f};".format(i+1, avg_valid_loss, avg_valid_acc))

            metrics.append([
                i, avg_train_loss, avg_train_acc, avg_valid_loss, avg_valid_acc
            ])
        return valid_min_loss
            

def accuracy(y_pred,y_true):
    y_pred = F.softmax(y_pred,dim = 1)
    top_p,top_class = y_pred.topk(1,dim = 1)
    equals = top_class == y_true.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))


def train_model(rank, args):
    
    print(f'setting up {rank} {args.world_size}')

    # set up the master's ip address so this child process can coordinate
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    print(f"{args.master_addr} {args.master_port}")

    # Initializes the default distributed process group, and this will also initialize the distributed package.
    dist.init_process_group(backend="nccl", rank=rank, world_size=args.world_size)
    print(f"{rank} init complete")
    
    args.world_size = args.world_size
    args.rank = rank
    args.local_rank = local_rank = int(os.getenv("LOCAL_RANK", -1))
    args.batch_size = 32
    args.device = device = torch.device('cuda', rank)
    
    if args.verbose:
        print(
            "Hello from rank",
            rank,
            "of local_rank",
            local_rank,
            "in world size of",
            args.world_size,
        )

    if not torch.cuda.is_available():
        raise CUDANotFoundException(
            "Must run smdistributed.dataparallel training on CUDA-capable devices."
        )

    torch.manual_seed(args.seed)

    # select a single rank per node to download data
    is_first_local_rank = local_rank == 0
    if is_first_local_rank:
        trainset = CustomDataset(root_dir=train_dir, dim=img_size, channels=channels, TotalSamples=10000)

    dist.barrier()  # prevent other ranks from accessing the data early

    if not is_first_local_rank:
        trainset = CustomDataset(root_dir=train_dir, dim=img_size, channels=channels, TotalSamples=10000)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, num_replicas=args.world_size, rank=rank
    )
    trainloader = DataLoader(trainset,batch_size=args.batch_size,shuffle=False,num_workers=0,pin_memory=True,sampler=train_sampler)
    
    #if rank == 0:
    testset = CustomDataset(root_dir=test_dir, dim=img_size, channels=channels, TotalSamples=1000)
    testloader = DataLoader(testset,batch_size=args.batch_size,shuffle=True)
    
    model = timm.create_model('vit_large_patch32_224_in21k', pretrained=False, img_size=img_size, in_chans=4, num_classes=8, drop_rate=0.5)
    
    model = model.to(device)

    model = DDP(model,
                      device_ids=[rank],
                      output_device=rank,
                      find_unused_parameters=True)

    summary(model,input_size=(channels,img_size,img_size))


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = args.lr)

    trainer = CellPhenotypingTrainer(criterion,optimizer)

    trainer.fit(model,trainloader,testloader,args,epochs = args.epochs)

    model.load_state_dict(torch.load(os.path.join(model_dir, model_file)))
    model.to(args.device)
    model.eval()
    with torch.autograd.profiler.profile(use_cuda=True) as inf_profiler:
        for images,labels in tqdm(testloader):
            images = images.to(args.device) 
            labels = labels.to(args.device)
            logits = model(images)
    
    print(inf_profiler.total_average())

    with open("/home/ec2-user/output/ml/inference_logs.txt", "w") as f:
        f.write(str(inf_profiler.total_average()))
    df = pd.DataFrame(metrics, columns=[
        "epoch", 
        "avg_train_loss",
        "avg_train_acc",
        "avg_valid_loss", 
        "avg_valid_acc"
    ])
    df.to_csv("/home/ec2-user/output/ml/metrics.csv", index=False)


    print('Closing summary writer')
    writer.flush()
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', type=int, default=2, metavar='W',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', type=int, default=40, metavar='E',
                        help='number of total epochs to run (default: 2)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='BS',
                        help='batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='initial learning rate (default: 0.001)')

    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N",
                        help="how many batches to wait before logging training status")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="For displaying smdistributed.dataparallel-specific logs")
    args = parser.parse_args()
    args.world_size = 8
    args.master_addr = '127.0.0.1'
    args.master_port = find_free_port()
    mp.spawn(train_model, args=(args,), nprocs=args.world_size)
