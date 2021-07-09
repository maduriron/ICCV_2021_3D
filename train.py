import os
import torch 
from models_scripts import i3_res50, i3_res50_nl, disable_bn, enable_bn
from utilities_scripts import SAM, LR_Scheduler, get_criterion, LoadingBar, Log, initialize, RandAugment
from dataset_scripts import CTDataset
import json

from torch.utils.data import DataLoader
import torchvision

############ DEFINE PARAMETERS ############
batch_size = 2
cuda_device_index = 0
rho = 0.05
learning_rate = 0.001
momentum = 0.9
weight_decay = 0.005
warmup_epochs = 5
epochs = 100
n_class = 2 # extend number of classes
fold_id = "1" #the current fold running
root = "/home/sentic/storage2/iccv_madu/fold_1"
num_workers = 2 # workers for dataloader
fold_train_path = "./train_folding.json"
fold_valid_path = "./valid_folding.json"
checkpoint_dir = "/home/sentic/storage2/iccv_madu/checkpoints/"
device = torch.device("cuda:" + str(cuda_device_index) if torch.cuda.is_available() else "cpu")
prepath = ""
replacer = ""
clip_len = 128
#######################################################

############ MODEL STUFF ################
pretrained = None

model = i3_res50_nl(n_class)

if pretrained is not None:
    model.load_state_dict(torch.load(pretrained, map_location='cuda:' + str(cuda_device_index)))

model.to(device)

#########################################

########## DATASET STUFF ##############
with open(fold_train_path) as fhandle:
    fold_splitter_train = json.load(fhandle)
    
with open(fold_valid_path) as fhandle:
    fold_splitter_valid = json.load(fhandle)
    
dataset_train = CTDataset(root=root, 
                      fold_id=fold_id, 
                      fold_splitter=fold_splitter_train,
                      transforms=None,
                      replacer="",
                      prepath="",
                      clip_len=clip_len,
                      split="train"
                      )

dataset_valid = CTDataset(root=root, 
                      fold_id=fold_id, 
                      fold_splitter=fold_splitter_valid,
                      transforms=None,
                      replacer="",
                      prepath="",
                      clip_len=clip_len,
                      split="val"
                      )

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=True, num_workers=num_workers)
#########################################################

############# CHECKPOINTING #################
#checkpoint = "/home/sentic/storage2/iccv_madu/checkpoints/checkpoint_model1_1_17.pth"
checkpoint = None

epoch_checkpoint = None
net_state_dict = None
optimizer_state_dict = None

if checkpoint is not None:
    dict_checkpoint = torch.load(checkpoint)
    epoch_checkpoint = dict_checkpoint['epoch'] + 1
    net_state_dict = dict_checkpoint['model_state_dict']
    optimizer_state_dict = dict_checkpoint['optimizer_state_dict']
    print("Initializing from checkpoint")

for param in model.parameters():
    param.requires_grad = True

if net_state_dict is not None:
    model.load_state_dict(net_state_dict)
    print("Loading model weights from checkpoint")
    
if epoch_checkpoint is not None:
    if epoch_checkpoint > warmup_epochs:
        warmup_epochs = 0
    else:
        warmup_epochs = warmup_epochs - epoch_checkpoint
    print("Setting warmup_epochs to {}".format(warmup_epochs))

if epoch_checkpoint is None:
    epoch_checkpoint = 0
    
############################

######## SOLVER STUFF ########
base_optimizer = torch.optim.SGD
optimizer = SAM(model.parameters(), base_optimizer, rho=rho, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

if optimizer_state_dict is not None:
    optimizer.load_state_dict(optimizer_state_dict)

scheduler = LR_Scheduler('cos',
                        base_lr=learning_rate,
                        num_epochs=epochs - epoch_checkpoint,
                        iters_per_epoch=len(dataloader_train),
                        warmup_epochs=warmup_epochs)

criterion = get_criterion(smooth=0.1)
log = Log(log_each=10)

###############################

##### TRAIN LOOP #####
saving_epochs = list(range(epochs))

best_pred = 0

print("Starting from epoch {}".format(epoch_checkpoint))
for epoch in range(epoch_checkpoint, epochs):
    model.train()
    log.train(len_dataset=len(dataloader_train))
    
    for ix, batch in enumerate(dataloader_train):
        scheduler(optimizer, ix, epoch, best_pred)
        inputs, targets = (b.to(device) for b in batch)
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.mean().backward()
        optimizer.first_step(zero_grad=True)

        # second forward-backward step
        # disable_bn(model)
        criterion(model(inputs), targets).mean().backward()
        # enable_bn(model)
        optimizer.second_step(zero_grad=True)

        with torch.no_grad():
            correct = torch.argmax(predictions.data, 1) == targets
            log(model, loss.cpu(), correct.cpu(), optimizer.param_groups[0]["lr"])
                
    model.eval()
    log.eval(len_dataset=len(dataloader_valid))

    with torch.no_grad():
        for batch in dataloader_valid:
            inputs, targets = (b.to(device) for b in batch)

            predictions = model(inputs)
            loss = criterion(predictions, targets)
            correct = torch.argmax(predictions, 1) == targets
            log(model, loss.cpu(), correct.cpu())
            
    if epoch in saving_epochs:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            }, os.path.join(checkpoint_dir, "checkpoint_model7_" + str(fold_id) + "_" + str(epoch) + ".pth")
        )

log.flush()   

######################################