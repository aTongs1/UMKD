import random
import argparse
import sys, os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms

from utils.stream_metrics import StreamClsMetrics, AverageMeter
from utils.metric import *
from datasets import StanfordDogs, CUB200, DRDataset, APTOSDataset
from models.resnet import resnet18, resnet34, resnet50, resnet101
from utils import mkdir_if_missing
from torchvision.transforms import InterpolationMode
from loss import FocalLoss, DiceLoss
from models.mobilenetv2 import mobile_half

_model_dict = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'MobilenetV2': mobile_half,
}

def get_parser():
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument("--data_root", type=str, default='/data4/tongshuo/Grading/CommonFeatureLearning/data')
    parser.add_argument("--dataset", type=str, default='aptos01234',
                        choices=['dogs', 'cub200', 'dr012', 'dr34', 'dr01234', 'aptos01234'])
    parser.add_argument("--model", type=str, default='MobilenetV2')

    # Train opts
    parser.add_argument("--batch_size", type=int, default=18)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu_id", type=str, default='3')
    parser.add_argument("--random_seed", type=int, default=1337)
    parser.add_argument("--download", action='store_true', default=False)
    parser.add_argument("--epochs", type=int, default=30)

    # Restore
    parser.add_argument("--ckpt", type=str, default=None)
    return parser


def train_one_epoch(cur_epoch, criterion, model, optim, train_loader, device, scheduler=None, print_interval=50):
    """Train and return epoch loss"""
    print("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))

    avgmeter = AverageMeter()
    for cur_step, (images, labels) in enumerate(train_loader):
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        # N, C, H, W
        optim.zero_grad()
        outputs,_ = model(images)
        loss = criterion(outputs, labels)

        #labels_onehot = torch.zeros((labels.size(0), 3), device=device)
        #labels_onehot[torch.arange(labels.size(0)), labels] = 1
 #########使用BCELoss时将输出转换为0-1
        #outputs_sigmoid = nn.Sigmoid()(outputs)  
        #loss = criterion(outputs_sigmoid, labels_onehot)  
   
        loss.backward()
        optim.step()
        avgmeter.update('loss', loss.item())
        avgmeter.update('interval loss', loss.item())

        if (cur_step+1) % print_interval == 0:
            interval_loss = avgmeter.get_results('interval loss')
            acc = accuracy(outputs, labels)
            mae = MAE(outputs, labels)
            print("Epoch %d, Batch %d/%d, Loss=%f, Acc=%f, MAE=%f" % \
                  (cur_epoch, cur_step+1, len(train_loader), interval_loss, acc, mae))
            avgmeter.reset('interval loss')
    
    if scheduler is not None:
        scheduler.step()

    #return avgmeter.get_results('loss') / len(train_loader) # epoch loss

def validate(model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    total_logit = []
    total_target = []
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs,_ = model(images)
            preds = outputs.detach()
            targets = labels

            if i == 0:
                total_logit = preds
                total_target = targets
            else:
                total_logit = torch.cat([total_logit, preds], dim=0)
                total_target = torch.cat([total_target, targets], 0)
            metrics.update(preds, targets)
        score = metrics.get_results()
        mae = MAE(total_logit, total_target)
        score['MAE'] = mae
    return score


def main():
    opts = get_parser().parse_args()

    # dir and log
    mkdir_if_missing('checkpoints')
    mkdir_if_missing('logs')
    #sys.stdout = Logger(os.path.join('logs', 'teacher_%s.txt'%(opts.dataset)))


    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up random seed
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    if opts.dataset == 'cub200':
        data_root = os.path.join(opts.data_root, 'cub200')
        dataset = CUB200
        num_classes = 200
    elif opts.dataset == 'dogs':
        data_root = os.path.join(opts.data_root, 'dogs')
        dataset = StanfordDogs
        num_classes = 120
    elif opts.dataset == 'dr01234':
        data_root = os.path.join(opts.data_root, 'dr012')
        dataset = DRDataset
        num_classes = 5
    elif opts.dataset == 'aptos01234':
        dataset = APTOSDataset
        num_classes = 5
    resnet = _model_dict[opts.model]
    #latest_ckpt = '/data4/tongshuo/Grading/CommonFeatureLearning/checkpoints/%s_%s_latest_es.pth'%(opts.dataset, opts.model)
    max_acc_ckpt = '/data4/tongshuo/Grading/CommonFeatureLearning/results/baselines/%s_%s_max_acc_ce_notran.pth'%(opts.model, opts.dataset)
    min_mae_ckpt = '/data4/tongshuo/Grading/CommonFeatureLearning/results/baselines/%s_%s_min_mae_ce_notran.pth'%(opts.model, opts.dataset)
    # Set up dataloader
    '''
    train_dst = dataset(root=data_root, split='train',
                        transforms=transforms.Compose([
                            transforms.Resize(size=224),
                            transforms.RandomCrop(size=(224, 224)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])]),
                        download=opts.download)

    val_dst = dataset(root=data_root, split='test',
                      transforms=transforms.Compose([
                          transforms.Resize(size=224),
                          transforms.CenterCrop(size=(224, 224)),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])]),
                      download=False)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, drop_last=True, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.batch_size, drop_last=True, shuffle=False, num_workers=4)              
    '''
    tran = transforms.Compose([
            transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            # transforms.RandomApply([
            #     transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            # ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    train_data = dataset(None,None,'train',tran,0)
    val_data = dataset(None,None,'valid',tran,0)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=opts.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=opts.batch_size, shuffle=False, drop_last=True)

    model = mobile_half(num_classes = num_classes).to(device)
    # model = resnet(pretrained=True, num_classes=num_classes).to(device)
    metrics = StreamClsMetrics(num_classes)

    params_1x = []
    params_10x = []
    for name, param in model.named_parameters():
        if 'linear' in name:
            params_10x.append(param)
        else:
            params_1x.append(param)
    #使用 不同的学习率 来优化模型的不同部分（全连接层和其他层）
    #对于 不包含 'fc' 的层（例如卷积层、批归一化层等），使用标准的学习率 opts.lr。
    #对于 包含 'fc' 的全连接层，使用较高的学习率 opts.lr * 10，以便加速这些层的训练。
    optimizer = torch.optim.Adam(params=[{'params': params_1x,  'lr': opts.lr},
                                        {'params': params_10x, 'lr': opts.lr*10}],
                                lr=opts.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.1)
    criterion = nn.CrossEntropyLoss(reduction='mean')
#######损失函数替换为针对类别不平衡的Focal Loss
    #criterion = FocalLoss([0.25, 0.25, 0.5])
#######损失函数替换为针对类别不平衡的Dice Loss
    #criterion = DiceLoss(square_denominator=True, set_level=True)
#######损失函数替换为针对类别不平衡的BCE Loss
    #criterion = nn.BCELoss()    
    # Restore
    max_acc = 0.0
    min_mae = 1000
    cur_epoch = 0
    #恢复训练：当训练过程中断（例如由于硬件故障或预设的中断条件），可以使用此代码从保存的检查点恢复训练，继续从上次中断的轮数开始训练。
    #加载预训练模型：如果你想在某个特定的检查点恢复模型以进行推理或评估，也可以使用此代码加载模型的状态。
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        cur_epoch = checkpoint["epoch"]+1
        max_acc = checkpoint['max_acc']
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] No Restoration")

    # save
    def save_ckpt(path):
        """ save current model
        """
        state = {
            "epoch": cur_epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "max_acc": max_acc,
            "min_mae": min_mae
        }
        torch.save(state, path)
        print("Model saved as %s" % path)

    # Train Loop
    while cur_epoch < opts.epochs:
        model.train()
        train_one_epoch(model=model,
                                        criterion=criterion,
                                        cur_epoch=cur_epoch,
                                        optim=optimizer,
                                        train_loader=train_loader,
                                        device=device,
                                        scheduler=scheduler)
        # print("End of Epoch %d/%d, Average Loss=%f" % (cur_epoch, opts.epochs, epoch_loss))

        # =====  Latest Checkpoints  =====
        # save_ckpt(latest_ckpt)

        # =====  Validation  =====
        print("validate on val set...")
        model.eval()           
        val_score = validate(model=model,
                             loader=val_loader,
                             device=device,
                             metrics=metrics)
        print("Confusion Matrix:\n",val_score['Confusion Matrix'])
        print(metrics.to_str(val_score))
        score_path = '/data4/tongshuo/Grading/CommonFeatureLearning/results/baselines/%s_score_%s_ce_notran.txt'%(opts.model, opts.dataset)
        score_best_path = '/data4/tongshuo/Grading/CommonFeatureLearning/results/baselines/%s_score_%s_best_ce_notran.txt'%(opts.model, opts.dataset)
        
        with open(score_path, mode='a') as f:
            f.write("Confusion Matrix:\n")
            f.write(str(val_score['Confusion Matrix']) + "\n")   
            f.write(metrics.to_str(val_score))
        # =====  Save Best Model  =====
        if val_score['Overall Acc'] > max_acc:  # save best model
            max_acc = val_score['Overall Acc']
            save_ckpt(max_acc_ckpt)

            with open(score_best_path, mode='a') as f:
                f.write("Confusion Matrix:\n")
                f.write(str(val_score['Confusion Matrix']) + "\n")   
                f.write(metrics.to_str(val_score))

        if val_score['MAE'] < min_mae:  # save best model
            min_mae = val_score['MAE']
            save_ckpt(min_mae_ckpt)

            with open(score_best_path, mode='a') as f:
                f.write("Confusion Matrix:\n")
                f.write(str(val_score['Confusion Matrix']) + "\n")   
                f.write(metrics.to_str(val_score))
        cur_epoch += 1

if __name__ == '__main__':
    main()
