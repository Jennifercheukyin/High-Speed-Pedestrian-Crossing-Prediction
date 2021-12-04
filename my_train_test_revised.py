from dataset import JAADDataset
from PIL import Image 
from torchvision import transforms, utils
from action_predict import *
from jaad_data import *
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from my_model import MyModel
from my_model_mobilenets import MobileNetModel
import pdb, os, sys
from my_utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, MovingAverage, AverageMeter_Mat, worker_init_fn
import argparse
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log1', help='Log dir [default: log]')
parser.add_argument('--epochs', type=int, default=2, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 4]')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--lamb', type=float, default=1.0, help='Weight for balancing loss terms')
parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay')

args = parser.parse_args()

args.log_dir = os.path.join('logs', args.log_dir)
os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(os.path.join(args.log_dir, 'files'), exist_ok=True)
os.system('cp *.py %s' %(os.path.join(args.log_dir, 'files')))
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

LOG_FOUT = open(os.path.join(args.log_dir, 'log_train.txt'), 'w')
LOG_FOUT.write(str(args)+'\n')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

log_string(' '.join(sys.argv))

torch.cuda.empty_cache()

train_dataset = JAADDataset('train', 'MASK_PCPA_jaad_2d')
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
test_dataset = JAADDataset('test', 'MASK_PCPA_jaad_2d')
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

model = MyModel().cuda()
weight = torch.Tensor([1760.0/8613.0, 1-1760.0/8613.0]).cuda() # [1760/8613, 1-1760/8613] for jaad_all
label_criterion = nn.CrossEntropyLoss(weight=weight)
bce_criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, \
    milestones=[50, 75], gamma=0.1)

_lambda = args.lamb

max_acc = 0.0

writer = SummaryWriter(args.log_dir + '/tb_out')

for e in range(args.epochs): 

    # train_running_loss = 0.0
    # train_running_acc = 0.0
    # test_running_loss = 0.0
    # test_running_acc = 0.0
    train_running_loss = MovingAverage(20)
    train_running_acc = MovingAverage(20)
    test_running_loss = AverageMeter()
    test_running_acc = AverageMeter()
    train_auc_roc_score = MovingAverage(20)
    test_auc_roc_score = AverageMeter()

    log_string('epoch %d' %e)
    model.train()
    for i, data in enumerate(train_dataloader): 
        # image_r, image_n = data # image_r dim n*3*h*w

        train_img_seq, train_labels, train_poses, train_speed = data
        train_img_seq = train_img_seq.cuda()
        train_labels = train_labels.cuda().long().squeeze()
        train_poses = train_poses.cuda()
        train_speed = train_speed.cuda()

        optimizer.zero_grad()
        h0 = torch.zeros(2,args.batch_size,512).cuda() # (n_layers * n_directions, batch_size, hidden_size)
        train_outputs, train_predicted_poses, train_predicted_speed = model(train_img_seq, h0) 
        print("train_outputs", train_outputs)
        
        try: 
            auc_score = roc_auc_score(train_labels.cpu(), train_outputs.detach().cpu()[:,1])
            train_auc_roc_score.update(auc_score)
            print("auc_roc_score", auc_score)
        except ValueError: 
            pass
        
        prediction = torch.softmax(train_outputs.detach(), dim=1)[:,1] > 0.5
        # print("prediction", prediction)
        prediction = prediction * 1.0
        # pdb.set_trace()
        correct = (prediction == train_labels.float()) * 1.0

        loss_labels = label_criterion(train_outputs, train_labels)
        loss_poses = bce_criterion(train_predicted_poses, train_poses)
        loss_speed = bce_criterion(train_predicted_speed, train_speed)
        loss = loss_labels + _lambda * loss_poses + _lambda * loss_speed

        log_string('pose loss: %.4f' %loss_poses.item())
        log_string('speed loss: %.4f' %loss_speed.item())

        acc = correct.sum() / train_labels.shape[0]
        train_running_loss.update(loss.item())
        train_running_acc.update(acc.item())

        loss.backward()
        optimizer.step()
        
        writer.add_scalar('Accuracy', acc.item())
        if (i + 1) % 10 == 0:    
            log_string('Train loss: %.4f, Train acc: %2.2f%%, Train auc-roc score: %2.2f%%' %(train_running_loss.avg(), train_running_acc.avg()*100.0, train_auc_roc_score.avg()))

    model.eval()
    for i, data in enumerate(test_dataloader): 
        test_img_seq, test_labels, test_poses, test_speed = data
        test_img_seq = test_img_seq.cuda()
        test_labels = test_labels.cuda().long().squeeze()
        test_poses = test_poses.cuda()
        test_speed = test_speed.cuda()

        h0 = torch.zeros(2,args.batch_size,512).cuda()
        test_outputs, test_predicted_poses, test_predicted_speed = model(test_img_seq, h0)
        
        try:
            auc_score = roc_auc_score(test_labels.cpu(), test_outputs.detach().cpu()[:,1])
            test_auc_roc_score.update(auc_score)
            print("auc_roc_score:", auc_score)
        except ValueError: 
            pass


        prediction = torch.softmax(test_outputs.detach(), dim=1)[:,1] > 0.5
        prediction = prediction * 1.0 
        # pdb.set_trace()
        correct = (prediction == test_labels.float()) * 1.0

        loss_labels = label_criterion(test_outputs, test_labels)
        loss_poses = bce_criterion(test_predicted_poses, test_poses)
        loss_speed = bce_criterion(test_predicted_speed, test_speed)
        loss = loss_labels + _lambda * loss_poses + _lambda * loss_speed
        
        acc = correct.sum() / test_labels.shape[0]
        test_running_loss.update(loss.item())
        test_running_acc.update(acc.item())
        
        if (i + 1) % 10 == 0:
            log_string('Test loss: %.4f, Test acc: %2.2f%%, Test auc-roc acore: %2.2f%%' %(test_running_loss.avg, test_running_acc.avg*100.0, test_auc_roc_score.avg()))
    log_string('Train loss: %.4f ' %train_running_loss.avg())
    log_string('Train accuracy: %2.2f%% ' %(train_running_acc.avg()*100))
    log_string('Test loss: %.4f ' %test_running_loss.avg)
    log_string('Test accuracy: %2.2f%% ' %(test_running_acc.avg*100))

    if test_running_acc.avg > max_acc:
        max_acc = test_running_acc.avg 
        torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_model.pth'))
    
    lr_scheduler.step()


