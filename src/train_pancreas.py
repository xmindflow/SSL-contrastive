## Code based is based on https://github.com/WYC-321/MCF paper.
import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from networks.vnet import VNet
from networks.ResNet34 import Resnet34
from utils import ramps, losses
from dataloaders.la_heart import Pancras, RandomCrop, ToTensor, TwoStreamBatchSampler

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='C:/code/dataset/pancras', help='Name of Experiment') 
parser.add_argument('--root_pathlist', type=str, default='C:/code/deep_learning/MCF/dataset/Pancreas', help='Name of Experiment')               # todo change dataset path
parser.add_argument('--exp', type=str,  default="pancras_flod0_new2", help='model_name')                   # todo model name
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.999, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')

args = parser.parse_args()

train_data_path = args.root_path
train_data_pathlist = args.root_pathlist

snapshot_path = "../model/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (96, 96, 96)
T = 0.1
Good_student = 0 # 0: vnet 1:resnet

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def worker_init_fn(worker_id):
    random.seed(args.seed+worker_id)

def gateher_two_patch(vec):
    b, c, num = vec.shape
    cat_result = []
    for i in range(c-1):
        temp_line = vec[:,i,:].unsqueeze(1)  # b 1 c
        star_index = i+1
        rep_num = c-star_index
        repeat_line = temp_line.repeat(1, rep_num,1)
        two_patch = vec[:,star_index:,:]
        temp_cat = torch.cat((repeat_line,two_patch),dim=2)
        cat_result.append(temp_cat)

    result = torch.cat(cat_result,dim=1)
    return  result


def loss_fn(candidates, prototype):
    
    x = F.normalize(candidates, dim=0, p=2).permute(1,0).unsqueeze(0)
    y = F.normalize(prototype,  dim=0, p=2).permute(1,0).unsqueeze(0)
    

    loss = torch.cdist(x, y, p=2.0).mean()
    
    return loss


def get_confidence_prediction(prediction, percent = 40):
    _,  target = torch.max(prediction, dim=1)
    with torch.no_grad():
        prob = prediction
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
        thresh = np.percentile(
            entropy.detach().cpu().numpy().flatten(), percent
        )
        thresh_mask = entropy.ge(thresh).bool()
        target[thresh_mask] = 0
    return target


def compute_uxi_loss(predicta, predictb, represent_a, percent = 20):
    batch_size, num_class, h, w, d = predicta.shape

    logits_u_a,  label_u_a = torch.max(predicta, dim=1)
    logits_u_a, label_u_b = torch.max(predictb, dim=1)

    target = label_u_a | label_u_b
    with torch.no_grad():
        # drop pixels with high entropy from a
        prob_a = predicta
        
        entropy_a = -torch.sum(prob_a * torch.log(prob_a + 1e-10), dim=1)
        
        thresh_a = np.percentile(
            entropy_a.detach().cpu().numpy().flatten(), percent
        )
        
        thresh_mask_a = entropy_a.ge(thresh_a).bool()
        
        # drop pixels with high entropy from b
        prob_b = predictb
        
        entropy_b = -torch.sum(prob_b * torch.log(prob_b + 1e-10), dim=1)
        
        thresh_b = np.percentile(
            entropy_b.detach().cpu().numpy().flatten(), percent
        )
        
        thresh_mask_b = entropy_b.ge(thresh_b).bool()        
        
        
        thresh_mask = torch.logical_and(thresh_mask_a, thresh_mask_b)

        target[thresh_mask] = 2

        target_clone = torch.clone(target.view(-1))
        represent_a = represent_a.permute(1,0,2,3,4)
        # print(represent_a.size())
        represent_a = represent_a.contiguous().view(represent_a.size(0), -1)
        prototype_f = represent_a[:, target_clone==1].mean(dim = 1)
        prototype_b = represent_a[:, target_clone==0].mean(dim = 1)
        
        forground_candidate  = represent_a[:, (target_clone==2) & (label_u_a.view(-1)==1)]
        background_candidate = represent_a[:, (target_clone==2) & (label_u_a.view(-1)==0)]
 
        num_samples = forground_candidate.size(1) // 100
        selected_indices_f = torch.randperm(forground_candidate.size(1))[:num_samples]
        selected_indices_b = torch.randperm(background_candidate.size(1))[:num_samples]
    

        contrastive_loss_f = loss_fn(forground_candidate[:, selected_indices_f], prototype_f.unsqueeze(dim = 1))
        contrastive_loss_b = loss_fn(background_candidate[:, selected_indices_b], prototype_b.unsqueeze(dim = 1))
        contrastive_loss_c = loss_fn(prototype_f.unsqueeze(dim = 1), prototype_b.unsqueeze(dim = 1))

        con_loss = contrastive_loss_f + contrastive_loss_b + contrastive_loss_c
        
        weight = batch_size * h * w * d / torch.sum(target != 2)
    
    loss_a = weight * F.cross_entropy(predicta, target, ignore_index=2)
    loss_b = weight * F.cross_entropy(predictb, target, ignore_index=2)
    
    return loss_a, loss_b, con_loss


if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(name ='vnet'):
        # Network definition
        if name == 'vnet':
            net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
            model = net.cuda()
        if name == 'resnet34':
            net = Resnet34(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
            model = net.cuda()
        return model

    model_vnet = create_model(name='vnet')
    model_resnet = create_model(name='resnet34')

    db_train = Pancras(base_dir=train_data_path, base_dir_list = train_data_pathlist, 
                               split='train',
                               train_flod='train0.list',                   # todo change training flod
                               common_transform=transforms.Compose([
                                   RandomCrop(patch_size),
                               ]),
                               sp_transform=transforms.Compose([
                                   ToTensor(),
                               ]))
    
    labeled_idxs = list(range(12))           # todo set labeled num
    unlabeled_idxs = list(range(12, 62))     # todo set labeled num all_sample_num

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    vnet_optimizer = optim.SGD(model_vnet.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    resnet_optimizer = optim.SGD(model_resnet.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    model_vnet.train()
    model_resnet.train()
    Thresh = 0.6
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            print('epoch:{},i_batch:{}'.format(epoch_num,i_batch))
            volume_batch1, volume_label1 = sampled_batch[0]['image'], sampled_batch[0]['label']
            volume_batch2, volume_label2 = sampled_batch[1]['image'], sampled_batch[1]['label']

            v_input,v_label = volume_batch1.cuda(), volume_label1.cuda()
            r_input,r_label = volume_batch2.cuda(), volume_label2.cuda()

            v_outputs, v_rep = model_vnet(v_input)
            r_outputs, r_rep = model_resnet(r_input)

            ## calculate the supervised loss
            v_loss_seg = F.cross_entropy(v_outputs[:labeled_bs], v_label[:labeled_bs])
            v_outputs_soft = F.softmax(v_outputs, dim=1)
            v_loss_seg_dice = losses.dice_loss(v_outputs_soft[:labeled_bs, 1, :, :, :], v_label[:labeled_bs] == 1)

            r_loss_seg = F.cross_entropy(r_outputs[:labeled_bs], r_label[:labeled_bs])
            r_outputs_soft = F.softmax(r_outputs, dim=1)
            r_loss_seg_dice = losses.dice_loss(r_outputs_soft[:labeled_bs, 1, :, :, :], r_label[:labeled_bs] == 1)

            if v_loss_seg_dice < r_loss_seg_dice:
                winner = 0
            else:
                winner = 1

            ## Cross reliable loss term
            
            v_probability, v_predict = torch.max(v_outputs_soft[:labeled_bs, :, :, :, :], 1, )
            r_probability, r_predict = torch.max(r_outputs_soft[:labeled_bs, :, :, :, :], 1, )
            conf_diff_mask = (((v_predict == 1) & (v_probability>=Thresh)) ^ ((r_predict == 1) & (r_probability>=Thresh))).to(torch.int32)

            v_mse_dist = consistency_criterion(v_outputs_soft[:labeled_bs, 1, :, :, :], v_label[:labeled_bs] )
            r_mse_dist = consistency_criterion(r_outputs_soft[:labeled_bs, 1, :, :, :], r_label[:labeled_bs] )
            v_mistake  = torch.sum(conf_diff_mask * v_mse_dist) / (torch.sum(conf_diff_mask) + 1e-16)
            r_mistake  = torch.sum(conf_diff_mask * r_mse_dist) / (torch.sum(conf_diff_mask) + 1e-16)


            v_supervised_loss =  (v_loss_seg + v_loss_seg_dice) + 0.5 * v_mistake 
            r_supervised_loss =  (r_loss_seg + r_loss_seg_dice) + 0.5 * r_mistake 
            
            v_outputs_clone = v_outputs_soft[labeled_bs:, :, :, :, :].clone().detach()
            r_outputs_clone = r_outputs_soft[labeled_bs:, :, :, :, :].clone().detach()

            consistency_weight = get_current_consistency_weight(iter_num//150)
            if winner == 0:
                loss_u_r, loss_u_v , con_loss = compute_uxi_loss(r_outputs_clone, v_outputs_clone , r_rep[labeled_bs:], percent = 20)
                v_loss = v_supervised_loss + loss_u_v
                r_loss = r_supervised_loss + loss_u_r + consistency_weight * con_loss
            else:
                loss_u_v, loss_u_r, con_loss = compute_uxi_loss(v_outputs_clone, r_outputs_clone, v_rep[labeled_bs:], percent = 20)
                v_loss = v_supervised_loss + loss_u_v + consistency_weight * con_loss
                r_loss = r_supervised_loss + loss_u_r
            
            vnet_optimizer.zero_grad()
            resnet_optimizer.zero_grad()
            v_loss.backward()
            r_loss.backward()
            vnet_optimizer.step()
            resnet_optimizer.step()
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/v_loss', v_loss, iter_num)
            writer.add_scalar('loss/v_loss_seg', v_loss_seg, iter_num)
            writer.add_scalar('loss/v_loss_seg_dice', v_loss_seg_dice, iter_num)
            writer.add_scalar('loss/v_supervised_loss', v_supervised_loss, iter_num)
            writer.add_scalar('loss/con_loss', con_loss, iter_num)
            writer.add_scalar('loss/r_loss', r_loss, iter_num)
            writer.add_scalar('loss/r_loss_seg', r_loss_seg, iter_num)
            writer.add_scalar('loss/r_loss_seg_dice', r_loss_seg_dice, iter_num)
            writer.add_scalar('loss/r_supervised_loss', r_supervised_loss, iter_num)
            writer.add_scalar('train/Good_student', Good_student, iter_num)

            logging.info(
                'iteration ： %d v_supervised_loss : %f v_loss_seg : %f v_loss_seg_dice : %f r_supervised_loss : %f r_loss_seg : %f r_loss_seg_dice : %f Good_student: %f'  %
                (iter_num,
                 v_supervised_loss.item(), v_loss_seg.item(), v_loss_seg_dice.item(), 
                 r_supervised_loss.item(), r_loss_seg.item(), r_loss_seg_dice.item(), Good_student))

            ## change lr
        
            if iter_num % 2500 == 0 and iter_num!= 0:
                lr_ = lr_ * 0.1
                for param_group in vnet_optimizer.param_groups:
                    param_group['lr'] = lr_
                for param_group in resnet_optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num >= max_iterations:
                break
            time1 = time.time()

            iter_num = iter_num + 1
            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            break

    save_mode_path_vnet = os.path.join(snapshot_path, 'vnet_iter_' + str(max_iterations) + '.pth')
    torch.save(model_vnet.state_dict(), save_mode_path_vnet)
    logging.info("save model to {}".format(save_mode_path_vnet))

    save_mode_path_resnet = os.path.join(snapshot_path, 'resnet_iter_' + str(max_iterations) + '.pth')
    torch.save(model_resnet.state_dict(), save_mode_path_resnet)
    logging.info("save model to {}".format(save_mode_path_resnet))

    writer.close()