# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.utils.box_utils import match, match_with_ignorance, log_sum_exp, one_hot_embedding

# I do not fully understand this part, It completely based on https://github.com/kuangliu/pytorch-retinanet/blob/master/loss.py

def _one_hot(num_classes, labels):
    I = torch.eye(num_classes, device=labels.device)
    t = I[labels]
    return t

class FocalLoss(nn.Module):
    """SSD Weighted Loss Function
    Focal Loss for Dense Object Detection.
        
        Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

    The losses are averaged across observations for each minibatch.
    Args:
        alpha(1D Tensor, Variable) : the scalar factor for this criterion
        gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                putting more focus on hard, misclassiﬁed examples
        size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                            However, if the field size_average is set to False, the losses are
                            instead summed for each minibatch.
    """

    def __init__(self, cfg, priors, use_gpu, cfg_loss):
        super(FocalLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = cfg.NUM_CLASSES
        self.background_label = cfg.BACKGROUND_LABEL
        self.negpos_ratio = cfg.NEGPOS_RATIO
        self.threshold = cfg.MATCHED_THRESHOLD
        self.unmatched_threshold = cfg.UNMATCHED_THRESHOLD
        self.variance = cfg.VARIANCE
        self.priors = priors
        #cfg.alpha=0.70
        #cfg.gamma=2.0
        #for harpic, with dense  sku
        self.alpha = Variable(torch.ones(self.num_classes, 1) * cfg_loss.ALPHA)
        self.gamma = cfg_loss.GAMMA

        self.alpha_scalar=cfg_loss.ALPHA
        self.gamma_scalar=cfg_loss.GAMMA

        self.focal_loss = self.focal_loss_softmax if cfg_loss.CONF_DISTR=="softmax" else self.focal_loss_sigmoid


    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data = predictions
        batch_num = loc_data.size(0)
        priors = self.priors
        # priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        
        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(batch_num, num_priors, 4)
        conf_t = torch.LongTensor(batch_num, num_priors)
        for idx in range(batch_num):
            truths = targets[idx][:,:-1].data
            labels = targets[idx][:,-1].data
            defaults = priors.data
            #match(self.threshold,truths,defaults,self.variance,labels,loc_t,conf_t,idx)
            match_with_ignorance(self.threshold,self.unmatched_threshold, \
                                 truths,defaults,self.variance,labels,loc_t,conf_t,idx)

        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t,requires_grad=False)

        pos = conf_t > 0
        num_pos = max(pos.sum().item(),1.0)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1,4)
        loc_t = loc_t[pos_idx].view(-1,4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
        loss_l/=num_pos

        # Confidence Loss (Focal loss)
        # Shape: [batch,num_priors,1]
        #loss_c = self.focal_loss(conf_data.view(-1, self.num_classes), conf_t.view(-1,1))
        #loss_c = self.focal_loss_softmax(conf_data.view(-1, self.num_classes), conf_t)
        loss_c = self.focal_loss(conf_data.view(-1, self.num_classes), conf_t)

        return loss_l,loss_c

    def focal_loss_softmax(self, inputs, targets):
        '''
        targets: [batch_num, anchor_num], element type is long, <0 means ignore it, 0 mean bg, 1,2,3...is  class_num
        Focal loss.
        mean of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        '''
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)
        #P = torch.clamp(P,1e-6, 1.0- 1e-6)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)

        #in the
        ignore_mask=ids<0
        loss_mask = ids>=0
        #pos_mask = ids >0
        #neg_mask = ids == 0
        ids[ignore_mask]=0

        pos_num= max((ids>0).sum(),1)

        #get one hot
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        alpha_weight = torch.where(ids>0, alpha, 1-alpha)
        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()

        batch_loss = -alpha_weight*(torch.pow((1-probs), self.gamma))*log_p
        #batch_loss = -(torch.pow((1-probs), self.gamma))*log_p

        batch_loss_2=batch_loss[loss_mask]

        #loss = batch_loss_2.sum()*(loss_mask.sum().float()/(ids.shape[0]*targets.shape[0]))
        #loss = 5*batch_loss_2.sum()*(loss_mask.shape[0]/(ids.shape[0]*targets.shape[0]))
        #loss = batch_loss_2.sum()*(loss_mask.shape[0]/(ids.shape[0]*targets.shape[0]))
        #loss = 4 * batch_loss_2.sum()*(ids.shape[0]/(batch_loss_2.shape[0]*targets.shape[0]))
        loss = 2 * batch_loss_2.sum()*(ids.shape[0]/(batch_loss_2.shape[0]*targets.shape[0]))
        return loss


        #loss = batch_loss_2.sum()/pos_num*200

        #pos_loss = batch_loss [pos_mask].mean() * self.alpha_scalar

        #neg_loss = batch_loss [neg_mask].mean() * (1-self.alpha_scalar)

        #return  (pos_loss+neg_loss)*400

    def focal_loss_sigmoid(self, inputs, targets):
        '''
        targets: [batch_num, anchor_num], element type is long, <0 means ignore it, 0 mean bg, 1,2,3...is  class_num
        Focal loss.
        mean of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        '''

        #gamma = 2
        #alpha = 0.75
        #alpha = 0.50

        #N = inputs.size(0)
        #C = inputs.size(1)

        ids = targets.view(-1, 1)
        ids_mask = ids>=0 # -2 is the ignored box
        #pos_num= max((ids>0).sum(),1)
        # if (ids<0).sum()>0:
        #     xxxx=0
        #     xxxx=1

        good_ids = ids[ids_mask]

        good_inputs= inputs[ids_mask.squeeze(1)]
        alpha_tensor = inputs.data.new(good_ids.shape[0]).fill_(self.alpha_scalar)
        alpha_weight= torch.where(good_ids>0,alpha_tensor,1-alpha_tensor)

        class_mask = _one_hot(self.num_classes, good_ids)
        p = torch.clamp(good_inputs.sigmoid(), 1e-5, 1.0-1e-5)
        #p = good_inputs.sigmoid()
        # pt = torch.where( t > 0, p, 1-p)
        pt = class_mask * p + (1 - class_mask) * (1 - p)
        # with torch.no_grad():
        focal_weight = torch.pow(1 - pt, self.gamma_scalar)
        #alpha_weight = class_mask * alpha + (1 - class_mask) * (1 - alpha)

        # binary_cross_entropy uses sigmoid
        # per_entry_cross_entropy = -t * (p + 1e-7).log()  # - ( 1 - t ) * (1-p+1e-7).log()
        bce = -(class_mask * torch.log(p) + (1.0 - class_mask) * torch.log(1.0 - p))
        loss1 = torch.sum(focal_weight * bce,dim=1)*alpha_weight
        #loss = loss1.sum()*(4500/(good_ids.shape[0]))
        loss = loss1.sum()*(6000/(good_ids.shape[0]))
        #loss = 2*loss1.sum()*(ids.shape[0]/(good_ids.shape[0]*targets.shape[0]))
        #loss = 20*loss1.sum()/pos_num
        return loss


