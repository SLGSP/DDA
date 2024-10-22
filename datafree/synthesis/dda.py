from multiprocessing import reduction
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import datafree
from .base import BaseSynthesis
from datafree.hooks import DeepInversionHook
from datafree.criterions import kldiv, get_image_prior_losses
from datafree.utils import ImagePool, DataIter, clip_images
import torchvision.transforms.functional as TF
from torchvision import transforms
from kornia import augmentation


class MLPHead(nn.Module):
    def __init__(self, dim_in, dim_feat, dim_h=None):
        super(MLPHead, self).__init__()
        if dim_h is None:
            dim_h = dim_in

        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_h),
            nn.ReLU(inplace=True),
            nn.Linear(dim_h, dim_feat),
        )

    def forward(self, x):
        x = self.head(x)
        return F.normalize(x, dim=1, p=2)


def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

class DDASynthesizer(BaseSynthesis):
    def __init__(self, teacher, student, generator, nz, num_classes, img_size,init_dataset=None, iterations=1,
                 lr_g=1e-3, lr=0.1, synthesis_batch_size=128, sample_batch_size=128, epochs = 200,adv=0,
                 bn=0, oh=0, dda=0,normalizer=None, device='cpu', save_dir='run/dda', transform=None,dataset=None,
                 # TODO: FP16 and distributed training 
                 autocast=None, use_fp16=False, distributed=False):
        super(DDASynthesizer, self).__init__(teacher, student)
        assert len(img_size)==3, "image size should be a 3-dimension tuple"
        self.img_size = img_size 
        self.iterations = iterations
        self.nz = nz
        self.num_classes=num_classes
        self.normalizer = normalizer
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.epochs = epochs
        self.init_dataset = init_dataset

        # scaling factors
        self.lr = lr
        self.lr_g = lr_g
        self.bn = bn
        self.oh = oh
        self.dda = dda
        self.adv = adv

        # generator
        self.generator = generator.to(device).train()
        # self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=(0.5,0.999))
        self.distributed = distributed
        self.use_fp16 = use_fp16
        self.autocast = autocast # for FP16
        self.device = device

        # hooks for deepinversion regularization
        self.hooks = []
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append( DeepInversionHook(m) )
        
        # datapool
        self.save_dir = save_dir
        self.data_pool = ImagePool(root=self.save_dir)
        self.transform = transform
        self.data_iter = None
        
        with torch.no_grad():
            student.eval()
            fake_inputs = torch.randn(size=(1, *img_size), device=device)
            _, fea = student(fake_inputs, True)
            fea_dim = fea[-1].shape[1]
            del fake_inputs
        
        # auxiliary classifier
        self.head = MLPHead(fea_dim,self.num_classes*4).to(device)
        self.optimizer_head = torch.optim.SGD(self.head.parameters(), lr=self.lr, weight_decay=1e-4, momentum=0.9)
        self.dda_criterion = nn.CrossEntropyLoss().to(device)

        if dataset == 'svhn':
            self.ori_aug =transforms.Compose([ 
                normalizer,
            ])
        else:
            self.ori_aug =transforms.Compose([ 
                    augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                    augmentation.RandomHorizontalFlip(),
                    normalizer,
                ])
    
    def synthesize(self):
        self.student.eval()
        self.teacher.eval()
        self.head.eval()

        best_cost = 1e6

        best_inputs = None
        z = torch.randn(size=(self.synthesis_batch_size, self.nz), device=self.device).requires_grad_() 
        targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
        targets = targets.sort()[0] # sort for better visualization
        targets = targets.to(self.device)

        reset_model(self.generator)
        optimizer = torch.optim.Adam([{'params': self.generator.parameters()}, {'params': [z]}], self.lr_g, betas=[0.5, 0.999])

        for it in range(self.iterations):
            images = self.generator(z)
            inputs = self.ori_aug(images)

            t_out = self.teacher(inputs)
            # bns loss
            loss_bn = sum([h.r_feature for h in self.hooks])
            # one hot loss
            loss_oh = F.cross_entropy( t_out, targets )
            
            # adversarial loss
            if self.dda>0:
                size = inputs.shape[1:]
                dda_inputs = torch.stack([torch.rot90(inputs, k, (2, 3)) for k in range(4)], 1).view(-1, *size)
                _, features = self.student(dda_inputs,True)
                dda_results = self.head(features[-1])
                dda_labels = torch.stack([targets*4+i for i in range(4)], 1).view(-1)
                loss_dda = - self.dda_criterion(dda_results,dda_labels)
                loss = self.oh * loss_oh + self.bn * loss_bn + self.dda * loss_dda
            elif self.adv>0:
                s_out = self.student(inputs)
                loss_adv = -(kldiv(s_out, t_out, reduction='none').sum(1)).mean()
                loss = self.oh * loss_oh + self.bn * loss_bn + self.adv * loss_adv
            else:
                loss = self.oh * loss_oh + self.bn * loss_bn 

            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = images.data

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
       
        self.data_pool.add( best_inputs,targets=targets )
       
        dst = self.data_pool.get_dataset(transform=self.transform,labeled=True)
        if self.init_dataset is not None:
            init_dst = datafree.utils.LabeledImageDataset(self.init_dataset,transform=self.transform)
            dst = torch.utils.data.ConcatDataset([dst, init_dst])
        train_sampler = None
        loader = torch.utils.data.DataLoader(
        dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
        num_workers=4, pin_memory=True, sampler=train_sampler)
        self.data_iter = DataIter(loader)

        return {"synthetic-dda": best_inputs}
    
    def sample(self): 
        images,targets = self.data_iter.next()
        self.images = images
        self.targets = targets
        return images,targets

    # train auxiliary classifier
    def train_head(self):
        if self.dda>0:
            self.head.train()
            self.student.eval()
            self.teacher.eval()
        
            images = self.images.to(self.device)
            targets = self.targets.to(self.device)
            size = images.shape[1:]
            dda_inputs = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1).view(-1, *size)
            _, features = self.student(dda_inputs,True)
            dda_results = self.head(features[-1].detach())
            dda_labels = torch.stack([targets*4+i for i in range(4)], 1).view(-1)
            loss_dda = self.dda_criterion(dda_results,dda_labels)

            self.optimizer_head.zero_grad()
            loss_dda.backward()
            self.optimizer_head.step()

            self.student.train()

        else:
           pass

    
    def reset_head(self):
        reset_model(self.head)


