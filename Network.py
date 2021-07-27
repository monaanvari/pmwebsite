import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision.models import resnet
from data_utils import ALLTYPES, MAX_TYPE_COUNT

base_models = {'resnet50': (resnet.resnet50, 2048),
               'resnet34': (resnet.resnet34, 512),
               'resnet18': (resnet.resnet18, 512)}

def Network_setup(args):
    args.odir = 'results/%s/%s' % (args.dataset, args.NET)
    args.odir += "_FULL" * args.regression
    args.odir = "%s/%s/MLP%d/A%.4f/" % (args.odir, args.base_model,
                                     args.mlp, args.alpha)
    args.odir += 'lr%.4f' % (args.lr)
    args.odir += '_' + args.optim
    args.odir += '_B%d' % (args.batch_size)
    args.odir = args.odir.replace(".", "p")

def Network_create_model(args):
    """ Creates model """
    model = nn.DataParallel(Network(args))
    args.nparams = sum([p.numel() for p in model.module.parameters()])
    print('Total number of parameters: {}'.format(args.nparams))
    print(model)
    model.cuda()
    return model


def Network_step(args, item):
    gts = {}
    for key in ['subid', 'classid', 'img']:
        gts[key] = Variable(torch.from_numpy(item[key]).float(), requires_grad=True).cuda()
    for key in ALLTYPES:
        gts[key] = [Variable(torch.from_numpy(item[key][i]).float(), requires_grad=True).cuda() for i in range(2)]
    imgs = gts['img'].permute((0, 3, 1, 2)).contiguous()
    outputs, loss = args.model(imgs, gts)
    loss = loss.mean()
    for key in outputs:
        outputs[key] = outputs[key].data.cpu().numpy()
    if args.model.training:
        return loss, outputs
    else:
        return loss.item(), outputs

def get_mlp(start, num_outputs, Np, scale):
    mlp = []
    for i in range(Np):
        if i == Np - 1:
            end = num_outputs
            mlp.append(nn.Linear(start,end)) #added
        else:
            end = int(start/scale)
            mlp.append(nn.Linear(start,end)) #moved inside else
            mlp.append(nn.ReLU()) #added
        start = end
    return nn.ModuleList(mlp)

def foward_modulelist(x, mod):
    for i in range(len(mod)):
        x = mod[i](x)
    return x

class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model = base_models[args.base_model][0]()
        Nfeats = base_models[args.base_model][1]
        if self.args.regression:
            self.Nc = {'b': 2, 'c': 3, 'f': 1, 'f2': 2, 'f3':3}
            fcs = [[ptype, get_mlp(Nfeats, self.Nc[ptype]*MAX_TYPE_COUNT[ptype], args.mlp, scale=2)]
                    for ptype in ALLTYPES]
            self.fcs = nn.ModuleDict(fcs)

        self.fc_material = get_mlp(Nfeats, args.num_classes, args.mlp, scale=1)
        self.conv1.weight.data.copy_(self.base_model.conv1.weight.data)
        self.class_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)

        x = self.base_model.avgpool(x)
        x = x.view(x.size(0), -1)

        output = {}
        output['classlogits'] = foward_modulelist(x, self.fc_material)
        output['classid'] = torch.argmax(output['classlogits'], dim=1)
        if self.args.regression:
            for ptype in ALLTYPES:
                out = nn.Sigmoid()(foward_modulelist(x, self.fcs[ptype]))
                S = out.shape
                output[ptype] = out.reshape(S[0], -1, self.Nc[ptype])
            output['blogits'] = output['b']
            output['b'] = torch.argmax(output['blogits'], dim=2)
            output['blogits'] = output['blogits'].reshape(-1, 2)

        #loss = self.loss(output)
        return output

    def loss(self, output, gts):
        M = self.args.alpha
        loss = self.class_loss(output['classlogits'], gts['classid'].long().squeeze())
        if self.args.regression:
            loss += M*self.class_loss(output['blogits'], gts['b'][0].long().reshape(-1))
            for ptype in [m for m in ALLTYPES if m!= 'b']:
                loss += M*(torch.pow(output[ptype]-gts[ptype][0], 2)*gts[ptype][1]).sum()
        return loss

    def step(item):

        outputs= args.model(imgs)

        for key in outputs:
            outputs[key] = outputs[key].data.cpu().numpy()

        return outputs

