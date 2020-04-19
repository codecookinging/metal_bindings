import torch.nn as nn
import torch
import torch.nn.functional as F
# diceloss 需要input 和 label的shape一致
# 需要将label 设置成one_hot格式
# 将 类别进行soft_max
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        #print(N)
        smooth = 1

        input_flat = input.view(N, -1)
        #print(input_flat)
        target_flat = target.view(N, -1)
        intersection = input_flat * target_flat
        #print(intersection.sum(1))
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.10, gamma=2,OHEM_percent=0.2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.OHEM_percent = OHEM_percent
    def forward(self, output, target):
        output = output.contiguous().view(-1)
        target = target.contiguous().view(-1)

        max_val = (-output).clamp(min=0)
        loss = output - output * target + max_val + ((-max_val).exp() + (-output - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-output * (target * 2 - 1))
        focal_loss = self.alpha * (invprobs * self.gamma).exp() * loss

        # Online Hard Example Mining: top x% losses (pixel-wise). Refer to http://www.robots.ox.ac.uk/~tvg/publications/2017/0026.pdf
        OHEM, _ = focal_loss.topk(k=int(self.OHEM_percent * [*focal_loss.shape][0]))
        return OHEM.mean()



































# torch.set_grad_enabled(True)
# x= torch.tensor([[1,0.5,0.6,0.8],[2,0.2,0.4,0.1]],requires_grad=False)
# w = torch.tensor([[1.],[1]],requires_grad=True)
# b = torch.tensor([[1.],[1]],requires_grad=True)
# target=torch.tensor([[1.,0.,1.,0.],[0,1,0,1]],requires_grad=False)
# print(x.shape)
# y = w*x+b
# print(y.shape)
# for i in range(2):
#     y=w*x+b
#     y = F.softmax(y,dim=0)
#     print(y)
#     diceloss = DiceLoss().cuda()
#     loss=diceloss(y,target)
#
#     optimizer = torch.optim.Adam([w,b], lr = 0.001)
#     loss.backward()
#     optimizer.step()
#
#     print('loss',loss)
    # print(w,b)

