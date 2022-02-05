import torch
def maskNLLLoss(inp,target,mask,device):
    nTotal = mask.sum()
    # print(mask)
    # print(inp)
    # print(target)
    # print(torch.gather(inp,1,target.view(-1,1)))
    crossEntropy = -torch.log(torch.gather(inp,1,target.view(-1,1)).squeeze(1))
    
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss,nTotal.item()
