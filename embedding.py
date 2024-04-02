import torch   
def encode(model, loader, device):
    embs_list = []
    labels_list = []
    with torch.no_grad():  
        for batch in loader:
            inputs = batch["graph"].to(device)
            inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
            inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
            embs_list.append(model.encode_part(inputs).to(device=torch.device('cpu'))) 
            labels_list.append(batch["label"].to(device=torch.device('cpu')))
    return embs_list, labels_list

def get_embs(test_loaders, model, device):
    model = model.eval()
    metr = []
    embs = []
    lbs = []
    for loader in test_loaders:
        e_list, l_list = encode(model, loader, device)
        embs.append(torch.cat(e_list,dim=0).numpy())
        lbs.append(torch.cat(l_list,dim=0).numpy())
    return embs, lbs    
