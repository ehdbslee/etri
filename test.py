import torch
import torch.nn as nn
from torcheval.metrics import R2Score

from torch.utils.data import DataLoader

from model import Embedding, Net


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_single(partition, args, imputation=False):
    model = Net(args.in_dim, args.out_dim, args.hid_dim, args.n_layer, args.act, args.use_bn, args.use_xavier).to(device)
    
    model.load_state_dict(torch.load(f"models/{args.exp_name}/[{args.t}, {args.hid_dim}, {args.n_layer}]model.pth"))

    criterion = nn.MSELoss()
    metric = R2Score()
    
    ## Test ##
    test_loader = DataLoader(partition[f'{args.exp_name}'], batch_size=args.test_batch_size, shuffle=False)
    
    model.eval()

    test_loss = 0 
    r_square, total = 0, 0 

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        if args.mask == True:
            if imputation == True:
                for num in range(len(args.num_features)):
                    masking_idx = torch.zeros(inputs.shape[0], args.len_categories[num]).to(device) if num in args.except_emb else torch.ones(inputs.shape[0], args.len_categories[num]).to(device)
                    inputs[:,args.num_features[num]] = masking_idx*inputs[:,args.num_features[num]]
            
            else:
                feat_num = 0
                for num in range(len(args.len_categories)):
                    masking_idx = torch.where(torch.empty(inputs.shape[0], 1).uniform_(0, 1) > args.threshold, 1.0, 0.0).squeeze().to(device)
                    inputs[:,num] = masking_idx*inputs[:,num]
                    feat_num += args.len_categories[num]
            
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        metric.update(outputs, labels)

        test_loss += loss.item()
        r_square += metric.compute().item()

    test_loss /= len(test_loader)
    test_accuracy = (r_square / len(test_loader))

    return test_loss, test_accuracy


def test_multi(partition, args, imputation=False):
    model = Net(args.emb_dim, args.out_dim, args.hid_dim, args.n_layer, args.act, args.use_bn, args.use_xavier).to(device)
    emb_list = [Embedding(args.len_categories[i], args.emb_dim, args.emb_n_layer).to(device) for i in range(len(args.len_categories))]

    model.load_state_dict(torch.load(f"models/{args.exp_name}/[{args.t}, {args.emb_dim}, {args.emb_n_layer}, {args.hid_dim}, {args.n_layer}]model.pth"))
    for i in range(len(emb_list)):
        emb_list[i].load_state_dict(torch.load( f"models/{args.exp_name}/[{args.t}, {args.emb_dim}, {args.emb_n_layer}, {args.hid_dim}, {args.n_layer}]embedding_{i}.pth"))

    criterion = nn.MSELoss()
    metric = R2Score()
    
    ## Test ##
    test_loader = DataLoader(partition[f'{args.exp_name}'], batch_size=args.test_batch_size, shuffle=False)
    
    model.eval()
    for i in range(len(emb_list)):
        emb_list[i] = emb_list[i].eval()
        
    test_loss = 0
    r_square, total = 0, 0 

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        if imputation == True:
            embed_inputs, feat_num = 0, 0
            for num in range(len(args.len_categories)):

                col_inputs = inputs[:, feat_num:feat_num+args.len_categories[num]]
                masking_idx = torch.zeros(inputs.shape[0], args.emb_dim).to(device) if num in args.except_emb else torch.ones(inputs.shape[0], args.emb_dim).to(device)
                embed_inputs += masking_idx*emb_list[num](col_inputs)
                feat_num += args.len_categories[num]
        
        else:
            embed_inputs, feat_num = 0, 0
            for num in range(len(args.len_categories)):
                masking_idx = torch.tile(torch.where(torch.empty(inputs.shape[0], 1).uniform_(0, 1) > args.threshold, 1.0, 0.0), (1,args.emb_dim)).to(device)

                col_inputs = inputs[:, feat_num:feat_num+args.len_categories[num]]
                embed_inputs += masking_idx*emb_list[num](col_inputs)
                feat_num += args.len_categories[num]
            
        outputs = model(embed_inputs)

        loss = criterion(outputs, labels)
        metric.update(outputs, labels)

        test_loss += loss.item()
        r_square += metric.compute().item()

    test_loss /= len(test_loader)
    test_accuracy = (r_square / len(test_loader))

    return test_loss, test_accuracy