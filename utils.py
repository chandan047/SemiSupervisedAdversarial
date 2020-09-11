import torch
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# def save_checkpoint(save_path, model):

#     if save_path == None:
#         return

#     model.save_pretrained(save_path)

#     print(f'Model saved to ==> {save_path}')


# def load_checkpoint(load_path, model):

#     if load_path==None:
#         return

#     state_dict = torch.load(load_path, map_location=device)
#     print(f'Model loaded from <== {load_path}')

#     model.load_state_dict(state_dict['model_state_dict'])
#     return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    # print(f'Metrics saved to ==> {save_path}')


def load_metrics(load_path):

    if load_path==None:
        return

    state_dict = torch.load(load_path, map_location=device)
    # print(f'Metrics loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


def evaluate(model, test_loader):
    y_conf = []
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluate unlabeled"):
            masks = batch[1].type(torch.LongTensor)
            masks = masks.to(device)
            comments = batch[0].type(torch.LongTensor)
            comments = comments.to(device)
            outputs = model(input_ids=comments, attention_mask=masks)

            logits = outputs
            # print (logits.shape)
            output = F.softmax(logits, dim=1)
            y_pred.extend(torch.argmax(output, 1).tolist())
            y_conf.extend(torch.max(output, 1).values.tolist())
    
    return y_pred, y_conf


def evaluate_metrics(model, test_loader):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            labels = batch[4].type(torch.LongTensor)
            labels = labels.to(device)
            b_masks = batch[3].type(torch.LongTensor) 
            b_masks = b_masks.to(device) 
            masks = batch[1].type(torch.LongTensor) 
            masks = masks.to(device) 
            b_comments = batch[2].type(torch.LongTensor)  
            b_comments = b_comments.to(device)
            comments = batch[0].type(torch.LongTensor)  
            comments = comments.to(device)
            outputs = model(input_ids=comments, input_ids_adv=b_comments,
                            attention_mask=masks, attention_mask_adv=b_masks,
                            labels=labels)
            loss, logits = outputs[:2]
            
            output = F.softmax(logits, dim=1)
            # print (output)
            y_pred.extend(torch.argmax(output, 1).tolist())
            y_true.extend(labels.tolist())

    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[0,1,2,3,4,5], digits=4))
    

def agreement(outputs):
    agreecnt = 0
    preds_ens = [output[0] for output in outputs]

    for preds in zip(*preds_ens):
        cnts = [0, 0, 0, 0, 0, 0]
        for pred in preds:
            cnts[pred] += 1
        if max(cnts) == 5:
            agreecnt += 1

    print ("Agreement in ensemble is {:.2f}".format(agreecnt * 100 / len(preds_ens[0])))
    
    
def predict_ensemble(outputs):
    y_pred = []
    y_conf = []

    ens_pred = []   # (ENS_NUM, BATCH_SIZE) 
    ens_conf = []   # (ENS_NUM, BATCH_SIZE) 
    
    for output in outputs:
        pred, conf = output
        ens_pred.append(pred) 
        ens_conf.append(conf) 
        
    # calculate agreement between predictions
    agreement(outputs)

    for b in range(len(ens_pred[0])):
        counts = [0, 0, 0, 0, 0, 0]
        confs = [[], [], [], [], [], []]

        for e in range(len(ens_pred)): 
            pred = ens_pred[e][b]
            counts[pred] += 1
            confs[pred].append(ens_conf[e][b])

        for i in range(6):
            confs[i].sort(reverse=True)
        
        pred = counts.index(max(counts))
        
        if max(counts) >= 2 and confs[pred][1] > 0.8: 
            # 4:1 or 5:0
            y_pred.append(pred) 
            y_conf.append(confs[pred][1]) 
        else:
            # ignore this case
            y_pred.append(-1)
            y_conf.append(0.)

    y_pred = np.array(y_pred) 
    y_conf = np.array(y_conf) 

    return y_pred, y_conf 
    
    
# def predict_ensemble(ens_model, test_loader):
#     y_pred = []
#     y_conf = []

#     ens_model.eval()
#     with torch.no_grad():
#         for batch in tqdm(test_loader, desc='Ensemble'):
#             masks = batch[1].type(torch.LongTensor) 
#             masks = masks.to(device) 
#             comments = batch[0].type(torch.LongTensor) 
#             comments = comments.to(device) 
#             outputs = ens_model(comments, masks) 


#             ens_pred = []   # (ENS_NUM, BATCH_SIZE) 
#             ens_conf = []   # (ENS_NUM, BATCH_SIZE) 

#             for output, in outputs:
#                 output = F.softmax(output, dim=1)
#                 ens_pred.append(torch.argmax(output, 1).tolist()) 
#                 ens_conf.append(torch.max(output, 1).values.tolist()) 

#             for b in range(len(ens_pred[0])):
#                 pos, neg = 0, 0
#                 best_pconf, best_nconf = float('inf'), float('inf') 

#                 for e in range(len(ens_pred)): 
#                     if (ens_pred[e][b] == 0): 
#                         neg += 1 
#                         best_nconf = min(best_nconf, ens_conf[e][b]) 
#                     else: 
#                         pos += 1 
#                         best_pconf = min(best_pconf, ens_conf[e][b]) 

#                 if (pos > neg): 
#                     y_pred.append(1) 
#                     y_conf.append(best_pconf) 
#                 else: 
#                     y_pred.append(0) 
#                     y_conf.append(best_nconf) 

#     y_pred = np.array(y_pred) 
#     y_conf = np.array(y_conf) 

#     return y_pred, y_conf 

