import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
import os
import re
import logging
from tqdm import tqdm

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler,
                              TensorDataset)

logger = logging.getLogger()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

DATA_HOME = '/dgxhome/cra5302/MMHS'
TRAINFILE = 'Train/train.csv'
TESTFILE = 'Test/test.csv'
UNLABELED = 'Train/unlabeled.csv'

NEWTRAINFILE = 'train_iter'
NEWTESTFILE = 'test_iter'
NEWUNLABELED = 'unlabeled_iter'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    return parsed_text


def tokenize_sample(text, tokenizer, max_seq_length):
    text = preprocess(text)
    text = text.lower()

    tokens = tokenizer.tokenize(text) 
    tokens = tokens[ : (max_seq_length - 2)] 
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]

    token_len = len(tokens)
    padding_len = max_seq_length - token_len

    tokens = tokens + ([tokenizer.pad_token] * padding_len)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * token_len
    input_mask = input_mask + ([0] * padding_len)

    assert(len(input_ids) == max_seq_length)
    assert(len(input_mask) == max_seq_length)

    return input_ids, input_mask


def tokenize(df, tokenizer, max_seq_length):
    input_ids = []
    input_masks = []

    for index, row in tqdm(df.iterrows(), desc="Tokenize"):
        sent = row["Text"]
        ids, mask = tokenize_sample(sent, tokenizer, max_seq_length)
        input_ids.append(ids)
        input_masks.append(mask)
    
    b_input_ids = []
    b_input_masks = []

    for index, row in tqdm(df.iterrows(), desc="b_Tokenize"):
        sent = row["bt_Text"]
        ids, mask = tokenize_sample(sent, tokenizer, max_seq_length)
        b_input_ids.append(ids)
        b_input_masks.append(mask)

    return np.array(input_ids), np.array(input_masks), np.array(b_input_ids), np.array(b_input_masks)


# Model parameter
MAX_SEQ_LEN = 128


def read_files():
    train_df = pd.read_csv(os.path.join(DATA_HOME, TRAINFILE))
    test_df = pd.read_csv(os.path.join(DATA_HOME, TESTFILE))
    unlabeled_df = pd.read_csv(os.path.join(DATA_HOME, UNLABELED))
    unlabeled_df = unlabeled_df.head(30000)

    return train_df, test_df, unlabeled_df


def read_files_ensemble(ensemble_home, iteration=1):
    train_df = pd.read_csv(ensemble_home + '/' + NEWTRAINFILE + str(iteration) + ".csv") 
    test_df = pd.read_csv(os.path.join(DATA_HOME, TESTFILE))
    unlabeled_df = pd.read_csv(ensemble_home + '/' + NEWUNLABELED + str(iteration) + ".csv")

    return train_df, test_df, unlabeled_df


def get_dataset(tokenizer, fix_length=MAX_SEQ_LEN, 
                label="Label", cache_path=None, 
                ensemble_home=None, iteration=0):
    if iteration == 0:
        train_df, test_df, unlabeled_df = read_files()
    else:
        train_df, test_df, unlabeled_df = read_files_ensemble(ensemble_home, iteration=iteration)
        
    
    def make_weights_for_balanced_classes(y, nclasses=6):
        count = [0] * nclasses                                                      
        for item in y:                                                         
            count[item] += 1                                                     
        print ("Counts", count)
        weight_per_class = [0.] * nclasses                                      
        N = float(sum(count))                                                   
        for i in range(nclasses):                                                   
            weight_per_class[i] = N/float(count[i])
        weight = [0] * len(y)                                              
        for idx, val in enumerate(y):                                          
            weight[idx] = weight_per_class[val]                                  
        return weight
    
    
    def iter_folds(): 
        X, mask, b_X, b_mask = tokenize(train_df, tokenizer, max_seq_length=fix_length) 
        y = train_df[label].to_numpy() 
        
        kf = StratifiedKFold(n_splits=5)

        for index, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            # print ("Before shuffling", train_idx)
            np.random.shuffle(train_idx)
            # print ("After shuffling", train_idx)

            yield ( 
                TensorDataset( 
                    torch.tensor(X[train_idx], dtype=torch.long), 
                    torch.tensor(mask[train_idx], dtype=torch.long), 
                    torch.tensor(b_X[train_idx], dtype=torch.long), 
                    torch.tensor(b_mask[train_idx], dtype=torch.long), 
                    torch.tensor(y[train_idx], dtype=torch.long) 
                ), 
                make_weights_for_balanced_classes(y[train_idx]),
                TensorDataset( 
                    torch.tensor(X[val_idx], dtype=torch.long), 
                    torch.tensor(mask[val_idx], dtype=torch.long), 
                    torch.tensor(b_X[val_idx], dtype=torch.long), 
                    torch.tensor(b_mask[val_idx], dtype=torch.long), 
                    torch.tensor(y[val_idx], dtype=torch.long) 
                ) 
            ) 

            if index == 0:
                break
        
    # test 
    X_test, mask_test, b_X_test, b_mask_test = tokenize(test_df, tokenizer, max_seq_length=fix_length) 
    y_test = test_df[label].to_numpy() 

    X_test = torch.tensor(X_test, dtype=torch.long) 
    mask_test = torch.tensor(mask_test, dtype=torch.long) 
    b_X_test = torch.tensor(b_X_test, dtype=torch.long) 
    b_mask_test = torch.tensor(b_mask_test, dtype=torch.long) 
    y_test = torch.tensor(y_test, dtype=torch.long) 

    cache_file = os.path.join(cache_path, "unlabeled_processed_" + str(iteration) + ".pt")
    if not os.path.exists(cache_file):
        # unlabeled
        X_unlabeled, mask_unlabeled, b_X_unlabeled, b_mask_unlabeled = tokenize(unlabeled_df, tokenizer, max_seq_length=fix_length) 
        torch.save((X_unlabeled, mask_unlabeled, b_X_unlabeled, b_mask_unlabeled), cache_file)
    else:
        print ("Loading processed unlabeled samples from {}".format(cache_file))
        X_unlabeled, mask_unlabeled, b_X_unlabeled, b_mask_unlabeled = torch.load(cache_file, map_location=device)
        
    X_unlabeled = torch.tensor(X_unlabeled, dtype=torch.long) 
    mask_unlabeled = torch.tensor(mask_unlabeled, dtype=torch.long) 
    b_X_unlabeled = torch.tensor(b_X_unlabeled, dtype=torch.long) 
    b_mask_unlabeled = torch.tensor(b_mask_unlabeled, dtype=torch.long) 

    return iter_folds(), TensorDataset(X_test, mask_test, b_X_test, b_mask_test, y_test), TensorDataset(X_unlabeled, mask_unlabeled, b_X_unlabeled, b_mask_unlabeled)


def get_iterator(dataset, batch_size, shuffle=True, weights=None):
    print ("Get iterator of dataset with length {}".format(len(dataset)))
    if shuffle and weights != None: 
        weights = torch.DoubleTensor(weights)
        sampler = WeightedRandomSampler(weights, len(weights))
    elif shuffle:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    dataset_iter = DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size
    )
    return dataset_iter

def add_ensemble_data(labels, confs, LABEL="Label", ensemble_home=None, iteration=0):
    print ("Ensemble iteration", iteration) 
    print ("Total samples", len(labels)) 
    if iteration == 0: 
        unlabeled_df = pd.read_csv(os.path.join(DATA_HOME, UNLABELED)) 
        unlabeled_df = unlabeled_df.head(30000) 
        print ("Original test file:", os.path.join(DATA_HOME, UNLABELED)) 
    else: 
        unlabeled_df = pd.read_csv(ensemble_home + '/' + NEWUNLABELED + str(iteration) + '.csv') 
        print ("Ensembled test file:", ensemble_home + '/' + NEWUNLABELED + str(iteration) + '.csv') 

    unlabeled_df[LABEL] = labels 
    
    if iteration == 0: 
        train_df = pd.read_csv(os.path.join(DATA_HOME, TRAINFILE)) 
        print (train_df.shape)
    else: 
        train_df = pd.read_csv(ensemble_home + '/' + NEWTRAINFILE + str(iteration) + ".csv") 
    
    # keep the train distribution same in the new trainset
    counts = [10534,   950,   279,   309,    13,   465] 
    print ('New train samples sizes', counts)

    sorted_conf_ids = np.argsort(-confs)
    
    selected_ids = []
    for lbl in range(6):
        ids = [i for i in sorted_conf_ids if labels[i] == lbl]
        ids = ids[:counts[lbl]]
        print ("Least confidence for label {} is {}".format(lbl, confs[ids[-1]]))
        selected_ids.extend(ids)
    
    unselected_ids = list(set(range(unlabeled_df.shape[0])) - set(selected_ids))

    unlabeled_df_sel = unlabeled_df.iloc[selected_ids, :] 
    unlabeled_df_unsel = unlabeled_df.iloc[unselected_ids, :] 

    new_train_df = train_df.append(unlabeled_df_sel, ignore_index=True, sort=False) 
    new_train_df.fillna(0, inplace=True) 
    print ('New train set size', new_train_df.shape[0]) 
    
    newtrain_filename = ensemble_home + '/' + NEWTRAINFILE + str(iteration + 1) + '.csv' 
    print ("New train file", newtrain_filename) 
    new_train_df.to_csv(newtrain_filename, index=False) 
    
    newunlabeled_filename = ensemble_home + '/' + NEWUNLABELED + str(iteration + 1) + '.csv' 
    print ("New train file", newunlabeled_filename) 
    unlabeled_df_unsel.to_csv(newunlabeled_filename, index=False) 
    
    print ("\n\n ==================================================================== \n\n") 
    
    return unlabeled_df 

