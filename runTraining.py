import torch
import torch.nn as nn
import torch.multiprocessing as mp
import zarr
import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from string import punctuation
from LSTMmodel import LSTMMultiClass


party_dict = {
    'SPD' : 0,
    'FDP' : 1,
    'GRUENE' : 2,
    'CDU/CSU' : 3,
    'AfD' : 4,
    'PDS/LINKE' : 5,
    'independent' : 6,
}

noWeight = [
    'independent'
]

config_arr = [
    # {'hiddenDim' : 256, 'nLayers': 2, 'lr': 0.001},
    # # {'hiddenDim' : 256, 'nLayers': 2, 'lr': 0.01},
    # # {'hiddenDim' : 512, 'nLayers': 2, 'lr': 0.001},
    # {'hiddenDim' : 1024, 'nLayers': 2, 'lr': 0.001},
    # {'hiddenDim' : 1024, 'nLayers': 2, 'lr': 0.0001}
    # {'hiddenDim' : 1536, 'nLayers': 2, 'lr': 0.001}, #using too much mem
    {'hiddenDim' : 1024, 'nLayers': 2, 'lr': 0.0005} #best so far
    # {'hiddenDim' : 1024, 'nLayers': 2, 'lr': 0.0001}
    # {'hiddenDim' : 1024, 'nLayers': 2, 'lr': 0.01}
    # {'hiddenDim' : 256, 'nLayers': 4, 'lr': 0.001},
    # {'hiddenDim' : 256, 'nLayers': 3, 'lr': 0.001},
    # {'hiddenDim' : 256, 'nLayers': 2, 'lr': 0.001},
    # {'hiddenDim' : 512, 'nLayers': 4, 'lr': 0.001},
    # {'hiddenDim' : 384, 'nLayers': 5, 'lr': 0.001}
]

def remove_comments(speech: str, delimiters: list) -> str:
    paren = 0
    res = ''
    for ch in speech:
        if ch == delimiters[0]:
            paren += 1
        elif ch == delimiters[1] and paren == 1:
            paren -= 1
        elif not paren:
            res += ch

    return res

def cut_text(text_data, label_data, maxLength):
    ret_text = []
    ret_label = []

    for i, txt in enumerate(text_data):
        crrLbl = label_data[i]
        crrTxt = txt
        tmpTxt = []
        tmpLbl = []
        tmpTxt.append(crrTxt[:maxLength])
        tmpLbl.append(crrLbl)
        crrTxt = crrTxt[maxLength:]
        while len(crrTxt) > maxLength:
            tmpTxt.append(crrTxt[:maxLength])
            tmpLbl.append(crrLbl)
            crrTxt = crrTxt[maxLength:]

        ret_text.extend(tmpTxt)
        ret_label.extend(tmpLbl)

    return np.array(ret_text, dtype=object), np.array(ret_label)


def pad_text(text_data, seq_length):

    ret = np.zeros((len(text_data), seq_length), dtype=int)

    for i, txt in enumerate(text_data):
        lenTxt = len(txt)

        if lenTxt <= seq_length:
            ret[i, (seq_length - lenTxt):] = np.array(txt)
        else:
            ret[i, :] = np.array(txt[:seq_length])

    return ret


def accuracy(y_pred, y_true):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_preds = (y_pred_tags == y_true).float()
    acc = correct_preds.sum() / len(correct_preds)

    acc = torch.round(acc * 100)

    return acc
    

def training_loop(net, train_loader, valid_loader, lr=0.001, epochs=4, batch_size=50, device=torch.device('cpu'), weights=None):

    if weights is not None:
        print('Doing weighted training!')
        weights = weights.float().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    val_losses_mean = []
    val_acc_mean = []
    
    counter = 0
    print_every = 50
    clip = 10

    net.to(device)
    
    net = net.train()

    for e in range(epochs):
        
        counter_epoch = 0
        h = net.init_hidden(batch_size)

        for inputs, labels in train_loader:
            counter += 1
            counter_epoch += 1
            
            h = tuple([each.data for each in h])
            h = tuple([each.to(device) for each in h])
            
            net.zero_grad()

            inputs = inputs.type(torch.LongTensor)
            inputs, labels = inputs.to(device), labels.to(device)
            output, h = net(inputs, h)

            loss = criterion(output, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            if counter % print_every == 0:

                val_h = net.init_hidden(batch_size)
                val_losses = []
                val_accs = []
                net.eval()

                for inputs_val, labels_val in valid_loader:

                    val_h = tuple([each.data for each in val_h])
                    val_h = tuple([each.to(device) for each in val_h])

                    inputs_val = inputs_val.type(torch.LongTensor)
                    inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)

                    output_val, val_h = net(inputs_val, val_h)
                    val_loss = criterion(output_val, labels_val)
                    val_acc = accuracy(output_val, labels_val)

                    val_losses.append(val_loss.item())
                    val_accs.append(val_acc.item())

                net.train()
                val_losses_mean.append(np.mean(val_losses))
                val_acc_mean.append(np.mean(val_accs))

                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)),
                      "Val Acc: {:.6f}".format(np.mean(val_accs)))
                
            # if e>1 and counter_epoch>10*print_every:
            #     if np.amin(np.mean(val_losses_mean[:-5])) < np.amin(val_losses_mean[-5:]):
            #         print('Early stopping, val loss did not improve over last 5 times it was evaluated!')
            #         return net
            
    return net


def main(options):

    print('Loading Data!')
    zarr_dir = zarr.load(options.inFile)
    data_x = zarr_dir['text']
    data_y = zarr_dir['party']

    print('Preparing Data!')
    data_x = [ent.lower() for ent in data_x]
    data_x = [remove_comments(ent, ['(', ')']) for ent in data_x]
    data_x = [''.join([c for c in ent if ent not in punctuation]) for ent in data_x]

    print('Getting encoding!')
    words = ' '.join(data_x).split()
    sorted_words = Counter(words).most_common(len(words))
    vocab_to_int = {w:i+1 for i, (w, c) in enumerate(sorted_words)}

    print('Getting encoded Data!')
    text_int = []
    for tt in data_x:
        text_int.append([vocab_to_int[w] for w in tt.split()])
    text_int = np.array(text_int, dtype=object)

    print('Making labels!')
    enc_party_label = np.array([party_dict[lab] if lab in party_dict.keys() else -1 for lab in data_y])
    text_int = text_int[enc_party_label != -1]
    enc_party_label = enc_party_label[enc_party_label != -1]

    print('Cutting and padding text')
    cutTxt, cutLbl = cut_text(text_int, enc_party_label, 500)
    paddedTxt = pad_text(cutTxt, 500)

    trainTxt, testTxt, trainLbl, testLbl = train_test_split(paddedTxt, cutLbl, test_size=0.2)
    trainTxt, valTxt, trainLbl, valLbl = train_test_split(trainTxt, trainLbl, test_size=0.2)
    counterLabels = Counter(trainLbl)
    # weightsVec = np.array([1./float(counterLabels[i]) for i in range(len(party_dict.keys()))])
    # weightsVec[weightsVec>1.5*np.median(weightsVec)] = 1.5 * np.median(weightsVec)
    # weightsVec[weightsVec<0.5*np.median(weightsVec)] = 0.5 * np.median(weightsVec)
    # weightsVec *= len(weightsVec)/float(weightsVec.sum())
    # print('Weights: ', weightsVec)
    # weightsTens = torch.from_numpy(weightsVec)
    
    train_data = TensorDataset(torch.from_numpy(trainTxt), torch.from_numpy(trainLbl))
    valid_data = TensorDataset(torch.from_numpy(valTxt), torch.from_numpy(valLbl))
    test_data = TensorDataset(torch.from_numpy(testTxt), torch.from_numpy(testLbl))

    batch_size = 30
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)

    train_on_gpu = torch.cuda.is_available()
    avail_gpus = options.cudaDevice.split(',') if ',' in options.cudaDevice else [options.cudaDevice]
    device = torch.device('cuda' if train_on_gpu else 'cpu') #.format(avail_gpus[0])
    print('Training on gpu') if train_on_gpu else print('Training on cpu')

    print('Getting network!')
    config = config_arr[options.config]
    print('Using config: ', config)
    net = LSTMMultiClass(len(vocab_to_int)+1, len(party_dict.keys()), 400, config['hiddenDim'], config['nLayers'], device=device)
    if len(avail_gpus) > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.distributed.init_process_group(backend='nccl', world_size=len(avail_gpus), rank=0) #store=torch.distributed.FileStore('{}/torchDistributedFilestore'.format(options.tmpDir), len(avail_gpus))
        net = nn.parallel.DistributedDataParallel(net, device_ids=avail_gpus)
        
    print('Starting to train!')
    # if len(avail_gpus) > 1:
        # mp.spawn(training_loop, kwargs=)
    net = training_loop(net, train_loader=train_loader, valid_loader=valid_loader, lr=config['lr'], epochs=10, batch_size=batch_size, device=device) #weights=weightsTens

    torch.save(net, options.saveFile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cudaDevice', '-c', action='store',default='0', type=str)
    parser.add_argument(
        '--inFile', '-i', action='store', type=str)
    parser.add_argument(
        '--tmpDir', '-t', action='store', type=str)
    parser.add_argument(
        '--saveFile', '-s', action='store', type=str)
    parser.add_argument(
        '--config', action='store', type=int)
    options = parser.parse_args()
    main(options)
