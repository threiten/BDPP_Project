import torch
import torch.nn as nn
import torch.multiprocessing as mp
import zarr
import numpy as np
import argparse
import os
import pickle as pkl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from string import punctuation
from LSTMmodel import LSTMMultiClassWrapper
from utils import party_dict, remove_comments, cut_text, pad_text

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
    {'hiddenDim': 1024, 'nLayers': 2, 'lr': 0.0005}  # best so far
    # {'hiddenDim' : 1024, 'nLayers': 2, 'lr': 0.0001}
    # {'hiddenDim' : 1024, 'nLayers': 2, 'lr': 0.01}
    # {'hiddenDim' : 256, 'nLayers': 4, 'lr': 0.001},
    # {'hiddenDim' : 256, 'nLayers': 3, 'lr': 0.001},
    # {'hiddenDim' : 256, 'nLayers': 2, 'lr': 0.001},
    # {'hiddenDim' : 512, 'nLayers': 4, 'lr': 0.001},
    # {'hiddenDim' : 384, 'nLayers': 5, 'lr': 0.001}
]


def main(options):

    print('Loading Data!')
    zarr_dir = zarr.load(options.inFile)
    data_x = zarr_dir['text']
    data_y = zarr_dir['party']

    print('Preparing Data!')
    data_x = [ent.lower() for ent in data_x]
    data_x = [remove_comments(ent, ['(', ')']) for ent in data_x]
    data_x = [''.join([c for c in ent if ent not in punctuation])
              for ent in data_x]

    print('Getting encoding!')
    words = ' '.join(data_x).split()
    sorted_words = Counter(words).most_common(len(words))
    vocab_to_int = {w: i+1 for i, (w, c) in enumerate(sorted_words)}
    with open('{}/out/vocab.pkl'.format(options.tmpDir), 'wb') as f:
        pkl.dump(vocab_to_int, f)
        f.close()

    print('Getting encoded Data!')
    text_int = []
    for tt in data_x:
        text_int.append([vocab_to_int[w] for w in tt.split()])
    text_int = np.array(text_int, dtype=object)

    print('Making labels!')
    enc_party_label = np.array(
        [party_dict[lab] if lab in party_dict.keys() else -1 for lab in data_y])
    text_int = text_int[enc_party_label != -1]
    enc_party_label = enc_party_label[enc_party_label != -1]

    print('Cutting and padding text')
    cutTxt, cutLbl = cut_text(text_int, enc_party_label, 500)
    paddedTxt = pad_text(cutTxt, 500)

    trainTxt, testTxt, trainLbl, testLbl = train_test_split(
        paddedTxt, cutLbl, test_size=0.2)
    trainTxt, valTxt, trainLbl, valLbl = train_test_split(
        trainTxt, trainLbl, test_size=0.2)
    counterLabels = Counter(trainLbl)
    # weightsVec = np.array([1./float(counterLabels[i]) for i in range(len(party_dict.keys()))])
    # weightsVec[weightsVec>1.5*np.median(weightsVec)] = 1.5 * np.median(weightsVec)
    # weightsVec[weightsVec<0.5*np.median(weightsVec)] = 0.5 * np.median(weightsVec)
    # weightsVec *= len(weightsVec)/float(weightsVec.sum())
    # print('Weights: ', weightsVec)
    # weightsTens = torch.from_numpy(weightsVec)

    train_data = TensorDataset(torch.from_numpy(
        trainTxt), torch.from_numpy(trainLbl))
    valid_data = TensorDataset(torch.from_numpy(
        valTxt), torch.from_numpy(valLbl))
    test_data = TensorDataset(torch.from_numpy(
        testTxt), torch.from_numpy(testLbl))

    batch_size = 30
    train_loader = DataLoader(train_data, shuffle=True,
                              batch_size=batch_size, drop_last=True)
    valid_loader = DataLoader(valid_data, shuffle=True,
                              batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_data, shuffle=True,
                             batch_size=batch_size, drop_last=True)

    train_on_gpu = torch.cuda.is_available()
    avail_gpus = options.cudaDevice.split(',') if ',' in options.cudaDevice else [
        options.cudaDevice]
    # .format(avail_gpus[0])
    device = torch.device('cuda' if train_on_gpu else 'cpu')
    print('Training on gpu') if train_on_gpu else print('Training on cpu')

    print('Getting network!')
    config = config_arr[options.config]
    print('Using config: ', config)
    net = LSTMMultiClassWrapper(len(vocab_to_int)+1, len(party_dict.keys()),
                                400, config['hiddenDim'], config['nLayers'], device=device)

    print('Starting to train!')
    net.train(train_loader=train_loader, valid_loader=valid_loader,
              lr=config['lr'], epochs=10, batch_size=batch_size)  # weights=weightsTens

    net.save(options.saveFile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cudaDevice', '-c', action='store', default='0', type=str)
    parser.add_argument(
        '--inFile', '-i', action='store', type=str)
    parser.add_argument(
        '--tmpDir', '-t', action='store', default='./', type=str)
    parser.add_argument(
        '--saveFile', '-s', action='store', type=str)
    parser.add_argument(
        '--config', action='store', type=int)
    options = parser.parse_args()
    main(options)
