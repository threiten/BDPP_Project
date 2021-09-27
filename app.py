import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import plotly.express as px

import LSTMmodel
import torch
import pickle as pkl
from string import punctuation
from utils import pad_text, remove_comments, party_dict, cut_text
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import zarr
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter


def predictParty(text):
    predTxt = text.lower()
    predTxt = ''.join([c for c in predTxt if c not in punctuation])
    predArr = predTxt.split()
    predIntsRaw = []
    predIntsRaw.append(
        [vocab[wrd] if wrd in vocab.keys() else 0 for wrd in predArr])
    predInts = pad_text(predIntsRaw, 500)
    predIntsTns = torch.from_numpy(predInts)
    clsPred = net.predict(predIntsTns)
    clsProbs = net.predict(predIntsTns, True)
    nUnrec = Counter(predIntsRaw[0])[0]

    return clsPred, clsProbs, nUnrec


vocab = pkl.load(
    open('/eos/home-h/hig19016review/BDPP_Project/Data/vocab.pkl', 'rb'))
net = LSTMmodel.LSTMMultiClassWrapper(vocab_size=len(
    vocab)+1, output_size=7, embedding_dim=400, hidden_dim=1024, n_layers=2)
net.loadModel(
    '/eos/home-h/hig19016review/BDPP_Project/Data/LSTMMultiClass_trained.pt')
net.net = net.net.to(torch.device('cpu'))
net.net.device = torch.device('cpu')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div([
    html.Label('Text Input'),
    dcc.Textarea(
        value='Your text here ...',
        id='textField',
        style={'width': '100%', 'height': 300}
    ),
    html.Div(id='nWords', style={'float': 'right'}),
    html.Div(id='unrecWarn', style={'float': 'right', 'color': 'red'}),
    html.Button('Submit', id='submitButton', n_clicks=0),
    html.Div([
        html.H4(id='predParty', style={'textAlign': 'center'})
    ]),
    html.Div([
        dcc.Graph(id='probBars')
    ])
])


@app.callback(
    Output('nWords', 'children'),
    Output('nWords', 'style'),
    Output('submitButton', 'disabled'),
    Input('textField', 'value'),
    Input('nWords', 'style')
)
def count_words(text, styleDic):
    nWds = len(text.split())
    if nWds > 500:
        retStr = u'Too many words: {}. Needs to be 500 or less'.format(
            nWds)
        styleDic['color'] = 'red'
        btnDisable = True
    else:
        retStr = u'{}/500'.format(nWds)
        styleDic['color'] = 'black'
        btnDisable = False

    return retStr, styleDic, btnDisable


@app.callback(
    Output('probBars', 'figure'),
    Output('predParty', 'children'),
    Output('unrecWarn', 'children'),
    Input('submitButton', 'n_clicks'),
    State('textField', 'value')
)
def update_figure(n_clicks, text):
    unrecStr = ''
    if n_clicks > 0:
        prt, probs, nUnrec = predictParty(text)
        if nUnrec > 0:
            unrecStr = u'There are {} unrecognized words in your text. The prediction might not be very accurate.'.format(
                nUnrec)
        df = pd.DataFrame(
            {'party': list(party_dict.keys()), 'probs': probs.detach().numpy().flatten()})
        prtStr = 'Your text is most similar to speeches of a {} member'.format(
            list(party_dict.keys())[prt.detach().numpy().flatten()[0]])
    else:
        df = pd.DataFrame({'party': list(party_dict.keys()),
                           'probs': np.zeros(len(list(party_dict.keys())))})
        prtStr = 'Submit your input to get a prediction'
    bars = {'x': df['party'], 'y': df['probs'],
            'type': 'bar', 'name': 'party_probs'}
    layout = {'title': 'Probability per Party'}

    return {'data': [bars], 'layout': layout}, prtStr, unrecStr


if __name__ == "__main__":
    app.run_server(port=8265)
