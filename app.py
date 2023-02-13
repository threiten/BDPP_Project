import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import os

import transformers
import datasets
import torch
from utils import party_dict


party_dict['unknown'] = 6
del party_dict['independent']
del party_dict['']

def predictParty(text, pipeline):
    inp = pipeline.tokenizer(text, return_tensors='pt', truncation=True)
    logits = pipeline.model(**inp)['logits']
    softm = torch.log_softmax(logits, dim=1)
    party = list(party_dict.keys())[torch.max(softm, dim=1).indices.detach().numpy()[0]]
    probs = torch.exp(softm).detach().numpy().flatten()

    return party, probs

if os.path.exists('./Bundestag-v2'):
    test_dataset = datasets.load_dataset('./Bundestag-v2', split='test', ignore_verifications=True)
else:
    test_dataset = datasets.load_dataset('threite/Bundestag-v2', split='test')
inpDf = test_dataset.to_pandas()
if os.path.exists('./xlm-roberta-base-finetuned-partypredictor-test'):
    pipeline = transformers.pipeline('text-classification', model='./xlm-roberta-base-finetuned-partypredictor-test', tokenizer='./xlm-roberta-base-finetuned-partypredictor-test', local_files_only=True)
else:
    pipeline = transformers.pipeline(model='threite/xlm-roberta-base-finetuned-partypredictor-test')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div([
    html.Div([
        html.Label('Select party'),
        dcc.Dropdown(
            id='partyDropdown',
            options=[{'label': key, 'value': key}
                     for key in party_dict.keys()],
            value='SPD',
            style={'width': '50%'}
        ),
        html.Button('Use random speech', id='randSpeechButton',
                    n_clicks=0, className='input-group-addon')
    ], className='input-group'),
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
        dcc.Loading(
            id='loading-probBars',
            type='graph',
            children=[
                html.Div([
                    html.H4(id='predParty', style={'textAlign': 'center'})
                ]),
                html.Div([
                    dcc.Graph(id='probBars'),
                ])
            ]
        )
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
    if nWds > 364:
        retStr = u'Too many words: {}. Needs to be 364 or less'.format(
            nWds)
        styleDic['color'] = 'red'
        btnDisable = True
    else:
        retStr = u'{}/364'.format(nWds)
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
def update_figure(n_clicks, text, pipeline=pipeline):
    unrecStr = ''
    if n_clicks > 0:
        party, probs = predictParty(text, pipeline)
        # if nUnrec > 0:
        #     unrecStr = u'There are {} unrecognized words in your text. The prediction might not be very accurate.'.format(
        #         nUnrec)
        df = pd.DataFrame(
            {'party': list(party_dict.keys()), 'probs': probs})
        prtStr = 'Your text is most similar to speeches of a {} member'.format(party)
    else:
        df = pd.DataFrame({'party': list(party_dict.keys()),
                           'probs': np.zeros(len(list(party_dict.keys())))})
        prtStr = 'Submit your input to get a prediction'
    bars = {'x': df['party'], 'y': df['probs'],
            'type': 'bar', 'name': 'party_probs'}
    layout = {'title': 'Probability per Party'}

    return {'data': [bars], 'layout': layout}, prtStr, unrecStr


@app.callback(
    Output('textField', 'value'),
    Input('randSpeechButton', 'n_clicks'),
    State('textField', 'value'),
    State('partyDropdown', 'value')
)
def selectRandSpeech(n_clicks_rs, text, party):
    retStr = text
    if n_clicks_rs > 0:
        text = inpDf.loc[inpDf['party'] == party,
                         'text'].sample(replace=True).values[0]
        text = text.split()[:364]
        while len(text) < 250:
            text = inpDf.loc[inpDf['party'] == party,
                             'text'].sample(replace=True).values[0]
            text = text.split()[:364]
        retStr = ''
        for stt in text:
            retStr += '{} '.format(stt)
        retStr = retStr[:-1]

    return retStr


if __name__ == "__main__":
    app.run_server(port=8265)
