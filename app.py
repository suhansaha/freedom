# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from datetime import datetime as dt
import dash_daq as daq


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Tabs(id="tabs", value='tab-2', children=[
        dcc.Tab(label='Backtesting', value='tab-1'),
        dcc.Tab(label='Live Trading', value='tab-2'),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Backtesting'),
            dcc.Dropdown(
                options=[
                    {'label': 'New York City', 'value': 'NYC'},
                    {'label': 'Montréal', 'value': 'MTL'},
                    {'label': 'San Francisco', 'value': 'SF'}
                ],
                multi=True,
                value='MTL'
            ),
            dcc.DatePickerRange(
                id='date-picker-range',
                start_date=dt(1997, 5, 3),
                end_date_placeholder_text='Select a date!'
            ),
            dcc.Checklist(
                options=[
                    {'label': 'New York City', 'value': 'NYC'},
                    {'label': 'Montréal', 'value': 'MTL'},
                    {'label': 'San Francisco', 'value': 'SF'}
                ],
                values=['MTL', 'SF'],
                labelStyle={'display': 'inline-block'}
            ),
            dcc.RadioItems(
                options=[
                    {'label': 'New York City', 'value': 'NYC'},
                    {'label': 'Montréal', 'value': 'MTL'},
                    {'label': 'San Francisco', 'value': 'SF'}
                ],
                value='MTL',
                labelStyle={'display': 'inline-block'}
            ),
            html.Button('Submit', id='button')
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('Live Trading'),
            dcc.Dropdown(
                options=[
                    {'label': 'New York City', 'value': 'NYC'},
                    {'label': 'Montréal', 'value': 'MTL'},
                    {'label': 'San Francisco', 'value': 'SF'}
                ],
                multi=True,
                value='MTL'
            ),
            html.Button('Submit', id='button')
        ])


if __name__ == '__main__':
    app.run_server(debug=True)