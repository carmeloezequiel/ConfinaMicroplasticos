import dash
from dash import dcc
from dash import html
import dash_leaflet as dl
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
import json
import base64
import io
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import torchvision
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms import v2 as T
import numpy as np
from io import StringIO


if torch.cuda.is_available():
    model = torch.load('model_scripted2.pb')
    device = torch.device('cuda')
else:
    model = torch.load('model_scripted2.pb', map_location=torch.device('cpu'))
    device = torch.device('cpu')

model.eval()

# Define customs transformations for data
def get_transform(train):
    transforms = []
    # Train data, randomly rotae and Flip images
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomRotation((0, 180)))
    # Ensure scaled data type
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

eval_transform = get_transform(train=False)

class Data(object):
    def __init__(self, img = None, lat=None, lng=None, pred_labels=None):
        self.img = img
        self.lat = lat
        self.lng = lng
        self.pred_labels = pred_labels

Gdata = Data()

color_options = {'Negro':'grey', 'Blanco':'white', 'Rojo':'red', 'Azul':'blue', 
                         'Amarillo':'yellow', 'Verde':'green', 'Otros':'purple'}

# Initialize the Dash app with a Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MORPH])

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("MicroFinder: Detecta Microplásticos fácil y rápido."), className="text-center my-4")
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Markdown("""
                **¡Bienvenido a MicroFinder! Sé el protagonista en la lucha contra los microplásticos. 
                Te invitamos a subir una foto del agua de tu zona junto con tus coordenadas. Juntos averiguaremos si el agua contiene microplásticos. ¡Vamos a cuidar nuestro planeta!**
            """), className="mb-4"
        )
    ]),dbc.Row([dbc.Col(dbc.Placeholder(size='xs', color='primary', style={'width':'100%'}))],
           align='start'),
    dbc.Row([
        dbc.Col(
                dcc.Upload(
                    id='upload-image',
                    children=dbc.Button('Subir Imagen', id='upload-button', color="primary", className='me-1'),
                    multiple=False
                )
        )
    ]),
    dbc.Row([
        dbc.Col(
            html.Div(id='output-image-upload', className="mt-4"), align='center'
        ),
        dbc.Col(
            dl.Map(style={'width': '100%', 'height': '50vh'}, children=[
                dl.TileLayer(), dl.LocateControl(locateOptions={'enableHighAccuracy': True})
            ], id="map", center=[50, 10], zoom=6),
            className="my-4"
        )
    ]),
    dbc.Row(
        dbc.Col(html.Div(id="coordinate-click-id"))
    ),
    dbc.Row([dbc.Col(dbc.Placeholder(size='xs', color='primary', style={'width':'100%'}))],
           align='start'),
    dbc.Row(dbc.Col(html.Br())),
    dbc.Row(
        dbc.Col(html.H2('Resultado del Análisis'))
    ),
    dbc.Row(
      dbc.Col(dbc.Spinner(size="sm", children=html.Div(id='spinner-1'), color='primary'))
    ),
    dbc.Row(dbc.Col(html.Br())),
    dbc.Row([dbc.Col(
                 dbc.Spinner(size="sm", children=dcc.Dropdown(
                                                            id='spinner-2',
                                                            placeholder="Select an object",
                                                        )
                 , color="primary")
            ),
            dbc.Col(dbc.Spinner(size="sm", children=dcc.Dropdown(
                id='spinner-3',
                options=[{'label': color, 'value':color_options[color]} for color in color_options.keys()],
                placeholder="Select a color",
            ), color="secondary"))
    ]),
    dbc.Row(dbc.Col(html.Br())),
    dbc.Row([
        dbc.Col(html.H2('Histograma de Frecuencias.'), align='end'),
        dbc.Col(dbc.Button('Comparar Marcadores', id='b2', n_clicks=0), align='start')
    ]),
    dbc.Row(dbc.Col(html.Br())),
    dbc.Row(
        dbc.Col(dcc.Graph(id='hist'))
    ),
    dbc.Row(dbc.Col(dcc.Store(id='intermediate-value'))),
    dbc.Row(dbc.Col(html.Br())),
    dbc.Row(
        dbc.Col([
            dbc.Button('Descargar CSV', id='button_csv', n_clicks=0),
            dcc.Download(id="download-csv")
        ])
    )
])

def plot_image_with_boxes(image, boxes, colors):
    fig = go.Figure()

    # Add the image to the figure
    img_width, img_height = image.size
    fig.add_trace(go.Image(z=image))

    # Add boxes with selected colors
    for box, color in zip(boxes, colors):
        fig.add_shape(
            type="rect",
            x0=box["box"][0].item(),
            y0=box["box"][1].item(),
            x1=box["box"][2].item(),
            y1=box["box"][3].item(),
            line=dict(color=color.lower(), width=3)
        )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.update_layout(
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
            range=[0, img_width]
        ),
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            range=[img_height, 0]
        )
    )

    return fig


# Define a callback to handle image upload and object labeling
@app.callback(
    Output('spinner-1', 'children'),
    Output('spinner-2', 'options'),
    Input('upload-image', 'contents'),
    prevent_initial_call=True
)
def update_output(contents):
    if contents is not None:
        # Decode the base64 string
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Use PIL to open the image
        image = Image.open(io.BytesIO(decoded))

        # Transform image to tensor
        imgTensor = pil_to_tensor(image)
        # Scale tensor image
        t = torch.tensor(np.array(imgTensor) / 255).float()
        # Transform tensor
        t = eval_transform(t)
        # Prepare and send to device
        t = t[:3, ...].to(device)
        # Make prediction
        predictions = model([t, ])
        pred = predictions[0]

        # Prepare image
        #image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
        #image = image[:3, ...]

        # Make labels of detected objects
        #pred_labels = [{'label': f'Obj {n+1}', 'box': box, 'color':'purple'} for n, label, box in zip(range(len(pred['labels'])),
                                                                                   # pred["labels"],
                                                                                   # pred['boxes'].long())]
        pred_labels = [{'label': f'Obj {n+1}', 
                        'box': box, 
                        'color':'purple'} for n, label, box, score in zip(range(len(pred['labels'])),
                                                                                    pred["labels"],
                                                                                    pred['boxes'].long(),
                                                                                    pred['scores']) 
                                                               if score.item() > 0.09]           

        Gdata.pred_labels = pred_labels
        # Create the initial plot with red boxes
        fig = plot_image_with_boxes(image, pred_labels, [pl['color'] for pl in pred_labels])

        #Create options for the object dropdown
        object_options = [{'label': obj['label'], 'value': i} for i, obj in enumerate(pred_labels)]
        
        # Display the image and color selectors
        return dcc.Graph(figure=fig, id='image-graph'), object_options
    
    return None, None, None

@app.callback(
    Output('image-graph', 'figure'),
    [Input('spinner-3', 'value')],
    [State('spinner-2', 'value')],
    prevent_initial_call=True)
def update_labels(selected_color, selected_object):
    if selected_color is not None and selected_object is not None:
        # Use PIL to open the image
        image = Gdata.img
        
        # Default colors for the boxes
        Gdata.pred_labels[selected_object]['color'] = selected_color
        
        # Create the plot with the selected colors
        fig = plot_image_with_boxes(image, Gdata.pred_labels, [pl['color'] for pl in Gdata.pred_labels])
        
        return fig
    
    return None



@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents')
)
def update_output(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        # Use PIL to open the image
        image = Image.open(io.BytesIO(decoded))
        Gdata.img = image
        return html.Img(src=contents, style={'width': 'auto', 'height': '50vh'})
    return None

@app.callback(Output("coordinate-click-id", 'children'),
              [Input("map", 'clickData')])
def click_coord(e):
    if e is not None:
        Gdata.lng = e['latlng']['lng']
        Gdata.lat = e['latlng']['lat']
        return dbc.Alert("Tus coordenadas fueron guardadas", color="primary")
    else:
        return "-"

"""
@app.callback(Output('hist', 'figure'),
              Output('intermediate-value', 'data'),
              Input('b2', 'n_clicks'),
              prevent_initial_call=True)
def show_plot(n):
    if n == 0:
        raise PreventUpdate
    data = pd.DataFrame(Gdata.pred_labels)
    fig = px.histogram(data, x='color')
    return fig, data['color'].value_counts().to_json(orient='records')

@app.callback(Output("download-csv", "data"),
          Input("button_csv", "n_clicks"),
          Input("intermediate-value", 'data'),
          prevent_initial_call=True,
)
def func(n_clicks, data):
    if n_clicks > 0:
        print(data)
        return dcc.send_data_frame(df.to_csv, f'count_data_{Gdata.lat}-{Gdata.lng}.csv')
"""
#####

@app.callback(Output('hist', 'figure'),
              Output('intermediate-value', 'data'),
              Input('b2', 'n_clicks'),
              prevent_initial_call=True)
def show_plot(n):
    if n == 0:
        raise PreventUpdate
    data = pd.DataFrame(Gdata.pred_labels)
    fig = px.histogram(data, x='color')
    return fig, data['color'].value_counts().to_json(orient='index')

@app.callback(Output("download-csv", "data"),
          Input("button_csv", "n_clicks"),
          Input("intermediate-value", 'data'),
          prevent_initial_call=True,
)
def func(n_clicks, data):
    if n_clicks > 0:
        df = pd.read_json(StringIO(data), orient='index')
        df.columns = ['count']
        df.index.name = 'colors'
        return dcc.send_data_frame(df.to_csv, f'count_data_({Gdata.lat})-({Gdata.lng}).csv')

# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)