import dash
from dash import dcc, html
import dash_leaflet as dl
import dash_bootstrap_components as dbc
from dash_bootstrap_components import icons
from dash.dependencies import Input, Output, State
import base64
import io
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from exif import Image as ExifImage
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from dash.exceptions import PreventUpdate
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms import v2 as T
import numpy as np
from io import StringIO

# Configuración del modelo
if torch.cuda.is_available():
    model = torch.load('model_scripted2.pb')
    device = torch.device('cuda')
else:
    model = torch.load('model_scripted2.pb', map_location=torch.device('cpu'))
    device = torch.device('cpu')

model.eval()


# Definición de transformaciones
def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomRotation((0, 180)))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


eval_transform = get_transform(train=False)


class DetectedObject:
    def __init__(self, label, box):
        self.label = label
        self.box = box
        self.classification = None  # Inicialmente no clasificado


class Data:
    def __init__(self):
        self.img = None
        self.lat = None
        self.lng = None
        self.detected_objects = []


Gdata = Data()

color_options = {'Negro': 'grey', 'Blanco': 'white', 'Rojo': 'red', 'Azul': 'blue',
                 'Amarillo': 'yellow', 'Verde': 'green', 'Otros': 'purple'}


## Definir estilor:

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

custom_styles = {
    'hero-image': {
        'width': '100%',
        'height': '400px',
        'object-fit': 'cover',
        'object-position': 'center'
    },
    'hero-content': {
        'background-color': 'rgba(255, 255, 255, 0.8)',
        'padding': '20px',
        'border-radius': '5px'
    },
    'content-image': {
        'width': '100%',
        'max-width': '400px',
        'height': 'auto',
        'margin': 'auto'
    },
    'body-text': {
        'font-size': '22px'  # Aumenta el tamaño de texto base
    },
    'section-title': {
        'font-size': '28px',
        'font-weight': 'bold'
    },
    'subsection-title': {
        'font-size': '24px',
        'font-weight': 'bold'
    }
}

def create_color_guide():
    fig = go.Figure()

    for i, (name, color) in enumerate(color_options.items()):
        fig.add_trace(go.Bar(
            x=[name],
            y=[1],
            name=name,
            marker_color=color,
            text=name,
            textposition='inside',
            insidetextanchor='middle',
            hoverinfo='none'
        ))

    fig.update_layout(
        title={
            'text': 'Guía de Clasificación por Color',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=False,
        barmode='stack',
        xaxis={'title': ''},
        yaxis={'title': '', 'showticklabels': False},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig

color_guide = dbc.Card([
    dbc.CardHeader(html.H4("Guía de Clasificación por Color", className="text-center")),
    dbc.CardBody([
        dcc.Graph(
            figure=create_color_guide(),
            config={'displayModeBar': False}
        ),
        html.H5("Cómo elegir el color correcto:", className="mt-3"),
        html.Ul([
            html.Li("Observa cuidadosamente el color predominante del microplástico en la imagen."),
            html.Li("Compara este color con los de la guía de arriba."),
            html.Li("Selecciona el color más cercano al que observas."),
            html.Li("Si el color no coincide con ninguno de los principales, elige 'Otros'."),
        ]),
        dbc.Alert([
            html.Strong("Consejo: "),
            "Considera la iluminación de la imagen. Los microplásticos pueden parecer más claros u oscuros dependiendo de la luz, pero intenta identificar su color base."
        ], color="info", className="mt-3"),
        html.P([
            "Recuerda: La precisión en la selección del color es crucial para el análisis de datos. ",
            "Si no estás seguro, es mejor elegir 'Otros' que arriesgarse a una clasificación incorrecta."
        ], className="mt-3"),
    ])
])

# Inicialización de la app Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP], title='MicroFinder')

# Barra lateral actualizada
sidebar = html.Div(
    [
        html.H2("MicroFinder", className="display-4"),
        html.Hr(),
        html.P(
            "Detecta Microplásticos fácil y rápido", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Inicio", href="/", active="exact"),
                dbc.NavLink("Detector", href="/detector", active="exact"),
                dbc.NavLink("Acerca de", href="/about", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

# Contenido principal
content = html.Div(id="page-content", style=CONTENT_STYLE)

# Layout principal
app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content
])

landing_page = html.Div([
    # Sección Hero
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Img(src="/assets/heroImg.webp",
                         style=custom_styles['hero-image'],
                         alt="Microplásticos en el medio ambiente"),
            ], md=8),
            dbc.Col([
                html.Div([
                    html.H1("MicroFinder", className="display-3"),  # Aumentado de display-4 a display-3
                    html.P("Detecta microplásticos fácil y rápido. ¡Únete a la lucha contra la contaminación plástica!",
                           className="lead", style={'font-size': '24px'}),  # Tamaño de letra aumentado
                    dbc.Button("¡Empieza a Detectar!", color="primary", size="lg", href="/detector",
                               className="mt-3", style={'font-size': '22px'})  # Botón más grande
                ], style=custom_styles['hero-content'])
            ], md=4, className="d-flex align-items-center")
        ], className="py-5")
    ], fluid=True, className="px-0"),

    # Contenido principal
    dbc.Container([
        html.H2("La problemática de los microplásticos", className="text-center mb-4",
                style=custom_styles['section-title']),
        dbc.Row([
            # Columna izquierda
            dbc.Col([
                html.H2("¿Qué son los microplásticos?", style=custom_styles['subsection-title']),
                html.P("""
                    Los microplásticos son pequeñas partículas de plástico menores a 5 mm que se han convertido en una 
                    preocupación ambiental global. Estos diminutos fragmentos contaminan nuestros océanos, ríos y hasta 
                    el aire que respiramos.
                """, style=custom_styles['body-text']),
                html.Img(src="/assets/microplasticos.webp",
                         style=custom_styles['content-image'],
                         className="mb-3",
                         alt="Microplásticos de cerca"),
                html.H2("Fuentes de microplásticos", className="mt-4", style=custom_styles['subsection-title']),
                html.P("""
                    Provienen de diversas fuentes, incluyendo la degradación de plásticos más grandes, microfibras de ropa 
                    sintética, y microesferas en productos de cuidado personal.
                """, style=custom_styles['body-text']),
            ], md=6),
            # Columna derecha
            dbc.Col([
                html.H2("Impacto en el medio ambiente", style=custom_styles['subsection-title']),
                html.P("""
                    Los microplásticos afectan la vida marina y potencialmente nuestra salud. Su pequeño tamaño los hace 
                    difíciles de filtrar y pueden acumular toxinas, entrando así en la cadena alimenticia.
                """, style=custom_styles['body-text']),
                html.Img(src="/assets/microplastics2.webp",
                         style=custom_styles['content-image'],
                         className="mb-3",
                         alt="Impacto de los microplásticos"),
                html.H2("¿Por qué es importante detectarlos?", className="mt-4",
                        style=custom_styles['subsection-title']),
                html.P("""
                    La detección de microplásticos es crucial para entender la extensión de la contaminación, identificar 
                    las áreas más afectadas y desarrollar estrategias efectivas de mitigación. Cada muestra analizada nos 
                    acerca más a comprender y abordar este problema global.
                """, style=custom_styles['body-text']),
            ], md=6),
        ]),
        dbc.Row([
            dbc.Col([
                html.H2("¿Cómo puedes ayudar?", className="mt-4 mb-3", style=custom_styles['subsection-title']),
                html.P("""
                    Con MicroFinder, puedes contribuir directamente a la investigación sobre microplásticos. Al analizar 
                    muestras de agua de tu entorno, ayudas a crear un mapa global de la presencia de microplásticos y 
                    contribuyes a la ciencia ciudadana.
                """, style=custom_styles['body-text']),
                dbc.Button("Aprende más sobre cómo ayudar", color="secondary", href="/about",
                           className="mt-3", style={'font-size': '20px'})  # Botón más grande
            ], className="text-center")
        ])
    ])
], style={'font-size': '20px'})

detector_page = html.Div([
    dbc.Row([
        # Columna izquierda: Subir Imagen y Análisis de Imagen
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Subir Imagen", className="text-center")),
                dbc.CardBody([
                    dcc.Upload(
                        id='upload-image',
                        children=dbc.Button([html.I(className="bi bi-upload me-2"), "Subir Imagen"], color="primary",
                                            className='me-1'),
                        multiple=False
                    ),
                    html.Div(id='output-image-upload', className="mt-4")
                ])
            ], className="mb-4"),
            dbc.Card([
                dbc.CardHeader(html.H4("Análisis de Imagen", className="text-center")),
                dbc.CardBody([
                    dbc.Spinner(children=html.Div(id='spinner-1'), color='primary', type="grow"),
                    dbc.Row([
                        dbc.Col(dcc.Dropdown(id='spinner-2', placeholder="Selecciona un objeto"), md=6),
                        dbc.Col(dcc.Dropdown(id='spinner-3', placeholder="Selecciona un color"), md=6)
                    ], className="mt-3"),
                ])
            ], className="mb-4")
        ], md=6),

        # Columna derecha: Mapa y Guía de Color
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Mapa", className="text-center")),
                dbc.CardBody([
                    dl.Map(style={'width': '100%', 'height': '50vh'}, center=[50, 10], zoom=6, children=[
                        dl.TileLayer(), dl.LocateControl(locateOptions={'enableHighAccuracy': True})
                    ], id="map"),
                    html.Div(id="gps-info", className="mt-2"),
                    html.Div(id="coordinate-click-id", className="mt-2")
                ])
            ], className="mb-4"),
            color_guide  # Añadimos la guía de color aquí
        ], md=6)
    ]),

    # Fila para el histograma
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Distribución de Colores", className="text-center")),
                dbc.CardBody([
                    dcc.Graph(id='hist')
                ])
            ])
        ], md=12)
    ], className="mt-4"),

    # Fila para el botón de descarga
    dbc.Row([
        dbc.Col([
            dbc.Button([html.I(className="bi bi-download me-2"), "Descargar CSV"], id='button_csv', color="success",
                       className="mt-4")
        ], className="text-center")
    ]),

    dcc.Download(id="download-csv"),
    dcc.Store(id='intermediate-value')
])

# Página "Acerca de"
about_page = html.Div([
    html.H2("Acerca de MicroFinder"),
    html.P("MicroFinder es una aplicación diseñada para ayudar en la detección y clasificación de microplásticos en muestras de agua."),
    html.P("Utiliza técnicas avanzadas de visión artificial para analizar imágenes y identificar posibles partículas de microplásticos."),
    html.H3("Cómo usar MicroFinder"),
    dbc.ListGroup([
        dbc.ListGroupItem("1. Sube una imagen de tu muestra de agua."),
        dbc.ListGroupItem("2. La aplicación analizará automáticamente la imagen en busca de microplásticos."),
        dbc.ListGroupItem("3. Clasifica manualmente los objetos detectados por color."),
        dbc.ListGroupItem("4. Visualiza los resultados en el gráfico."),
        dbc.ListGroupItem("5. Descarga los datos en formato CSV para un análisis más detallado.")
    ]),
    html.P("¡Únete a la lucha contra la contaminación plástica y contribuye a la ciencia ciudadana con MicroFinder!"),
    dbc.Button("¡Empieza a Detectar!", color="primary", size="lg", href="/detector", className="mt-3"),
    html.H1("Acerca del Equipo", className="display-4"),
    html.P("Este proyecto fue desarrollado por el equipo de Ctrl+Alt+Defeat, un grupo de entusiastas de la ciencia, tecnología e innovación de alto impacto"),
    dbc.ListGroup([
        dbc.ListGroupItem("Roberth Marcano - Desarrollador de Modelo de Machine Learning"),
        dbc.ListGroupItem("Carmelo Garcia - Desarrollador de Modelo de Machine Learning"),
        dbc.ListGroupItem("Javier Salcedo - Diseñador - Project Manager"),
        dbc.ListGroupItem("Gabriel Cardona - Diseñador UX/UI - Ingeniero DevOps")
    ]),
])

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return landing_page
    elif pathname == "/detector":
        return detector_page
    elif pathname == "/about":
        return about_page
    # Si la ruta no es reconocida, muestra un mensaje de error 404
    return html.Div([
        html.H1("404: Not found", className="text-danger"),
        html.Hr(),
        html.P(f"La ruta {pathname} no fue reconocida..."),
    ])

@app.callback(
    Output("gps-info", "children"),
    [Input("map", "center")]
)
def update_gps_info(center):
    if center and center != [50, 10]:  # Asumiendo que [50, 10] es el centro por defecto
        return dbc.Alert(f"Coordenadas GPS detectadas: Latitud {center[0]:.6f}, Longitud {center[1]:.6f}", color="info")
    return ""

## Extracción de Metada de las imágenes:

def get_decimal_coordinates(info):
    for key in ['Latitude', 'Longitude']:
        if 'GPS' + key in info and 'GPS' + key + 'Ref' in info:
            e = info['GPS' + key]
            ref = info['GPS' + key + 'Ref']
            info[key] = (e[0][0] / e[0][1] +
                         e[1][0] / e[1][1] / 60 +
                         e[2][0] / e[2][1] / 3600) * (-1 if ref in ['S', 'W'] else 1)
    if 'Latitude' in info and 'Longitude' in info:
        return [info['Latitude'], info['Longitude']]


def get_geotagging(exif):
    if not exif:
        return None

    geotagging = {}
    for (idx, tag) in TAGS.items():
        if tag == 'GPSInfo':
            if idx not in exif:
                return None

            for (key, val) in GPSTAGS.items():
                if key in exif[idx]:
                    geotagging[val] = exif[idx][key]

    return geotagging


def extract_gps_info(image_file):
    try:
        image = ExifImage(image_file)
        if image.has_exif:
            try:
                return image.gps_latitude, image.gps_longitude
            except AttributeError:
                return None
    except:
        pass

    try:
        image = Image.open(image_file)
        exif = image._getexif()
        if exif:
            geotags = get_geotagging(exif)
            if geotags:
                return get_decimal_coordinates(geotags)
    except:
        pass

    return None

def process_predictions(predictions):
    pred = predictions[0]
    return [DetectedObject(f'Obj {n + 1}', box)
            for n, (label, box, score) in enumerate(zip(pred["labels"], pred['boxes'], pred['scores']))
            if score.item() > 0.09]


def plot_image_with_boxes(image, detected_objects, selected_index=None):
    fig = go.Figure()
    img_width, img_height = image.size
    fig.add_trace(go.Image(z=image))

    selected_color = 'rgb(255, 80, 0)'  # Naranja brillante para selección
    default_color = 'purple'  # Color por defecto para objetos no clasificados

    for i, obj in enumerate(detected_objects):
        is_selected = i == selected_index
        line_width = 6 if is_selected else 3

        # Color lógica
        if is_selected:
            line_color = selected_color
        elif obj.classification is None:
            line_color = default_color
        else:
            line_color = obj.classification

        opacity = 1 if is_selected else 0.7

        fig.add_shape(
            type="rect",
            x0=obj.box[0].item(),
            y0=obj.box[1].item(),
            x1=obj.box[2].item(),
            y1=obj.box[3].item(),
            line=dict(color=line_color, width=line_width),
            opacity=opacity
        )

        fig.add_annotation(
            x=obj.box[0].item(),
            y=obj.box[1].item(),
            text=str(i + 1),
            showarrow=False,
            font=dict(color="white", size=12),
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="white",
            borderwidth=1,
            borderpad=4,
            opacity=1
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


@app.callback(
    [Output('spinner-1', 'children'),
     Output('spinner-2', 'options'),
     Output('spinner-2', 'value'),
     Output('map', 'center'),
     Output('map', 'zoom')],
    [Input('upload-image', 'contents')],
    prevent_initial_call=True
)
def update_output(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded))
        Gdata.img = image

        # Extraer información GPS
        gps_info = extract_gps_info(io.BytesIO(decoded))

        imgTensor = pil_to_tensor(image)
        t = torch.tensor(np.array(imgTensor) / 255).float()
        t = eval_transform(t)
        t = t[:3, ...].to(device)
        predictions = model([t, ])
        Gdata.detected_objects = process_predictions(predictions)
        fig = plot_image_with_boxes(image, Gdata.detected_objects, selected_index=0)
        object_options = [{'label': obj.label, 'value': i} for i, obj in enumerate(Gdata.detected_objects)]

        # Actualizar el centro del mapa si se encontró información GPS
        if gps_info:
            Gdata.lat, Gdata.lng = gps_info
            return dcc.Graph(figure=fig, id='image-graph'), object_options, 0, gps_info, 15
        else:
            return dcc.Graph(figure=fig, id='image-graph'), object_options, 0, dash.no_update, dash.no_update

    return None, None, None, dash.no_update, dash.no_update

@app.callback(
    Output('image-graph', 'figure'),
    [Input('spinner-2', 'value'),
     Input('spinner-3', 'value')],
    prevent_initial_call=True
)
def update_classification(selected_object, selected_color):
    if selected_object is not None:
        if selected_color is not None:
            Gdata.detected_objects[selected_object].classification = selected_color
        fig = plot_image_with_boxes(Gdata.img, Gdata.detected_objects, selected_index=selected_object)
        return fig
    return dash.no_update


@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents')
              )
def update_output_img(contents):
    if contents is not None:
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


@app.callback(Output('hist', 'figure'),
              Output('intermediate-value', 'data'),
              Input('spinner-3', 'value'),
              prevent_initial_call=True)
def show_plot(value):
    if not Gdata.detected_objects:
        raise PreventUpdate
    data = pd.DataFrame([{'color': obj.classification or 'Otros'} for obj in Gdata.detected_objects])
    fig = px.histogram(data, x='color')
    return fig, data['color'].value_counts().to_json(orient='index')


@app.callback(
    Output("download-csv", "data"),
    [Input("button_csv", "n_clicks"),
     Input("intermediate-value", 'data')],
    prevent_initial_call=True
)
def export_csv(n_clicks, data):
    if n_clicks > 0:
        df = pd.read_json(StringIO(data), orient='index')
        df.columns = ['count']
        df.index.name = 'colors'

        # Crear un buffer de memoria para guardar el CSV
        csv_buffer = StringIO()

        # Exportar a CSV con punto y coma como separador
        df.to_csv(csv_buffer, sep=';', decimal=',')

        # Obtener el contenido del buffer
        csv_string = csv_buffer.getvalue()

        return dict(content=csv_string,
                    filename=f'count_data_({Gdata.lat})-({Gdata.lng}).csv',
                    type='text/csv',
                    base64=False)

    return dash.no_update


# Ejecutar la app
if __name__ == '__main__':
    app.run_server(debug=False)