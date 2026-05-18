import pandas as pd
import plotly.graph_objects as go
import numpy as np
import random
import os
from instances.instance import Instance
from dash import Dash, dcc, html, Input, Output

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def apply_rotation(w, d, h, rot):
    if rot == 0: return w, d, h
    elif rot == 1: return d, w, h
    elif rot == 2: return h, d, w
    elif rot == 3: return d, h, w
    elif rot == 4: return w, h, d
    elif rot == 5: return h, w, d
    else: raise ValueError("Invalid rotation")

def create_box(x, y, z, dx, dy, dz, color, name):
    vertices = np.array([
        [x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],
        [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]
    ])
    faces = [[0,1,2], [0,2,3], [4,5,6], [4,6,7], [0,1,5], [0,5,4],
             [2,3,7], [2,7,6], [1,2,6], [1,6,5], [0,3,7], [0,7,4]]
    i, j, k = zip(*faces)
    return go.Mesh3d(x=vertices[:,0], y=vertices[:,1], z=vertices[:,2],
                     i=i, j=j, k=k, color=color, opacity=0.5, name=name)

def create_container_wireframe(width, depth, height):
    x = [0, depth, depth, 0, 0, 0, depth, depth, 0]
    y = [0, 0, width, width, 0, 0, 0, width, width]
    z = [0, 0, 0, 0, 0, height, height, height, height]
    edges = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]
    traces = []
    for e in edges:
        traces.append(go.Scatter3d(
            x=[x[e[0]], x[e[1]]], y=[y[e[0]], y[e[1]]], z=[z[e[0]], z[e[1]]],
            mode='lines', line=dict(color='black', width=3), showlegend=False
        ))
    return traces

# ---------------------------
# CARICAMENTO DATI E CALCOLO TOTALI
# ---------------------------
dataset_name = 'DatasetA'
solver_name = 'solver_338874_2' 

try:
    inst = Instance(dataset_name)
    items = inst.df_items
    vehicles = inst.df_vehicles
    solution = pd.read_csv(os.path.join('results', f'sol_{dataset_name}_{solver_name}.csv'))
    
    veicoli_usati = sorted(solution["idx_vehicle"].unique())
    
    # --- CALCOLO TOTALI SOLUZIONE ---
    summary_vehicles = solution.groupby("idx_vehicle")["type_vehicle"].first()
    costo_totale = sum(vehicles.loc[t, "cost"] for t in summary_vehicles)
    max_value_totale = sum(vehicles.loc[t, "maxValue"] for t in summary_vehicles)
    
    # Valore totale degli oggetti (somma colonna 'value' degli items presenti in solution)
    valore_oggetti_totale = items.loc[solution["id_item"], "value"].sum()
    
except Exception as e:
    print(f"Errore: {e}")
    veicoli_usati = []
    costo_totale = 0
    max_value_totale = 0
    valore_oggetti_totale = 0

# ---------------------------
# SETUP DASH
# ---------------------------
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Visualizzatore Bin Packing 3D", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    
    # HEADER CON STATISTICHE TOTALI
    html.Div([
        html.Div([
            html.B("Costo Totale Flotta: "), html.Span(f"{costo_totale:.2f} €"),
            html.Span(" | ", style={'margin': '0 15px'}),
            html.B("Valore Totale Merce: "), html.Span(f"{valore_oggetti_totale:.2f}"),
            html.Br(),
            html.B("Capacità Max Value Flotta: "), html.Span(f"{max_value_totale:.2f}")
        ], style={
            'textAlign': 'center', 'backgroundColor': '#f0f4f7', 
            'padding': '15px', 'borderRadius': '10px', 'border': '1px solid #bdc3c7',
            'display': 'inline-block', 'fontFamily': 'Arial'
        })
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),

    html.Div([
        html.Label("Seleziona Veicolo: ", style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='vehicle-dropdown',
            options=[{'label': f'Veicolo {v}', 'value': v} for v in veicoli_usati],
            value=veicoli_usati[0] if veicoli_usati else None,
            clearable=False,
            style={'width': '300px'}
        )
    ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'gap': '15px', 'marginBottom': '10px'}),
    
    dcc.Graph(id='3d-plot', style={'height': '70vh'})
])

# ---------------------------
# CALLBACK
# ---------------------------
@app.callback(
    Output('3d-plot', 'figure'),
    Input('vehicle-dropdown', 'value')
)
def update_graph(idx_vehicle):
    if idx_vehicle is None or solution.empty:
        return go.Figure()

    df_sol = solution[solution["idx_vehicle"] == idx_vehicle]
    vehicle_type = df_sol.iloc[0]["type_vehicle"]
    vehicle = vehicles.loc[vehicle_type]
    
    v_w, v_d, v_h = vehicle["width"], vehicle["depth"], vehicle["height"]
    v_max_val = vehicle["maxValue"]
    
    # Valore oggetti in questo specifico veicolo
    valore_oggetti_veicolo = items.loc[df_sol["id_item"], "value"].sum()
    
    fig = go.Figure()
    vol_occupato = 0
    peso_totale = 0
    colors = {}

    for trace in create_container_wireframe(v_w, v_d, v_h):
        fig.add_trace(trace)

    for _, row in df_sol.iterrows():
        item_id = row["id_item"]
        item = items.loc[item_id]
        w, d, h = item["width"], item["depth"], item["height"]
        vol_occupato += (w * d * h)
        peso_totale += item["weight"]
        
        w_rot, d_rot, h_rot = apply_rotation(w, d, h, int(row["orient"]))
        
        if item_id not in colors:
            colors[item_id] = f"rgb({random.randint(50,200)},{random.randint(50,200)},{random.randint(50,200)})"

        fig.add_trace(create_box(row["x_origin"], row["y_origin"], row["z_origin"], 
                                 d_rot, w_rot, h_rot, colors[item_id], item_id))
        
    percent_riempimento = (vol_occupato / (v_w * v_d * v_h)) * 100
    
    info_text = (f"<b>VEICOLO {idx_vehicle}</b> | Valore Merce: <b>{valore_oggetti_veicolo:.2f}</b> / Max: {v_max_val}<br>"
                 f"Scatole: {len(df_sol)} | Riempimento: {percent_riempimento:.1f}% | "
                 f"Peso: {peso_totale:.1f}/{vehicle['maxWeight']} kg")

    fig.update_layout(
        scene=dict(xaxis_title="Depth (X)", yaxis_title="Width (Y)", zaxis_title="Height (Z)", aspectmode='data'),
        title=dict(text=info_text, font=dict(size=14)),
        margin=dict(l=0, r=0, b=0, t=60)
    )
    return fig

if __name__ == '__main__':
    app.run(debug=False)