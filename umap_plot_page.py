import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, callback, ctx
import plotly.graph_objects as go
import plotly.express as px

# ---------- 数据加载 ----------
print("Loading parameters...")
params = pd.read_csv('./data/training_positive_parameters_umap_n_neighbors=20, min_dist=0.1, n_components=2, metric=\'euclidean\'.csv')
print("Loading flux data...")
flux_data = np.load('./data/training_positive_id_flux_mapping.npy', allow_pickle=True).item()

# 归一化所有光谱（若未提前归一化）
print("Normalizing spectra...")
for sid in flux_data.keys():
    flux = flux_data[sid]
    min_val = np.min(flux)
    max_val = np.max(flux)
    if max_val - min_val > 0:
        flux_data[sid] = (flux - min_val) / (max_val - min_val)

wavelength = np.arange(4000, 10001, 10)

# 分子带位置
CN1, CN2 = 7732.905, 8949.855
C2_1, C2_2 = 4823.455, 5271.080
troughs = [7165.865, 8107.805, 9366.820, 4622.345, 5053.195, 5465.995]

# ---------- 构建散点图 ----------
scatter_fig = go.Figure()
scatter_fig.add_trace(go.Scattergl(
    x=params['umap0'],
    y=params['umap1'],
    mode='markers',
    marker=dict(
        color=params['bp_rp'],
        colorscale='Viridis',          # 可换成自定义三段色 ['blue','green','red']
        size=3,
        opacity=0.8,
        colorbar=dict(title='BP-RP')
    ),
    customdata=np.stack([
        params['source_id'],
        params['bp_rp'],
        params.get('teff_gspphot', np.nan),
        params.get('logg_gspphot', np.nan),
        params.get('mh_gspphot', np.nan)
    ], axis=-1),
    hovertemplate=(
        'Source ID: %{customdata[0]}<br>'
        'BP-RP: %{customdata[1]:.2f}<br>'
        'Teff: %{customdata[2]:.0f}<br>'
        'logg: %{customdata[3]:.2f}<br>'
        '[M/H]: %{customdata[4]:.2f}<extra></extra>'
    ),
    selected=dict(marker=dict(color='red')),
    unselected=dict(marker=dict(opacity=0.3))
))
scatter_fig.update_layout(
    title='UMAP Projection of Stellar Spectra',
    xaxis_title='UMAP Dimension 1',
    yaxis_title='UMAP Dimension 2',
    width=900,
    height=600,
    clickmode='event+select'
)

# ---------- 辅助函数：生成光谱图 ----------
def create_spectrum_figure(spectrum, title_text):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wavelength,
        y=spectrum,
        mode='lines',
        line=dict(color='black', width=2),
        name='Spectrum'
    ))
    # 添加分子带垂直线
    for pos, color in [(CN1, 'red'), (CN2, 'red'), (C2_1, 'red'), (C2_2, 'red')]:
        fig.add_vline(x=pos, line_width=1, line_dash='dash', line_color=color)
    for pos in troughs:
        fig.add_vline(x=pos, line_width=1, line_dash='dash', line_color='blue')
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14)),
        xaxis_title='Wavelength (Å)',
        yaxis_title='Normalized Flux',
        xaxis=dict(range=[4000, 10000]),
        yaxis=dict(range=[-0.05, 1.05]),
        width=800,
        height=400,
        showlegend=False
    )
    return fig

def empty_spectrum_figure():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[4000,10000], y=[0,0], mode='lines', line=dict(color='lightgray')))
    fig.update_layout(
        title='Click a point or drag a box to view spectrum',
        xaxis_title='Wavelength (Å)',
        yaxis_title='Normalized Flux',
        xaxis=dict(range=[4000, 10000]),
        yaxis=dict(range=[-0.05, 1.05]),
        width=800,
        height=400
    )
    return fig

# ---------- Dash 应用 ----------
app = dash.Dash(__name__)
server = app.server   # 用于 gunicorn

app.layout = html.Div([
    html.H1('Interactive UMAP Spectral Browser', style={'textAlign': 'center'}),
    html.Div([
        html.Div([
            dcc.Graph(id='scatter-plot', figure=scatter_fig)
        ], style={'width': '55%', 'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            dcc.Graph(id='spectrum-graph', figure=empty_spectrum_figure()),
            html.Div(id='info-text', style={
                'margin-top': '10px',
                'padding': '10px',
                'border': '1px solid #ddd',
                'border-radius': '5px',
                'backgroundColor': '#f9f9f9',
                'font-family': 'monospace',
                'white-space': 'pre-wrap'
            })
        ], style={'width': '40%', 'display': 'inline-block', 'margin-left': '2%'})
    ])
])

# ---------- 回调：处理点击和框选 ----------
@callback(
    Output('spectrum-graph', 'figure'),
    Output('info-text', 'children'),
    Input('scatter-plot', 'clickData'),
    Input('scatter-plot', 'selectedData')
)
def update_spectrum(clickData, selectedData):
    triggered_id = ctx.triggered_id

    # 1. 点击事件
    if triggered_id == 'scatter-plot.clickData' and clickData:
        point = clickData['points'][0]
        custom = point['customdata']
        source_id = custom[0]
        bp_rp = custom[1]
        teff = custom[2] if not pd.isna(custom[2]) else None
        logg = custom[3] if not pd.isna(custom[3]) else None
        mh = custom[4] if not pd.isna(custom[4]) else None

        spectrum = flux_data.get(source_id)
        if spectrum is None:
            return empty_spectrum_figure(), "Spectrum not found."

        # 构建信息文本
        info = f"Source ID: {source_id}\nBP-RP: {bp_rp:.2f}"
        if teff is not None:
            info += f"\nTeff: {teff:.0f}"
        if logg is not None:
            info += f"\nlogg: {logg:.2f}"
        if mh is not None:
            info += f"\n[M/H]: {mh:.2f}"

        fig = create_spectrum_figure(spectrum, info)
        return fig, info

    # 2. 框选事件
    elif triggered_id == 'scatter-plot.selectedData' and selectedData and len(selectedData['points']) > 0:
        points = selectedData['points']
        indices = [p['pointIndex'] for p in points]

        spectra_list = []
        bp_rp_list = []
        for idx in indices:
            sid = params.iloc[idx]['source_id']
            spec = flux_data.get(sid)
            if spec is not None:
                spectra_list.append(spec)
                bp_rp_list.append(params.iloc[idx]['bp_rp'])

        if len(spectra_list) == 0:
            return empty_spectrum_figure(), "No spectral data in selection."

        avg_spectrum = np.mean(spectra_list, axis=0)
        avg_bp_rp = np.mean(bp_rp_list)
        info = f"Average spectrum (N={len(indices)})\nBP-RP: {avg_bp_rp:.2f}"
        fig = create_spectrum_figure(avg_spectrum, info)
        return fig, info

    # 3. 无选择
    else:
        return empty_spectrum_figure(), "Click a point or drag a box to view spectrum."

if __name__ == '__main__':
    app.run(debug=True)