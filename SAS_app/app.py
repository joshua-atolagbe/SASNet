"""
app.py — Interactive Seismic Attribute Explorer (Plotly Dash)
=============================================================
Run with:
    python app.py

Then open http://127.0.0.1:8050 in your browser.

Place this file alongside your attr_fixed/ folder, or update ATTR_PATH below.

Requirements:
    pip install dash dash-bootstrap-components plotly numpy pillow scikit-image scipy scikit-learn
"""

import sys, base64, io, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc

# ── path ──────────────────────────────────────────────────────────────────────
ATTR_PATH = Path("./attributes")
sys.path.insert(0, str(ATTR_PATH))

from scripts.attri import attrComp
from scripts.mask  import extMask, auto_threshold

# ── constants ─────────────────────────────────────────────────────────────────
ATTRIBUTES = {
    "enve":        "Envelope",
    "inphase":     "Inst. Phase",
    "cosphase":    "Cos. Inst. Phase",
    "infreq":      "Inst. Frequency",
    "inband":      "Inst. Bandwidth",
    "domfreq":     "Dom. Frequency",
    "sweetness":   "Sweetness",
    "ampcontrast": "Amp. Contrast",
    "ampacc":      "Amp. Acceleration",
    "apolar":      "Apparent Polarity",
    "resamp":      "Response Amplitude",
    "resfreq":     "Response Frequency",
    "resphase":    "Response Phase",
    "rms":         "RMS",
    "reflin":      "Reflection Intensity",
    "fder":        "First Derivative",
    "sder":        "Second Derivative",
    "timegain":    "Time Gain",
    "gradmag":     "Gradient Magnitude",
}

KERNELS = {
    "None":      None,
    "(1,1,3)":   (1, 1, 3),
    "(1,1,9)":   (1, 1, 9),
    "(3,3,1)":   (3, 3, 1),
    "(1,1,1)":   (1, 1, 1),
    "(10,9,1)":  (10, 9, 1),
}

NOISE_OPTS    = ["gaussian", "median", "convolution"]
COLORSCALES   = ["jet", "plasma", "inferno", "viridis", "RdBu", "seismic", "hot", "turbo"]
THRESH_METHODS = ["auto (otsu)", "auto (gmm)", "auto (triangle)", "manual"]

EMPTY_FIG = go.Figure().update_layout(
    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
    font_color="#4b5563",
    xaxis=dict(visible=False), yaxis=dict(visible=False),
    annotations=[dict(text="Upload an image to begin", x=0.5, y=0.5,
                      xref="paper", yref="paper", showarrow=False,
                      font=dict(size=14, color="#4b5563"))],
    margin=dict(l=0, r=0, t=0, b=0),
)

# ── helpers ───────────────────────────────────────────────────────────────────

def decode_upload(contents: str) -> np.ndarray:
    """Decode base64 upload → (H, W, 1) uint8 array."""
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    img = Image.open(io.BytesIO(decoded)).convert("L")
    arr = np.array(img)[:, :, np.newaxis]
    return arr


def make_imshow(arr2d, colorscale="gray", title="", zmin=None, zmax=None):
    """Return a compact Plotly imshow figure."""
    fig = px.imshow(
        arr2d,
        color_continuous_scale=colorscale,
        zmin=zmin, zmax=zmax,
        aspect="equal",
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#94a3b8"), x=0.5),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        margin=dict(l=4, r=4, t=36, b=4),
        coloraxis_showscale=True,
        coloraxis_colorbar=dict(
            thickness=10, len=0.8,
            tickfont=dict(size=9, color="#94a3b8"),
            outlinecolor="#1e293b",
        ),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )
    return fig


def compute_all(img_arr, attri_type, kernel_key, noise, thresh_method, manual_thresh):
    """Run attribute pipeline and return (ori, attr, mask, used_thresh, auto_results)."""
    kernel = KERNELS[kernel_key]
    ori, noise_red, attr = attrComp(
        data=img_arr,
        attri_type=attri_type,
        kernel=kernel,
        noise=noise,
    )
    auto_results = auto_threshold(attr, method="all")

    if thresh_method == "auto (otsu)":
        thresh = auto_results["otsu"]
    elif thresh_method == "auto (gmm)":
        thresh = auto_results["gmm"]
    elif thresh_method == "auto (triangle)":
        thresh = auto_results["triangle"]
    else:
        thresh = manual_thresh

    mask = extMask(attr, threshold=thresh)
    return ori, noise_red, attr, mask, thresh, auto_results


# ── layout ────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap",
    ],
    title="Seismic Attribute Explorer",
)



# ── style helpers ─────────────────────────────────────────────────────────────
def _label_style():
    return {
        "fontFamily": "Space Mono, monospace",
        "fontSize": "9px",
        "letterSpacing": "0.15em",
        "color": "#475569",
        "textTransform": "uppercase",
        "marginBottom": "4px",
        "marginTop": "14px",
    }

def _dd_style():
    return {
        "background": "#0f172a",
        "color": "#e2e8f0",
        "border": "1px solid #334155",
        "borderRadius": "6px",
        "fontSize": "12px",
    }


SIDEBAR = dbc.Col(
    [
        # ── logo / title ──────────────────────────────────────────────────────
        html.Div([
            html.Div("SEISMIC", style={
                "fontFamily": "Space Mono, monospace",
                "fontSize": "11px", "letterSpacing": "0.25em",
                "color": "#38bdf8", "marginBottom": "2px",
            }),
            html.Div("ATTRIBUTE EXPLORER", style={
                "fontFamily": "Space Mono, monospace",
                "fontSize": "17px", "fontWeight": "700",
                "color": "#f1f5f9", "letterSpacing": "-0.01em",
            }),
        ], style={"marginBottom": "24px", "paddingBottom": "16px",
                  "borderBottom": "1px solid #1e293b"}),

        # ── upload ────────────────────────────────────────────────────────────
        html.Div("INPUT IMAGE", style=_label_style()),
        dcc.Upload(
            id="upload",
            children=html.Div([
                html.Div("⬆", style={"fontSize": "28px", "color": "#38bdf8"}),
                html.Div("Drop PNG / JPEG here", style={"fontSize": "12px", "color": "#94a3b8"}),
                html.Div("or click to browse", style={"fontSize": "11px", "color": "#475569"}),
            ], style={"textAlign": "center", "padding": "20px 0"}),
            style={
                "border": "1px dashed #334155",
                "borderRadius": "8px",
                "background": "#0f172a",
                "cursor": "pointer",
                "marginBottom": "20px",
            },
            multiple=False,
        ),
        html.Div(id="upload-status", style={"fontSize": "11px", "color": "#64748b", "marginBottom": "16px"}),

        # ── attribute ─────────────────────────────────────────────────────────
        html.Div("ATTRIBUTE", style=_label_style()),
        dcc.Dropdown(
            id="dd-attri",
            options=[{"label": v, "value": k} for k, v in ATTRIBUTES.items()],
            value="enve",
            clearable=False,
            style=_dd_style(),
        ),

        # ── noise ─────────────────────────────────────────────────────────────
        html.Div("NOISE REDUCTION", style=_label_style()),
        dcc.Dropdown(
            id="dd-noise",
            options=[{"label": n.capitalize(), "value": n} for n in NOISE_OPTS],
            value="gaussian",
            clearable=False,
            style=_dd_style(),
        ),

        # ── kernel ────────────────────────────────────────────────────────────
        html.Div("KERNEL", style=_label_style()),
        dcc.Dropdown(
            id="dd-kernel",
            options=[{"label": k, "value": k} for k in KERNELS],
            value="None",
            clearable=False,
            style=_dd_style(),
        ),

        # ── colorscale ────────────────────────────────────────────────────────
        html.Div("ATTRIBUTE COLORSCALE", style=_label_style()),
        dcc.Dropdown(
            id="dd-cmap",
            options=[{"label": c, "value": c} for c in COLORSCALES],
            value="jet",
            clearable=False,
            style=_dd_style(),
        ),

        # ── threshold ─────────────────────────────────────────────────────────
        html.Div("THRESHOLD METHOD", style=_label_style()),
        dcc.Dropdown(
            id="dd-thresh-method",
            options=[{"label": t, "value": t} for t in THRESH_METHODS],
            value="auto (otsu)",
            clearable=False,
            style=_dd_style(),
        ),

        html.Div(id="manual-thresh-container", children=[
            html.Div("MANUAL THRESHOLD", style=_label_style()),
            dcc.Slider(
                id="slider-thresh",
                min=0, max=1, step=0.01, value=0.5,
                marks={0: "0", 0.5: "0.5", 1: "1"},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ], style={"display": "none"}),

        # ── vmin / vmax ───────────────────────────────────────────────────────
        html.Div("ATTRIBUTE DISPLAY RANGE  (vmin / vmax)", style=_label_style()),
        dcc.RangeSlider(
            id="slider-vrange",
            min=0, max=1, step=0.01, value=[0.0, 1.0],
            marks={0: "0", 0.5: "0.5", 1: "1"},
            tooltip={"placement": "bottom", "always_visible": True},
            allowCross=False,
        ),

        html.Br(),

        # ── run button ────────────────────────────────────────────────────────
        dbc.Button(
            "▶  COMPUTE",
            id="btn-compute",
            color="primary",
            style={
                "width": "100%",
                "fontFamily": "Space Mono, monospace",
                "fontSize": "13px",
                "letterSpacing": "0.1em",
                "background": "#0ea5e9",
                "border": "none",
                "borderRadius": "6px",
                "padding": "10px",
                "marginBottom": "16px",
            },
        ),

        # ── auto-threshold readout ────────────────────────────────────────────
        html.Div(id="thresh-readout"),

        # ── stats ─────────────────────────────────────────────────────────────
        html.Div(id="stats-panel"),

    ],
    width=2,
    style={
        "background": "#0d1117",
        "borderRight": "1px solid #1e293b",
        "minHeight": "100vh",
        "padding": "24px 16px",
        "overflowY": "auto",
    },
)

MAIN_AREA = dbc.Col(
    [
        # ── top row: Original | Denoised ──────────────────────────────────────
        dbc.Row([
            dbc.Col(dcc.Graph(id="fig-original",  figure=EMPTY_FIG,
                              config={"displayModeBar": False}, style={"height": "340px"}), width=6),
            dbc.Col(dcc.Graph(id="fig-denoised",  figure=EMPTY_FIG,
                              config={"displayModeBar": False}, style={"height": "340px"}), width=6),
        ], style={"marginBottom": "6px"}),

        # ── bottom row: Attribute | Mask ──────────────────────────────────────
        dbc.Row([
            dbc.Col(dcc.Graph(id="fig-attribute", figure=EMPTY_FIG,
                              config={"displayModeBar": True,
                                      "modeBarButtonsToRemove": ["select2d","lasso2d","autoScale2d","hoverClosestCartesian","hoverCompareCartesian","toggleSpikelines"]},
                              style={"height": "340px"}), width=6),
            dbc.Col(dcc.Graph(id="fig-mask",      figure=EMPTY_FIG,
                              config={"displayModeBar": False}, style={"height": "340px"}), width=6),
        ]),

        # ── histogram strip ───────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(dcc.Graph(id="fig-histogram", figure=EMPTY_FIG,
                              config={"displayModeBar": False}, style={"height": "200px"}), width=12),
        ], style={"marginTop": "6px"}),
    ],
    width=10,
    style={"padding": "16px 20px", "background": "#080c14"},
)

app.layout = dbc.Container(
    [
        dcc.Store(id="store-img"),        # raw image array as list
        dcc.Store(id="store-results"),    # computed results
        dbc.Row([SIDEBAR, MAIN_AREA], style={"margin": "0"}),
    ],
    fluid=True,
    style={"padding": "0", "background": "#080c14", "minHeight": "100vh"},
)



# ── callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("store-img", "data"),
    Output("upload-status", "children"),
    Input("upload", "contents"),
    State("upload", "filename"),
    prevent_initial_call=True,
)
def store_image(contents, filename):
    if contents is None:
        return no_update, no_update
    arr = decode_upload(contents)
    return arr.tolist(), f"✓ {filename}  {arr.shape[0]}×{arr.shape[1]}"


@app.callback(
    Output("manual-thresh-container", "style"),
    Input("dd-thresh-method", "value"),
)
def toggle_manual_slider(method):
    return {"display": "block"} if method == "manual" else {"display": "none"}


@app.callback(
    Output("fig-original",  "figure"),
    Output("fig-denoised",  "figure"),
    Output("fig-attribute", "figure"),
    Output("fig-mask",      "figure"),
    Output("fig-histogram", "figure"),
    Output("thresh-readout","children"),
    Output("stats-panel",   "children"),
    Input("btn-compute", "n_clicks"),
    State("store-img",        "data"),
    State("dd-attri",         "value"),
    State("dd-noise",         "value"),
    State("dd-kernel",        "value"),
    State("dd-cmap",          "value"),
    State("dd-thresh-method", "value"),
    State("slider-thresh",    "value"),
    State("slider-vrange",    "value"),
    prevent_initial_call=True,
)
def run_pipeline(n_clicks, img_data, attri_type, noise, kernel_key,
                 cmap, thresh_method, manual_thresh, vrange):

    if img_data is None:
        msg = dbc.Alert("Please upload an image first.", color="warning",
                        style={"fontSize": "12px"})
        return EMPTY_FIG, EMPTY_FIG, EMPTY_FIG, EMPTY_FIG, EMPTY_FIG, msg, ""

    img_arr = np.array(img_data, dtype=np.uint8)

    try:
        ori, noise_red, attr, mask, used_thresh, auto_res = compute_all(
            img_arr, attri_type, kernel_key, noise, thresh_method, manual_thresh
        )
    except Exception as e:
        err = dbc.Alert(f"Error: {e}", color="danger", style={"fontSize": "11px"})
        return EMPTY_FIG, EMPTY_FIG, EMPTY_FIG, EMPTY_FIG, EMPTY_FIG, err, ""

    attr_label = ATTRIBUTES[attri_type]

    # ── figures ───────────────────────────────────────────────────────────────
    fig_ori  = make_imshow(ori.squeeze(),        "gray",    "Original")
    fig_den  = make_imshow(noise_red.squeeze(),  "gray",    f"Denoised — {noise.capitalize()}")
    vmin_val, vmax_val = vrange if vrange else [0.0, 1.0]
    fig_attr = make_imshow(attr.squeeze(),       cmap,      f"{attr_label}  (vmin={vmin_val:.2f} vmax={vmax_val:.2f})", vmin_val, vmax_val)
    fig_mask = make_imshow(mask.squeeze(),       "gray",    f"Mask  (t = {used_thresh:.3f})")

    # add threshold line on attribute figure
    fig_attr.add_shape(
        type="line", x0=0, x1=1, y0=used_thresh * attr.shape[0],
        y1=used_thresh * attr.shape[0],
        xref="paper", line=dict(color="#f43f5e", width=1.5, dash="dot"),
    )

    # ── histogram ─────────────────────────────────────────────────────────────
    flat = attr.flatten()
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=flat, nbinsx=120,
        marker_color="#38bdf8", opacity=0.75,
        name="Attribute",
    ))
    # vertical lines for all thresholds
    thresh_colors = {"otsu": "#facc15", "gmm": "#4ade80", "triangle": "#f97316"}
    for meth, col in thresh_colors.items():
        val = auto_res[meth]
        fig_hist.add_vline(x=val, line_color=col, line_dash="dash", line_width=1.5,
                           annotation_text=f"{meth} {val:.2f}",
                           annotation_font=dict(size=9, color=col),
                           annotation_position="top right")
    if thresh_method == "manual":
        fig_hist.add_vline(x=manual_thresh, line_color="#f43f5e", line_width=2,
                           annotation_text=f"manual {manual_thresh:.2f}",
                           annotation_font=dict(size=9, color="#f43f5e"),
                           annotation_position="top left")
    fig_hist.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font_color="#94a3b8",
        margin=dict(l=40, r=20, t=24, b=36),
        xaxis=dict(title="Normalised value", color="#64748b", gridcolor="#1e293b"),
        yaxis=dict(title="Count",            color="#64748b", gridcolor="#1e293b"),
        title=dict(text="Attribute Histogram — all thresholds", font=dict(size=12), x=0.5),
        showlegend=False,
        bargap=0.02,
    )

    # ── threshold readout card ────────────────────────────────────────────────
    recommended = auto_res["recommended"]
    readout = html.Div([
        html.Div("AUTO THRESHOLDS", style=_label_style()),
        *[
            html.Div([
                html.Span(f"{m.upper()}", style={
                    "fontFamily": "Space Mono, monospace",
                    "fontSize": "9px", "color": c,
                    "display": "inline-block", "width": "68px",
                }),
                html.Span(f"{auto_res[m]:.3f}", style={
                    "fontFamily": "Space Mono, monospace",
                    "fontSize": "12px", "color": "#e2e8f0",
                }),
                html.Span("  ★" if m == recommended else "", style={"color": "#facc15", "fontSize": "10px"}),
            ], style={"marginBottom": "4px"})
            for m, c in [("otsu", "#facc15"), ("gmm", "#4ade80"), ("triangle", "#f97316")]
        ],
        html.Div(f"Using: {used_thresh:.3f}", style={
            "fontFamily": "Space Mono, monospace",
            "fontSize": "10px", "color": "#f43f5e",
            "marginTop": "6px", "paddingTop": "6px",
            "borderTop": "1px solid #1e293b",
        }),
    ])

    # ── stats card ────────────────────────────────────────────────────────────
    salt_pct  = float((mask.squeeze() == 0).mean() * 100)
    sed_pct   = 100 - salt_pct
    stats = html.Div([
        html.Div("SEGMENTATION STATS", style=_label_style()),
        _stat_row("Salt",     f"{salt_pct:.1f} %",  "#38bdf8"),
        _stat_row("Sediment", f"{sed_pct:.1f} %",   "#94a3b8"),
        _stat_row("Attr min", f"{float(attr.min()):.4f}", "#64748b"),
        _stat_row("Attr max", f"{float(attr.max()):.4f}", "#64748b"),
        _stat_row("Attr mean",f"{float(attr.mean()):.4f}","#64748b"),
    ])

    return fig_ori, fig_den, fig_attr, fig_mask, fig_hist, readout, stats


def _stat_row(label, value, color="#94a3b8"):
    return html.Div([
        html.Span(label, style={
            "fontSize": "9px", "fontFamily": "Space Mono, monospace",
            "color": "#475569", "display": "inline-block", "width": "70px",
            "textTransform": "uppercase", "letterSpacing": "0.08em",
        }),
        html.Span(value, style={
            "fontSize": "12px", "fontFamily": "Space Mono, monospace",
            "color": color, "fontWeight": "600",
        }),
    ], style={"marginBottom": "3px"})


# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8050)
