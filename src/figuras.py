"""Figuras compartidas por el dashboard (app.py) y el Jupyter Book.

Cada función recibe artefactos ya calculados (data/) y devuelve una figura de Plotly. Al usar las
mismas funciones en ambos productos, lo que se interpreta en el libro es exactamente lo que se ve
en el dashboard.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from modelo_utils import ETIQUETAS

DATA = Path(__file__).resolve().parents[1] / "data"

# ---------------------------------------------------------------- estilo
AZUL, MORADO, NARANJA, GRIS, ROJO, VERDE = "#1c7ed6", "#7048e8", "#f08c00", "#868e96", "#e03131", "#2f9e44"
COLOR_MODELO = {"Regresión logística": AZUL, "SVM lineal": MORADO, "Persistencia": NARANJA,
                "Dummy (estratificado)": GRIS, "Dummy (clase mayoritaria)": "#ced4da"}
COLOR_CLASE = {0: "#adb5bd", 1: ROJO}
COLOR_LISA = {"Alto-Alto": "#c92a2a", "Bajo-Bajo": "#1864ab", "Alto-Bajo": "#ff8787",
              "Bajo-Alto": "#74c0fc", "No significativo": "#dee2e6"}
EVENTOS = [("2020-03-23", "2020-05-31", "Confinamiento"), ("2021-04-28", "2021-06-15", "Paro nacional")]

pio.templates["sipsa"] = go.layout.Template(layout=go.Layout(
    font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=12, color="#212529"),
    paper_bgcolor="white", plot_bgcolor="white",
    xaxis=dict(gridcolor="#eef0f2", zerolinecolor="#dee2e6", linecolor="#ced4da", automargin=True),
    yaxis=dict(gridcolor="#eef0f2", zerolinecolor="#dee2e6", linecolor="#ced4da", automargin=True),
    colorway=[AZUL, MORADO, NARANJA, VERDE, ROJO, GRIS], hoverlabel=dict(font_size=12),
    margin=dict(l=60, r=20, t=92, b=50),
    title=dict(x=0.01, xanchor="left", y=0.985, yanchor="top", yref="container", font=dict(size=15)),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, font=dict(size=11)),
))
pio.templates.default = "sipsa"


LEYENDA_ABAJO = dict(orientation="h", yanchor="top", y=-0.3, x=0)


def etiqueta_variable(v):
    """Nombre legible de una columna del modelo (incluye one-hot y splines)."""
    m = re.match(r"^(ratio_t(?:_1)?)_sp_(\d+)$", v)
    if m:
        return f"{ETIQUETAS[m.group(1)]} · spline {int(m.group(2)) + 1}"
    for pref, txt in [("mercado_", "Central: "), ("depto_origen_", "Depto.: "), ("grupo_dom_", "Grupo: ")]:
        if v.startswith(pref):
            return txt + v[len(pref):].replace("infrequent_sklearn", "otras (infrecuentes)")
    return ETIQUETAS.get(v, v)


def nombre_propio(texto):
    """'VALLE DEL CAUCA' -> 'Valle del Cauca' (minúsculas para artículos y preposiciones)."""
    menores = {"de", "del", "la", "las", "los", "y", "el"}
    palabras = str(texto).lower().split()
    return " ".join(w if (i > 0 and w in menores) else w[:1].upper() + w[1:] for i, w in enumerate(palabras))


def _fmt_pct(x):
    return f"{100 * x:.1f} %".replace(".", ",")


# ---------------------------------------------------------------- datos
@lru_cache(maxsize=None)
def resumen():
    return json.loads((DATA / "resumen.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def tabla(nombre):
    return pd.read_parquet(DATA / f"{nombre}.parquet")


@lru_cache(maxsize=None)
def geojson():
    return json.loads((DATA / "municipios.geojson").read_text())


def metricas_umbral(y, p, t, score=None):
    """Métricas de clasificación para un umbral dado."""
    yhat = (p >= t).astype(int)
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    tp = int(((yhat == 1) & (y == 1)).sum())
    return {"accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat), "F1": f1_score(y, yhat),
            "AUC-ROC": roc_auc_score(y, p if score is None else score),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp, "alertas": int(yhat.sum())}


# ---------------------------------------------------------------- 1. contexto
def fig_corredor_ejemplo(tipo="Irregular"):
    ej = tabla("corredor_ejemplo")
    ej = ej[ej["tipo"] == tipo]
    info = next(e for e in resumen()["kpis"]["ejemplos"] if e["tipo"] == tipo)
    fig = go.Figure()
    fig.add_scatter(x=ej["semana_t1"], y=ej["tasa_t1"] / 1e3, name="Abastecimiento en t+1 (t/día)",
                    line=dict(color=AZUL, width=1.6))
    fig.add_scatter(x=ej["semana_t1"], y=ej["umbral"] / 1e3, name="Umbral: 50 % del promedio de 8 semanas",
                    line=dict(color=GRIS, dash="dash"))
    i = ej[ej["interrupcion_t1"] == 1]
    fig.add_scatter(x=i["semana_t1"], y=i["tasa_t1"] / 1e3, mode="markers", name="Interrupción (y = 1)",
                    marker=dict(color=ROJO, size=9, symbol="x"))
    fig.update_layout(title=f"Corredor {tipo.lower()}: {nombre_propio(info['origen'])} → {info['central']} "
                            f"({_fmt_pct(info['tasa_interrupcion'])} de semanas con interrupción)",
                      yaxis_title="toneladas por día de encuesta", height=370)
    return fig


def fig_mapa_origenes():
    mun = tabla("municipios").copy()
    mun["tipo"] = np.where(mun["pdet"] == 1, "Municipio PDET", "Otro municipio")
    mun["t_dia"] = mun["toneladas_dia"].round(1)
    fig = px.scatter_map(mun, lat="lat", lon="lon", size=np.sqrt(mun["toneladas_dia"].clip(lower=0.01)),
                         color="tipo", color_discrete_map={"Municipio PDET": NARANJA, "Otro municipio": AZUL},
                         hover_name="nombre", hover_data={"depto": True, "t_dia": True, "corredores": True,
                                                          "lat": False, "lon": False, "tipo": False},
                         size_max=22, zoom=4.6, center={"lat": 6.2, "lon": -74.2}, height=540,
                         labels={"t_dia": "t/día", "corredores": "corredores", "depto": "departamento"})
    fig.update_layout(map_style="carto-positron", title="Municipios de origen de los corredores (tamaño = toneladas por día)",
                      margin=dict(l=0, r=0, t=50, b=0), legend_title_text="",
                      legend=dict(y=0.02, x=0.01, yanchor="bottom", orientation="v", bgcolor="rgba(255,255,255,0.85)"))
    return fig


def fig_toneladas_mercado(anio=2024, top=15):
    t = tabla("toneladas_mercado_anio")
    t = t[t["anio"] == anio].nlargest(top, "toneladas").sort_values("toneladas")
    fig = px.bar(t, x=t["toneladas"] / 1e3, y="mercado", orientation="h", height=440,
                 labels={"x": "miles de toneladas", "mercado": ""}, color_discrete_sequence=[AZUL])
    fig.update_xaxes(nticks=4, tickangle=0)
    fig.update_layout(title=f"Toneladas por central en {anio} (top {top})", margin=dict(t=60))
    return fig


# ---------------------------------------------------------------- 2. EDA
def fig_serie(dimension="total", valores=None, suavizado=13):
    s = tabla("serie_semanal")
    s = s[s["dimension"] == dimension]
    if valores:
        s = s[s["valor"].isin(valores)]
    fig = go.Figure()
    for v, g in s.groupby("valor"):
        g = g.sort_values("semana_t1")
        tasa = g["k"] / g["n"]
        if dimension == "total":
            fig.add_scatter(x=g["semana_t1"], y=tasa, name="Semanal", line=dict(color="#ffa8a8", width=1))
            fig.add_scatter(x=g["semana_t1"], y=tasa.rolling(suavizado, center=True).mean(),
                            name=f"Media móvil {suavizado} sem.", line=dict(color=ROJO, width=2.2))
        else:
            fig.add_scatter(x=g["semana_t1"], y=tasa.rolling(suavizado, center=True, min_periods=3).mean(),
                            name=str(v)[:28], mode="lines")
    for a, b, txt in EVENTOS:
        fig.add_vrect(x0=a, x1=b, fillcolor="#ffe066", opacity=0.35, line_width=0,
                      annotation_text=txt, annotation_position="top left", annotation_font_size=10)
    fig.update_layout(title="Tasa semanal de interrupción (entrenamiento)", yaxis_tickformat=".0%",
                      yaxis_title="corredores con interrupción", height=400)
    return fig


def fig_estacionalidad():
    s = tabla("serie_semanal")
    s = s[s["dimension"] == "total"].copy()
    s["tasa"] = s["k"] / s["n"]
    s["mes"] = s["semana_t1"].dt.month
    fest = pd.DataFrame(resumen()["eda"]["festivos"])
    fig = make_subplots(rows=1, cols=2, column_widths=[0.65, 0.35],
                        subplot_titles=("Tasa semanal por mes de la semana objetivo", "Tasa según festivos en t+1"))
    meses = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
    for m in range(1, 13):
        fig.add_box(y=s.loc[s["mes"] == m, "tasa"], name=meses[m - 1], marker_color="#ff8787", showlegend=False,
                    row=1, col=1)
    fig.add_bar(x=fest["festivos_t1"].astype(str), y=fest["mean"], marker_color=ROJO, showlegend=False,
                text=[_fmt_pct(v) for v in fest["mean"]], textposition="outside", row=1, col=2)
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", range=[0, fest["mean"].max() * 1.25], row=1, col=2)
    fig.update_xaxes(title_text="número de festivos", row=1, col=2)
    fig.update_layout(height=380, margin=dict(t=60))
    return fig


def fig_distribucion(variable, escala_log=False):
    m = tabla("muestra_eda")
    x = m[variable]
    if escala_log:
        x = np.log1p(x.clip(lower=0))
    df = pd.DataFrame({"x": x, "clase": m["interrupcion_t1"].map({0: "Sin interrupción", 1: "Interrupción"})})
    fig = px.histogram(df, x="x", color="clase", nbins=60, histnorm="probability density", barmode="overlay",
                       marginal="box", opacity=0.6, height=420,
                       color_discrete_map={"Sin interrupción": "#adb5bd", "Interrupción": ROJO},
                       labels={"x": ("log(1 + x) de " if escala_log else "") + ETIQUETAS.get(variable, variable)})
    fig.update_layout(title="Distribución por clase (muestra de entrenamiento)", legend_title_text="", yaxis_title="densidad")
    return fig


def fig_deciles(variable):
    d = tabla("deciles")
    d = d[d["variable"] == variable]
    prev = resumen()["kpis"]["tasa_train"]
    fig = go.Figure()
    fig.add_scatter(x=d["x"], y=d["tasa"], mode="lines+markers", line=dict(color=ROJO, width=2),
                    marker=dict(size=8), customdata=np.c_[d["decil"], d["n"]],
                    hovertemplate="decil %{customdata[0]}<br>valor medio %{x:.3f}<br>tasa %{y:.1%}<br>n = %{customdata[1]:,}<extra></extra>",
                    name="Tasa por decil")
    fig.add_hline(y=prev, line_dash="dot", line_color=GRIS, annotation_text=f"tasa de entrenamiento {_fmt_pct(prev)}")
    fig.update_layout(title="Tasa de interrupción por deciles",
                      xaxis_title="valor medio del decil", yaxis_title="tasa de interrupción en t+1",
                      yaxis_tickformat=".0%", height=400, showlegend=False)
    return fig


def fig_tasa_categoria(variable="grupo_dom"):
    t = tabla("tasa_por_categoria")
    t = t[t["variable"] == variable].sort_values("tasa")
    if variable == "pdet_origen":
        t["categoria"] = t["categoria"].map({"0": "No PDET", "1": "PDET"})
    fig = px.bar(t, x="tasa", y="categoria", orientation="h", text=[_fmt_pct(v) for v in t["tasa"]],
                 hover_data={"n": ":,"}, color_discrete_sequence=[ROJO], height=max(320, 22 * len(t) + 120),
                 labels={"tasa": "tasa de interrupción", "categoria": ""})
    fig.add_vline(x=resumen()["kpis"]["tasa_train"], line_dash="dot", line_color=GRIS)
    fig.update_layout(title=f"Tasa de interrupción por «{ETIQUETAS.get(variable, variable)}»", xaxis_tickformat=".0%")
    return fig


def fig_correlacion():
    c = tabla("correlacion_spearman")
    c.index = c.columns
    fig = px.imshow(c, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", height=620,
                    labels=dict(color="ρ de Spearman"))
    fig.update_layout(title="Correlación de Spearman entre predictoras numéricas")
    fig.update_xaxes(tickfont_size=9)
    fig.update_yaxes(tickfont_size=9)
    return fig


def fig_asociacion(medida="auc_univariado"):
    a = tabla("asociacion_objetivo").copy()
    if medida == "rbc":
        a = a[a["rbc"].notna()].assign(valor=lambda d: d["rbc"].abs())
        titulo, eje = "Tamaño del efecto frente al objetivo (|correlación biserial de rangos|)", "|rbc|"
    elif medida == "info_mutua":
        a = a[a["info_mutua"].notna()].assign(valor=lambda d: d["info_mutua"])
        titulo, eje = "Información mutua con el objetivo (relaciones no lineales)", "información mutua (nats)"
    else:
        a = a.assign(valor=a["auc_univariado"])
        titulo, eje = "Auditoría de fuga: AUC univariado de cada variable", "AUC univariado"
    a = a.sort_values("valor")
    color = np.where(a["tipo"].str.startswith("control"), ROJO, np.where(a["tipo"] == "auxiliar", GRIS, AZUL))
    fig = go.Figure(go.Bar(x=a["valor"], y=[ETIQUETAS.get(v, v) for v in a["variable"]], orientation="h",
                           marker_color=color, customdata=a["tipo"],
                           hovertemplate="%{y}<br>%{x:.3f}<br>%{customdata}<extra></extra>"))
    if medida == "auc_univariado":
        fig.add_vline(x=0.9, line_dash="dash", line_color=ROJO, annotation_text="alerta de fuga (0,9)")
        fig.update_xaxes(range=[0.45, 1.02])
    fig.update_layout(title=titulo, xaxis_title=eje, height=760, margin=dict(l=10))
    return fig


def fig_mapa_tasa(capa="tasa"):
    mun = tabla("municipios").copy()
    mun = mun[mun["n"] >= 52]
    mun["tasa_pct"] = (100 * mun["tasa"]).round(1)
    comun = dict(geojson=geojson(), locations="cod_mpio", featureidkey="properties.cod", hover_name="nombre",
                 center={"lat": 6.2, "lon": -74.2}, zoom=4.6, opacity=0.85, height=580)
    if capa == "lisa":
        fig = px.choropleth_map(mun, color="lisa", color_discrete_map=COLOR_LISA,
                                hover_data={"depto": True, "tasa_pct": True, "cod_mpio": False}, **comun)
        titulo = "Clusters LISA de la tasa de interrupción (k = 8 vecinos, p < 0,05)"
    elif capa == "gi":
        fig = px.choropleth_map(mun, color="gi", color_continuous_scale="RdBu_r", range_color=(-4, 4),
                                hover_data={"depto": True, "tasa_pct": True, "cod_mpio": False}, **comun)
        titulo = "Hotspots Getis-Ord Gi* (|z| > 1,96 es significativo)"
    elif capa == "cv8":
        fig = px.choropleth_map(mun, color="cv8", color_continuous_scale="Purples",
                                hover_data={"depto": True, "tasa_pct": True, "cod_mpio": False}, **comun)
        titulo = "Variabilidad media de los corredores (cv8) por municipio de origen"
    else:
        fig = px.choropleth_map(mun, color="tasa", color_continuous_scale="Reds", range_color=(0.1, 0.6),
                                hover_data={"depto": True, "tasa_pct": True, "n": ":,", "cod_mpio": False}, **comun)
        titulo = "Tasa de interrupción por municipio de origen (≥ 52 corredor-semanas)"
    fig.update_layout(map_style="carto-positron", title=titulo, margin=dict(l=0, r=0, t=50, b=0), legend_title_text="",
                      legend=dict(y=0.02, x=0.01, yanchor="bottom", bgcolor="rgba(255,255,255,0.85)", orientation="v"))
    return fig


def fig_stl():
    s = tabla("stl_tasa")
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=("Observado", "Tendencia", "Estacionalidad anual", "Residuo"))
    for i, c in enumerate(["observado", "tendencia", "estacional", "residuo"], start=1):
        fig.add_scatter(x=s["semana_t1"], y=s[c], line=dict(color=ROJO if i == 1 else AZUL, width=1.3),
                        showlegend=False, row=i, col=1)
    fig.update_layout(title="Descomposición STL de la tasa semanal de interrupción (período 52)", height=620)
    return fig


def fig_acf():
    t = resumen()["eda"]["temporal"]
    banda = t["banda_conf"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ACF de la tasa semanal", "PACF de la tasa semanal"))
    for j, (vals, n) in enumerate([(t["acf"], 60), (t["pacf"], 30)], start=1):
        fig.add_bar(x=list(range(len(vals))), y=vals, marker_color=AZUL, showlegend=False, row=1, col=j)
        fig.add_hrect(y0=-banda, y1=banda, fillcolor="#a5d8ff", opacity=0.35, line_width=0, row=1, col=j)
    fig.update_xaxes(title_text="rezago (semanas)")
    fig.update_layout(height=340, margin=dict(t=60))
    return fig


def fig_autocorr_corredor():
    a = pd.DataFrame(resumen()["eda"]["temporal"]["autocorr_corredor"])
    fig = px.bar(a, x=a["rezago"].astype(str), y="correlacion", text=a["correlacion"].round(2),
                 color_discrete_sequence=[AZUL], height=320,
                 labels={"x": "rezago (semanas)", "correlacion": "corr(y_t, y_t-k)"})
    fig.update_layout(title="Autocorrelación de la interrupción dentro de cada corredor")
    return fig


def fig_moran_dispersion():
    mun = tabla("municipios").dropna(subset=["z"])
    I = resumen()["eda"]["espacial"]["moran_k8"]
    fig = px.scatter(mun, x="z", y="rezago", color="lisa", color_discrete_map=COLOR_LISA, hover_name="nombre",
                     opacity=0.7, height=420, labels={"z": "tasa estandarizada", "rezago": "rezago espacial (k = 8)"})
    b = np.polyfit(mun["z"], mun["rezago"], 1)
    xx = np.linspace(mun["z"].min(), mun["z"].max(), 10)
    fig.add_scatter(x=xx, y=np.polyval(b, xx), mode="lines", line=dict(color="black"), name=f"pendiente = I = {I:.3f}")
    fig.add_hline(y=0, line_color="#adb5bd")
    fig.add_vline(x=0, line_color="#adb5bd")
    fig.update_layout(title="Diagrama de dispersión de Moran")
    return fig


def fig_semivariograma():
    e = resumen()["eda"]["espacial"]
    s = pd.DataFrame(e["semivariograma"])
    fig = px.line(s, x="distancia", y="semivarianza", markers=True, height=340, color_discrete_sequence=[AZUL],
                  labels={"distancia": "distancia (km)"})
    fig.add_hline(y=e["varianza_total"], line_dash="dash", line_color=GRIS, annotation_text="varianza total")
    fig.update_layout(title="Semivariograma empírico de la tasa municipal")
    return fig


def fig_pca():
    ind, car = tabla("pca_individuos"), tabla("pca_cargas")
    v = resumen()["eda"]["pca"]["varianza"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Varianza explicada", "Individuos (PC1-PC2)", "Cargas"))
    fig.add_bar(x=list(range(1, len(v) + 1)), y=v, marker_color=AZUL, showlegend=False, row=1, col=1)
    fig.add_scatter(x=list(range(1, len(v) + 1)), y=np.cumsum(v), mode="lines+markers", line=dict(color="black"),
                    showlegend=False, row=1, col=1)
    for k, nombre in [(0, "Sin interrupción"), (1, "Interrupción")]:
        d = ind[ind["y"] == k]
        fig.add_scattergl(x=d["pc1"], y=d["pc2"], mode="markers", marker=dict(size=3, color=COLOR_CLASE[k], opacity=0.35),
                          name=nombre, row=1, col=2)
    for _, r in car.iterrows():
        fig.add_scatter(x=[0, r["PC1"]], y=[0, r["PC2"]], mode="lines", line=dict(color=GRIS), showlegend=False, row=1, col=3)
        if np.hypot(r["PC1"], r["PC2"]) > 0.25:
            fig.add_annotation(x=r["PC1"], y=r["PC2"], text=r["variable"], showarrow=False, font_size=9, row=1, col=3)
    fig.update_layout(height=460, title="Análisis de componentes principales (exploratorio)", legend=LEYENDA_ABAJO, margin=dict(t=80, b=100))
    return fig


# ---------------------------------------------------------------- 3. modelos
def fig_roc_pr(modelos):
    c = tabla("curvas")
    prev = resumen()["kpis"]["tasa_test"]
    m = tabla("metricas_test")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Curva ROC", "Curva precisión-recall"))
    for nombre in modelos:
        auc = m[(m["modelo"] == nombre) & (m["metrica"] == "AUC-ROC")]["valor"].iloc[0]
        ap = m[(m["modelo"] == nombre) & (m["metrica"] == "AUC-PR")]["valor"].iloc[0]
        r = c[(c["modelo"] == nombre) & (c["curva"] == "ROC")]
        fig.add_scatter(x=r["x"], y=r["y"], name=f"{nombre} (AUC {auc:.3f})".replace(".", ","),
                        line=dict(color=COLOR_MODELO[nombre], width=2.2), legendgroup=nombre, row=1, col=1)
        p = c[(c["modelo"] == nombre) & (c["curva"] == "PR")]
        fig.add_scatter(x=p["x"], y=p["y"], name=f"AP {ap:.3f}".replace(".", ","), line=dict(color=COLOR_MODELO[nombre], width=2.2),
                        legendgroup=nombre, showlegend=False, row=1, col=2)
    fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(color="black", dash="dot", width=1), showlegend=False, row=1, col=1)
    fig.add_hline(y=prev, line_dash="dot", line_color="black", row=1, col=2,
                  annotation_text=f"prevalencia {_fmt_pct(prev)}", annotation_position="bottom right")
    fig.update_xaxes(title_text="tasa de falsos positivos", row=1, col=1)
    fig.update_yaxes(title_text="tasa de verdaderos positivos", row=1, col=1)
    fig.update_xaxes(title_text="recall", row=1, col=2)
    fig.update_yaxes(title_text="precisión", row=1, col=2)
    fig.update_layout(height=470, title="Desempeño en prueba (2025)", legend=LEYENDA_ABAJO, margin=dict(t=70, b=120))
    return fig


def fig_matriz(mt, titulo="Matriz de confusión"):
    z = [[mt["tn"], mt["fp"]], [mt["fn"], mt["tp"]]]
    total = sum(sum(f) for f in z)
    txt = [[f"{v:,}<br>({100 * v / total:.1f} %)".replace(",", ".") for v in fila] for fila in z]
    fig = go.Figure(go.Heatmap(z=z, x=["Predicho: no", "Predicho: sí"], y=["Real: no", "Real: sí"], text=txt,
                               texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=titulo, height=340, margin=dict(l=80, r=20, t=50, b=40))
    return fig


def fig_calibracion():
    c = tabla("curvas")
    c = c[c["curva"] == "Calibración"]
    pred = tabla("predicciones_test")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_histogram(x=pred["p_logistica"], nbinsx=40, marker_color="#e9ecef", name="distribución p (logística)",
                      secondary_y=True)
    for nombre, g in c.groupby("modelo"):
        fig.add_scatter(x=g["x"], y=g["y"], mode="lines+markers", name=nombre, line=dict(color=COLOR_MODELO[nombre]))
    fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(color="black", dash="dot", width=1), name="calibración perfecta")
    fig.update_xaxes(title_text="probabilidad predicha", range=[0, 1])
    fig.update_yaxes(title_text="frecuencia observada", range=[0, 1], secondary_y=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, secondary_y=True)
    fig.update_layout(title="Calibración de las probabilidades (prueba)", height=440, legend=LEYENDA_ABAJO, margin=dict(t=60, b=120))
    return fig


def fig_coeficientes(modelo="Regresión logística", top=20):
    archivo = "coeficientes_logistica" if modelo.startswith("Reg") else "coeficientes_svm"
    c = tabla(archivo)
    c = c.reindex(c["coeficiente"].abs().sort_values(ascending=False).index).head(top).iloc[::-1]
    etiquetas = [etiqueta_variable(v) for v in c["variable"]]
    fig = go.Figure(go.Bar(x=c["coeficiente"], y=etiquetas, orientation="h",
                           marker_color=np.where(c["coeficiente"] > 0, ROJO, AZUL),
                           hovertemplate="%{y}<br>coeficiente %{x:.3f}<extra></extra>"))
    fig.add_vline(x=0, line_color="black", line_width=1)
    unidad = ("log-odds" if modelo.startswith("Reg") else "peso del hiperplano") + \
        " (numéricas: por desviación estándar; categóricas y binarias: frente a la base)"
    fig.update_layout(title=f"{modelo}: {top} coeficientes de mayor magnitud", xaxis_title=unidad,
                      height=max(380, 24 * top + 120), margin=dict(l=10))
    return fig


def fig_cv(modelo="Regresión logística"):
    cv = tabla("cv_resultados")
    cv = cv[cv["modelo"] == modelo]
    g = cv.groupby("C").agg(valid=("auc_valid", "mean"), sd=("auc_valid", "std"), train=("auc_train", "mean")).reset_index()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("AUC medio según C", "AUC por pliegue (C elegido)"),
                        horizontal_spacing=0.14)
    etiquetas_c = [f"{c:g}".replace(".", ",") for c in g["C"]]
    fig.add_scatter(x=etiquetas_c, y=g["train"], name="entrenamiento", mode="lines+markers", line=dict(color=GRIS), row=1, col=1)
    fig.add_scatter(x=etiquetas_c, y=g["valid"], name="validación (media ± d.e.)", mode="lines+markers",
                    line=dict(color=COLOR_MODELO[modelo]), error_y=dict(type="data", array=g["sd"]), row=1, col=1)
    C = resumen()["modelos"][modelo]["C"]
    f = cv[cv["C"] == C]
    fig.add_bar(x=f["pliegue"].astype(str), y=f["auc_valid"], marker_color=COLOR_MODELO[modelo], showlegend=False,
                text=f["auc_valid"].round(3), textposition="outside", row=1, col=2)
    fig.update_xaxes(type="category", title_text="C", row=1, col=1)
    fig.update_xaxes(title_text="pliegue", row=1, col=2)
    fig.update_yaxes(range=[f["auc_valid"].min() - 0.02, f["auc_valid"].max() + 0.015], row=1, col=2)
    fig.update_layout(title=f"{modelo}: validación por bloques temporales", height=420, legend=LEYENDA_ABAJO,
                      margin=dict(t=80, b=110))
    return fig


def fig_cv_espacial():
    e = tabla("cv_espacial")
    fig = px.bar(e, x=e["pliegue"].astype(str), y="auc", color="modelo", barmode="group",
                 color_discrete_map=COLOR_MODELO, hover_data={"departamentos": True, "n": ":,"}, height=360,
                 labels={"x": "pliegue (grupo de departamentos no vistos)", "auc": "AUC-ROC"})
    fig.update_yaxes(range=[e["auc"].min() - 0.03, e["auc"].max() + 0.02])
    fig.update_layout(title="Departamentos no vistos (GroupKFold)", legend_title_text="", legend=LEYENDA_ABAJO,
                      margin=dict(t=60, b=120))
    return fig


def fig_curva_aprendizaje():
    lc = tabla("curva_aprendizaje")
    fig = go.Figure()
    for nombre, g in lc.groupby("modelo"):
        fig.add_scatter(x=g["filas"], y=g["auc_train"], name=f"{nombre}: entrenamiento", mode="lines+markers",
                        line=dict(color=COLOR_MODELO[nombre], dash="dot"))
        fig.add_scatter(x=g["filas"], y=g["auc_valid"], name=f"{nombre}: validación", mode="lines+markers",
                        line=dict(color=COLOR_MODELO[nombre]))
    fig.update_xaxes(type="log", title_text="filas de entrenamiento (escala log)")
    fig.update_yaxes(title_text="AUC-ROC")
    fig.update_layout(title="Curva de aprendizaje (validación: último pliegue)", height=460, legend=LEYENDA_ABAJO, margin=dict(t=60, b=150))
    return fig


def fig_umbral(modelo="Regresión logística"):
    r = resumen()["modelos"][modelo]
    cu = r["curva_umbral"]
    fig = go.Figure(go.Scatter(x=cu["umbral"], y=cu["f1"], line=dict(color=COLOR_MODELO[modelo])))
    fig.add_vline(x=r["umbral"], line_dash="dash", line_color=ROJO,
                  annotation_text=f"umbral elegido {r['umbral']:.2f}".replace(".", ","))
    fig.update_layout(title=f"{modelo}: F1 fuera de pliegue según el umbral (entrenamiento)",
                      xaxis_title="umbral de probabilidad", yaxis_title="F1", height=340)
    return fig


def fig_dependencia():
    d = tabla("dependencia_ratio_t")
    fig = px.line(d, x="ratio_t", y="prob", color="modelo", color_discrete_map=COLOR_MODELO, height=360,
                  labels={"ratio_t": "abastecimiento en t / promedio de 8 semanas", "prob": "probabilidad media de interrupción"})
    fig.update_layout(title="Dependencia parcial de ratio_t", yaxis_tickformat=".0%", legend_title_text="", legend=LEYENDA_ABAJO,
                      margin=dict(t=60, b=120))
    return fig


def fig_subgrupos(dimension="region_origen", modelo="Regresión logística"):
    pred = tabla("predicciones_test")
    col = "p_logistica" if modelo.startswith("Reg") else "p_svm"
    filas = []
    for v, g in pred.groupby(dimension):
        if len(g) < 300 or g["y"].nunique() < 2:
            continue
        filas.append({"grupo": str(v), "auc": roc_auc_score(g["y"], g[col]), "n": len(g), "prevalencia": g["y"].mean()})
    t = pd.DataFrame(filas).sort_values("auc")
    if dimension == "pdet_origen":
        t["grupo"] = t["grupo"].map({"0": "No PDET", "1": "PDET"})
    fig = px.bar(t, x="auc", y="grupo", orientation="h", text=t["auc"].round(3), color_discrete_sequence=[COLOR_MODELO[modelo]],
                 hover_data={"n": ":,", "prevalencia": ":.1%"}, height=max(320, 22 * len(t) + 120),
                 labels={"auc": "AUC-ROC en prueba", "grupo": ""})
    fig.update_xaxes(range=[max(0.5, t["auc"].min() - 0.05), min(1, t["auc"].max() + 0.04)])
    fig.update_layout(title=f"{modelo}: AUC por subgrupo (prueba 2025)", margin=dict(t=60))
    return fig


def fig_residuos():
    rs, rm = tabla("residuos_semana"), tabla("residuos_municipio")
    r = resumen()["modelos"]["residuos"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Residuo de Pearson medio por semana", "ACF del residuo semanal",
                                                        f"Residuo medio por municipio (I de Moran = {r['moran_residuo']:.3f})"),
                        specs=[[{}, {}, {"type": "scattergeo"}]])
    fig.add_scatter(x=rs["semana_t1"], y=rs["residuo"], line=dict(color="#495057"), showlegend=False, row=1, col=1)
    fig.add_bar(x=list(range(len(r["acf_residuo"]))), y=r["acf_residuo"], marker_color=AZUL, showlegend=False, row=1, col=2)
    fig.add_hrect(y0=-r["banda"], y1=r["banda"], fillcolor="#a5d8ff", opacity=0.35, line_width=0, row=1, col=2)
    fig.add_scattergeo(lat=rm["lat"], lon=rm["lon"], mode="markers", showlegend=False,
                       marker=dict(color=rm["r"].clip(-1, 1), colorscale="RdBu_r", cmin=-1, cmax=1, size=6,
                                   colorbar=dict(title="residuo", len=0.7)), row=1, col=3)
    fig.update_geos(fitbounds="locations", showcountries=True, countrycolor="#adb5bd", showland=True, landcolor="#f8f9fa")
    fig.update_layout(height=420, title="Diagnóstico de residuos de la regresión logística (prueba)", margin=dict(t=80))
    return fig
