"""
Dashboard del Entregable 2 · Interrupciones en corredores de abastecimiento de alimentos (SIPSA-A).

Tres pestañas: (1) contexto del problema, (2) análisis exploratorio y (3) modelos base
(regresión logística y SVM lineal frente a líneas base triviales).

Local:   python src/app.py   ->  http://127.0.0.1:8050
Render:  cd src && gunicorn app:server   (ver render.yaml)
"""
import os

import joblib
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html

import figuras as F
from modelo_utils import CATEGORICAS, ETIQUETAS, NUMERICAS, X_COLS

LIBRO = "https://camilomolinaguerrero-sudo.github.io/ml-sipsa-dashboard/"
REPO = "https://github.com/camilomolinaguerrero-sudo/ml-sipsa-dashboard"
ENTREGABLE1 = "https://camilomolinaguerrero-sudo.github.io/ml-sipsa-abastecimiento/"

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
           suppress_callback_exceptions=True, title="Corredores de abastecimiento · SIPSA-A")
server = app.server  # requerido por Render (gunicorn app:server)

R = F.resumen()
K, EDA, MOD = R["kpis"], R["eda"], R["modelos"]
PRED = F.tabla("predicciones_test")
MET = F.tabla("metricas_test")
COMP = F.tabla("comparaciones")
MODELO = joblib.load(F.DATA / "modelo_logistico.joblib")


def num(x, dec=0):
    s = f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x, dec=1):
    return num(100 * x, dec) + " %"


def metrica(modelo, m):
    return MET[(MET["modelo"] == modelo) & (MET["metrica"] == m)].iloc[0]


def tarjeta_kpi(valor, etiqueta, icono, color="primary", alto=True):
    return dbc.Card(dbc.CardBody([
        html.Div([html.I(className=f"bi bi-{icono} me-2"), etiqueta], className="kpi-etiqueta"),
        html.Div(valor, className=f"kpi-valor text-{color}"),
    ]), className="kpi h-100" if alto else "kpi")


def lectura(*parrafos, titulo="Lectura"):
    return dbc.Alert([html.Div([html.I(className="bi bi-lightbulb me-2"), titulo], className="fw-semibold mb-1")] +
                     [dcc.Markdown(p, className="mb-1") for p in parrafos], color="light", className="lectura")


def grafico(fig_or_id, **kw):
    if isinstance(fig_or_id, str):
        return dcc.Loading(dcc.Graph(id=fig_or_id, config={"displaylogo": False}, **kw), type="dot")
    return dcc.Graph(figure=fig_or_id, config={"displaylogo": False}, **kw)


def tabla_df(df, **kw):
    return dbc.Table.from_dataframe(df, striped=True, bordered=False, hover=True, size="sm", className="tabla", **kw)


# ============================================================================ encabezado
encabezado = html.Header(dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div("Entregable 2 · Machine Learning · Doctorado en Ingeniería", className="sobretitulo"),
            html.H1("Interrupciones en corredores de abastecimiento de alimentos", className="titulo"),
            html.Div("Camilo Molina Guerrero y Lina Margarita Buelvas · Datos: SIPSA-A (DANE), 2018-2025",
                     className="autores"),
        ], md=8),
        dbc.Col(html.Div([
            dbc.Button([html.I(className="bi bi-journal-text me-1"), "Informe (Jupyter Book)"], href=LIBRO,
                       target="_blank", color="light", size="sm", className="me-2"),
            dbc.Button([html.I(className="bi bi-github me-1"), "Repositorio"], href=REPO, target="_blank",
                       color="light", outline=True, size="sm"),
        ], className="text-md-end mt-3 mt-md-2"), md=4),
    ], align="center"),
], fluid="xl"), className="encabezado")


# ============================================================================ 1. contexto
def tab_contexto():
    return dbc.Container([
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H4("El problema", className="card-title"),
                dcc.Markdown(
                    "La resiliencia alimentaria depende de que los alimentos sigan llegando a las ciudades cuando algo "
                    "falla en el camino. En Colombia, las **centrales mayoristas** concentran ese flujo: cada semana "
                    "reciben carga de cientos de municipios. Cuando un **corredor** (municipio de origen → central) "
                    "se interrumpe, la oferta de la ciudad se resiente y los productores pierden su mercado.\n\n"
                    "Este proyecto, parte de una tesis doctoral sobre gobernanza de la logística alimentaria en "
                    "territorios **PDET**, se pregunta si es posible **anticipar con una semana de antelación** qué "
                    "corredores sufrirán una interrupción, usando solo información disponible al cierre de la semana."),
            ])), lg=7, className="mb-3"),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H4("Variable objetivo", className="card-title"),
                dcc.Markdown(
                    r"Para cada corredor $c$ y semana $t$, con $r_{c,t}$ = kilogramos por día de encuesta:"
                    "\n\n"
                    r"$$y_{c,t+1}=\mathbb{1}\left[r_{c,t+1}<0{,}5\cdot\bar r_{c,\,t-7:t}\right]$$"
                    "\n\n"
                    "Es decir, hay **interrupción** cuando el abastecimiento de la semana siguiente cae por debajo de la "
                    "mitad de su promedio de las 8 semanas previas. Solo se evalúan corredores regulares (con carga en "
                    "al menos 4 de esas 8 semanas).", mathjax=True),
                html.Div([dbc.Badge("Clasificación binaria", color="primary", className="me-1"),
                          dbc.Badge("Panel espacio-temporal", color="info", className="me-1"),
                          dbc.Badge(f"{pct(K['tasa_total'])} de interrupciones", color="danger")]),
            ])), lg=5, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(tarjeta_kpi(num(K["registros_crudos"]), "Cargas registradas (2018-2025)", "truck"), sm=6, md=4, xl=2, className="mb-3"),
            dbc.Col(tarjeta_kpi(num(K["observaciones_panel"]), "Corredor-semanas (panel)", "grid-3x3"), sm=6, md=4, xl=2, className="mb-3"),
            dbc.Col(tarjeta_kpi(num(K["corredores"]), "Corredores", "signpost-split"), sm=6, md=4, xl=2, className="mb-3"),
            dbc.Col(tarjeta_kpi(num(K["municipios_origen"]), "Municipios de origen", "geo-alt"), sm=6, md=4, xl=2, className="mb-3"),
            dbc.Col(tarjeta_kpi(num(K["centrales"]), "Centrales mayoristas", "shop"), sm=6, md=4, xl=2, className="mb-3"),
            dbc.Col(tarjeta_kpi(f"{K['pdet_en_sipsa']} de {K['pdet_oficiales']}", "Municipios PDET con corredores regulares", "flag", "warning"),
                    sm=6, md=4, xl=2, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(grafico(F.fig_mapa_origenes()))), lg=7, className="mb-3"),
            dbc.Col(dbc.Card(dbc.CardBody([
                dbc.Row([dbc.Col(html.Label("Año"), width="auto"),
                         dbc.Col(dcc.Dropdown(id="ctx-anio", options=list(range(2018, 2026)), value=2024, clearable=False))],
                        align="center", className="mb-2"),
                grafico("ctx-toneladas"),
            ])), lg=5, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                dbc.RadioItems(id="ctx-ejemplo", inline=True, value="Irregular",
                               options=[{"label": "Corredor irregular", "value": "Irregular"},
                                        {"label": "Corredor estable", "value": "Estable"}]),
                grafico("ctx-ejemplo-fig"),
                html.Div(id="ctx-ejemplo-lectura"),
            ])), lg=8, className="mb-3"),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("Datos y método", className="card-title"),
                html.Ol([
                    html.Li([html.B("Fuente: "), "microdatos SIPSA-A del DANE (cada carga que entra a 32 centrales), "
                             "listado oficial de municipios PDET y coordenadas DIVIPOLA."]),
                    html.Li([html.B("Panel: "), "cargas agregadas por corredor y semana; predictoras calculadas solo con "
                             "información hasta la semana t."]),
                    html.Li([html.B("Partición cronológica: "), f"entrenamiento con semanas objetivo hasta "
                             f"{pd.Timestamp(K['fin_train']).strftime('%d/%m/%Y')} y prueba con 2025, separadas por un hueco de 10 semanas."]),
                    html.Li([html.B("Modelos base: "), "regresión logística y SVM lineal en un Pipeline, validados con bloques "
                             "de semanas y comparados con líneas base triviales."]),
                ], className="small"),
                html.A([html.I(className="bi bi-box-arrow-up-right me-1"), "Entregable 1 (EDA completo)"],
                       href=ENTREGABLE1, target="_blank", className="small"),
            ])), lg=4, className="mb-3"),
        ]),
    ], fluid="xl", className="pt-3")


# ============================================================================ 2. EDA
DIMENSIONES = {"total": "Total nacional", "mercado": "Central mayorista", "region_origen": "Región de origen",
               "grupo_dom": "Grupo de alimentos dominante", "pdet_origen": "Origen PDET"}


def tab_eda():
    t, e = EDA["temporal"], EDA["espacial"]
    ev = {x["evento"]: x for x in t["eventos"]}
    asoc = F.tabla("asociacion_objetivo").set_index("variable")
    top_auc = asoc[asoc["tipo"] == "predictora"]["auc_univariado"].idxmax()
    sub_temporal = dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col([html.Label("Desagregar por"),
                     dcc.Dropdown(id="eda-dim", options=[{"label": v, "value": k} for k, v in DIMENSIONES.items()],
                                  value="total", clearable=False)], md=4),
            dbc.Col([html.Label("Valores (máx. 6)"), dcc.Dropdown(id="eda-valores", multi=True)], md=8),
        ], className="mb-2"),
        grafico("eda-serie"),
        lectura(f"La tasa semanal de entrenamiento es {pct(K['tasa_train'])} y es estacionaria (ADF p < 0,001; KPSS p = "
                f"{num(t['kpss_p'], 2)}), pero tiene choques comunes: el paro nacional de 2021 la llevó a "
                f"{pct(ev['Paro nacional 2021']['maximo'])} en su peor semana y el confinamiento de 2020 a una media de "
                f"{pct(ev['Confinamiento COVID-19']['tasa'])}, frente a {pct(t['tasa_fuera_eventos'])} fuera de esos periodos.",
                "Por eso la validación es cronológica y los intervalos de confianza remuestrean semanas completas."),
        grafico(F.fig_estacionalidad()),
        lectura(f"La estacionalidad anual explica parte de la variación (fuerza {num(t['fuerza_estacional'], 2)}); las semanas "
                "con dos o tres festivos (Semana Santa, fin de año) tienen más interrupciones. Se incluyen la semana del año "
                "(seno y coseno) y los festivos de t+1, que se conocen de antemano."),
        dbc.Row([dbc.Col(grafico(F.fig_acf()), lg=7), dbc.Col(grafico(F.fig_autocorr_corredor()), lg=5)]),
        lectura(f"Dentro de cada corredor la interrupción de una semana se correlaciona {num(t['autocorr_corredor'][0]['correlacion'], 2)} "
                f"con la de la semana previa; la correlación baja a {num(t['autocorr_corredor'][4]['correlacion'], 2)} a las 8 semanas y se "
                "mantiene cerca de 0,11 hasta un año, por diferencias persistentes entre corredores. Justifica usar rezagos y ventanas "
                "de 8 y 26 semanas."),
    ]))
    sub_variables = dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col([html.Label("Variable numérica"),
                     dcc.Dropdown(id="eda-var", options=[{"label": ETIQUETAS[c], "value": c} for c in NUMERICAS],
                                  value="cv8", clearable=False)], md=6),
            dbc.Col([html.Label("Escala"), dbc.RadioItems(id="eda-log", options=[{"label": "Original", "value": 0},
                                                                                  {"label": "log(1 + x)", "value": 1}],
                                                          value=0, inline=True)], md=3),
        ], className="mb-2"),
        dbc.Row([dbc.Col(grafico("eda-dist"), lg=6), dbc.Col(grafico("eda-deciles"), lg=6)]),
        html.Div(id="eda-var-lectura"),
        html.Hr(),
        dbc.Row([
            dbc.Col([html.Label("Variable categórica"),
                     dcc.Dropdown(id="eda-cat", options=[{"label": ETIQUETAS.get(c, "Región de origen"), "value": c}
                                                         for c in CATEGORICAS + ["region_origen", "pdet_origen"]],
                                  value="grupo_dom", clearable=False), grafico("eda-cat-fig")], lg=6),
            dbc.Col([html.Label("Medida de asociación"),
                     dbc.RadioItems(id="eda-medida", options=[{"label": "|rbc| (Mann-Whitney)", "value": "rbc"},
                                                              {"label": "Información mutua", "value": "info_mutua"}],
                                    value="rbc", inline=True), grafico("eda-asoc")], lg=6),
        ]),
        lectura("Con más de 600 mil filas todas las diferencias resultan significativas (incluso con corrección de Holm); por eso "
                "se ordenan las variables por tamaño del efecto. Las más asociadas describen el **tamaño y la regularidad** del "
                "corredor: los corredores pequeños e irregulares se interrumpen mucho más."),
        grafico(F.fig_correlacion()),
        lectura("Las medidas de tamaño y regularidad están muy correlacionadas entre sí (|ρ| > 0,8), pero todos los VIF quedan por "
                "debajo de 5: se conservan y el modelo usa regularización L2."),
    ]))
    lisa = e["lisa_conteo"]
    sub_espacial = dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col([html.Label("Capa del mapa"),
                     dbc.RadioItems(id="eda-capa", options=[{"label": "Tasa de interrupción", "value": "tasa"},
                                                            {"label": "Clusters LISA", "value": "lisa"},
                                                            {"label": "Hotspots Gi*", "value": "gi"},
                                                            {"label": "Variabilidad (cv8)", "value": "cv8"}],
                                    value="tasa", inline=True)], md=9),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(grafico("eda-mapa"), lg=8),
            dbc.Col([
                tarjeta_kpi(num(e["moran_k8"], 3), "I de Moran global (k = 8)", "diagram-3", alto=False),
                html.Div(className="mb-2"),
                tarjeta_kpi(f"{lisa.get('Alto-Alto', 0)} / {lisa.get('Bajo-Bajo', 0)}", "Municipios Alto-Alto / Bajo-Bajo", "bullseye", "danger", alto=False),
                html.Div(className="mb-2"),
                tarjeta_kpi(num(e["knox"]["razon"], 2) + "×", "Agrupamiento espacio-tiempo (Knox)", "clock-history", "warning", alto=False),
                lectura(f"La tasa municipal tiene autocorrelación espacial positiva (I = {num(e['moran_k8'], 3)}, p = {num(e['moran_p'], 3)}), "
                        f"robusta a la elección de k ({num(e['moran_k4'], 2)} con k = 4 y {num(e['moran_k12'], 2)} con k = 12). La mitad de los "
                        "municipios Alto-Alto está en el Caribe (sobre todo Cesar) y los Bajo-Bajo se concentran en Norte de Santander, "
                        "Antioquia, Santander y Caldas.",
                        "Los eventos de interrupción masiva ocurren juntos en espacio y tiempo más de lo esperado: se valida también "
                        "con bloques espaciales (departamentos no vistos)."),
            ], lg=4),
        ]),
    ]))
    sub_fuga = dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col(grafico(F.fig_asociacion("auc_univariado")), lg=7),
            dbc.Col([
                html.H5("Auditoría de fuga de datos"),
                tabla_df(pd.DataFrame([
                    ("tasa_t1, tasa_t1 / base8", "No", "Definen el objetivo", "Excluidas"),
                    ("cod_mpio (corredor)", "Sí", "Identificador: memorizaría la tasa histórica", "Excluido"),
                    ("Ventanas hasta t (ratio_t, cv8, ...)", "Sí", "Solo usan semanas ≤ t", "Predictoras"),
                    ("frac_caida_* (t)", "Sí", "Agregan la semana t, no t+1", "En observación"),
                    ("Festivos y semana de t+1", "Sí", "Calendario conocido de antemano", "Predictoras"),
                    ("Días de encuesta en t+1", "No", "Se conoce al final de t+1", "Solo define la población"),
                ], columns=["Variable", "¿Disponible en t?", "Justificación", "Decisión"])),
                lectura(f"Las variables construidas con t+1 delatan la fuga (AUC = {num(asoc.loc['ratio_t1', 'auc_univariado'], 2)}) y se "
                        f"excluyen. Ninguna predictora admitida supera AUC {num(asoc.loc[top_auc, 'auc_univariado'], 2)} "
                        f"({ETIQUETAS[top_auc]}), un valor coherente con una relación real y no con una fuga."),
            ], lg=5),
        ]),
    ]))
    return dbc.Container([
        dbc.Alert("Todo el análisis exploratorio usa solo el conjunto de entrenamiento (semanas objetivo de 2018 a octubre de 2024); "
                  "la prueba (2025) se reservó antes de cualquier decisión de modelado.", color="info", className="small py-2"),
        dbc.Tabs([
            dbc.Tab(sub_temporal, label="Objetivo y tiempo", tab_id="t"),
            dbc.Tab(sub_variables, label="Variables y relaciones", tab_id="v"),
            dbc.Tab(sub_espacial, label="Espacio", tab_id="e"),
            dbc.Tab(sub_fuga, label="Fuga de datos", tab_id="f"),
        ], active_tab="t", className="subtabs"),
    ], fluid="xl", className="pt-3")


@app.callback(Output("ctx-toneladas", "figure"), Input("ctx-anio", "value"))
def cb_toneladas(anio):
    return F.fig_toneladas_mercado(anio)


@app.callback(Output("ctx-ejemplo-fig", "figure"), Output("ctx-ejemplo-lectura", "children"), Input("ctx-ejemplo", "value"))
def cb_ejemplo(tipo):
    e = {x["tipo"]: x for x in K["ejemplos"]}
    est, irr = e["Estable"], e["Irregular"]
    txt = (f"Los dos corredores llegan a Bazurto (Cartagena). El estable, desde {F.nombre_propio(est['origen'])} ({F.nombre_propio(est['depto'])}), "
           f"mueve en promedio {num(est['t_dia'], 1)} t/día y solo {pct(est['tasa_interrupcion'])} de sus semanas de 2023-2024 cumplen la "
           f"definición de interrupción. El irregular, desde {F.nombre_propio(irr['origen'])} ({F.nombre_propio(irr['depto'])}), mueve {num(irr['t_dia'], 1)} t/día "
           f"y se interrumpe en {pct(irr['tasa_interrupcion'])} de las semanas: caídas bruscas seguidas de recuperación. "
           "Anticipar cuál de los dos patrones seguirá cada corredor es la tarea del modelo.")
    return F.fig_corredor_ejemplo(tipo), lectura(txt, titulo="Dos patrones de abastecimiento")


@app.callback(Output("eda-valores", "options"), Output("eda-valores", "value"), Input("eda-dim", "value"))
def cb_valores(dim):
    if dim == "total":
        return [], []
    s = F.tabla("serie_semanal")
    s = s[s["dimension"] == dim]
    orden = s.groupby("valor")["n"].sum().sort_values(ascending=False).index.tolist()
    etiqueta = (lambda v: {"0": "No PDET", "1": "PDET"}.get(v, v)) if dim == "pdet_origen" else (lambda v: v)
    return [{"label": etiqueta(v), "value": v} for v in orden], orden[: (2 if dim == "pdet_origen" else 4)]


@app.callback(Output("eda-serie", "figure"), Input("eda-dim", "value"), Input("eda-valores", "value"))
def cb_serie(dim, valores):
    return F.fig_serie(dim, (valores or [])[:6] if dim != "total" else None)


@app.callback(Output("eda-dist", "figure"), Output("eda-deciles", "figure"), Output("eda-var-lectura", "children"),
              Input("eda-var", "value"), Input("eda-log", "value"))
def cb_variable(var, log):
    d = F.tabla("descriptivos").set_index("variable").loc[var]
    a = F.tabla("asociacion_objetivo").set_index("variable").loc[var]
    dec = F.tabla("deciles")
    dec = dec[dec["variable"] == var]
    txt = (f"**{ETIQUETAS[var]}** · mediana {num(d['50%'], 3)}, asimetría {num(d['asimetria'], 2)}, curtosis {num(d['curtosis'], 2)}, "
           f"{num(d['pct_outliers_iqr'], 1)} % de valores fuera de 1,5·IQR. Mediana con interrupción {num(a['mediana_interr'], 3)} "
           f"frente a {num(a['mediana_no'], 3)} sin ella; tamaño del efecto |rbc| = {num(abs(a['rbc']), 3)} y AUC univariado "
           f"{num(a['auc_univariado'], 3)}. Por deciles, la tasa va de {pct(dec['tasa'].min())} a {pct(dec['tasa'].max())}.")
    return F.fig_distribucion(var, bool(log)), F.fig_deciles(var), lectura(txt, titulo="Resumen de la variable")


@app.callback(Output("eda-cat-fig", "figure"), Input("eda-cat", "value"))
def cb_cat(var):
    return F.fig_tasa_categoria(var)


@app.callback(Output("eda-asoc", "figure"), Input("eda-medida", "value"))
def cb_asoc(medida):
    fig = F.fig_asociacion(medida)
    fig.update_layout(height=560)
    return fig


@app.callback(Output("eda-mapa", "figure"), Input("eda-capa", "value"))
def cb_mapa(capa):
    return F.fig_mapa_tasa(capa)


# ============================================================================ 3. modelos
MODELOS_PROB = {"Regresión logística": "p_logistica", "SVM lineal": "p_svm"}


def tab_modelos():
    lg, sv = MOD["Regresión logística"], MOD["SVM lineal"]
    auc_l, auc_s = metrica("Regresión logística", "AUC-ROC"), metrica("SVM lineal", "AUC-ROC")
    auc_p = metrica("Persistencia", "AUC-ROC")
    cmp_ls = COMP[(COMP["modelo_a"] == "Regresión logística") & (COMP["modelo_b"] == "SVM lineal")].iloc[0]
    tabla_met = MET[MET["metrica"].isin(["accuracy", "precision", "recall", "F1", "AUC-ROC", "AUC-PR"])].copy()
    tabla_met["valor"] = [f"{v:.3f} [{a:.3f}; {b:.3f}]".replace(".", ",") if pd.notna(a) else f"{v:.3f}".replace(".", ",")
                          for v, a, b in zip(tabla_met["valor"], tabla_met["ic_inf"], tabla_met["ic_sup"])]
    tabla_met = tabla_met.pivot(index="modelo", columns="metrica", values="valor")[
        ["AUC-ROC", "AUC-PR", "F1", "recall", "precision", "accuracy"]].reindex(
        ["Regresión logística", "SVM lineal", "Persistencia", "Dummy (estratificado)", "Dummy (clase mayoritaria)"]).reset_index()
    tabla_cmp = COMP.assign(
        par=COMP["modelo_a"] + " vs " + COMP["modelo_b"],
        dAUC=[f"{d:+.3f} [{a:+.3f}; {b:+.3f}]".replace(".", ",") for d, a, b in zip(COMP["delta_auc"], COMP["ic_boot_inf"], COMP["ic_boot_sup"])],
        DeLong=[f"z = {z:.1f}; p = {p:.1e}".replace(".", ",") for z, p in zip(COMP["z"], COMP["p_delong_holm"])],
        McNemar=[f"χ² = {c:.0f}; p = {p:.1e}".replace(".", ",") for c, p in zip(COMP["chi2_mcnemar"], COMP["p_mcnemar"])],
    )[["par", "dAUC", "DeLong", "McNemar"]].rename(columns={"par": "Comparación", "dAUC": "ΔAUC [IC 95 % bootstrap]",
                                                            "DeLong": "DeLong (p Holm)", "McNemar": "McNemar (umbrales elegidos)"})
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.Label("Modelo"),
                    dbc.RadioItems(id="mod-modelo", options=[{"label": m, "value": m} for m in MODELOS_PROB],
                                   value="Regresión logística", className="mb-3"),
                    html.Label(id="mod-umbral-label"),
                    dcc.Slider(id="mod-umbral", min=0.05, max=0.8, step=0.01, value=lg["umbral"],
                               marks={0.1: "0,1", 0.3: "0,3", 0.5: "0,5", 0.7: "0,7"}, tooltip={"placement": "bottom"}),
                    html.Div(id="mod-umbral-nota", className="small text-muted mt-2"),
                    html.Hr(),
                    html.Div([html.B("Cómo se entrenó. "),
                              "Pipeline con imputación, log, winsorización, splines para la dinámica reciente, escalado y one-hot; "
                              f"C elegido por AUC en 5 bloques temporales: logística C = {lg['C']}, SVM C = {sv['C']}. "
                              "La SVM se calibra con Platt sobre predicciones fuera de pliegue."], className="small"),
                ])),
            ], lg=3, className="mb-3"),
            dbc.Col([
                dbc.Row(id="mod-kpis", className="g-2 mb-2"),
                dbc.Row([dbc.Col(grafico("mod-matriz"), md=5), dbc.Col(html.Div(id="mod-lectura-umbral"), md=7)]),
            ], lg=9, className="mb-3"),
        ]),
        dbc.Card(dbc.CardBody([
            dbc.Checklist(id="mod-curvas", inline=True, value=["Regresión logística", "SVM lineal", "Persistencia", "Dummy (estratificado)"],
                          options=[{"label": m, "value": m} for m in ["Regresión logística", "SVM lineal", "Persistencia", "Dummy (estratificado)"]]),
            grafico("mod-roc"),
            lectura(f"En la prueba (todo 2025) la regresión logística alcanza AUC-ROC {num(auc_l['valor'], 3)} "
                    f"(IC 95 %: {num(auc_l['ic_inf'], 3)} a {num(auc_l['ic_sup'], 3)}) y la SVM lineal {num(auc_s['valor'], 3)} "
                    f"(IC 95 %: {num(auc_s['ic_inf'], 3)} a {num(auc_s['ic_sup'], 3)}), frente a 0,50 de los Dummy y {num(auc_p['valor'], 3)} de la "
                    f"persistencia. La diferencia entre los dos modelos lineales es {num(cmp_ls['delta_auc'], 3)} puntos de AUC "
                    "(ver tabla de comparaciones): ambos ordenan el riesgo casi igual porque comparten predictoras y forma lineal."),
        ]), className="mb-3"),
        dbc.Card(dbc.CardBody([
            html.H5("Métricas en prueba con intervalos de confianza al 95 % (bootstrap por bloques de semana, 500 réplicas)"),
            html.Div("Cada modelo con su umbral elegido en entrenamiento (F1 fuera de pliegue); la persistencia usa la regla ratio_t < 0,5.",
                     className="small text-muted mb-2"),
            tabla_df(tabla_met.rename(columns={"modelo": "Modelo"})),
            html.H6("Comparación entre modelos", className="mt-3"),
            tabla_df(tabla_cmp),
        ]), className="mb-3"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([grafico(F.fig_calibracion()),
                                           lectura("Las probabilidades de la logística siguen la diagonal: si el modelo dice 40 %, cerca de 40 % de "
                                                   "esos corredores se interrumpe. Eso permite usarlas para priorizar recursos, no solo para clasificar.")])), lg=6, className="mb-3"),
            dbc.Col(dbc.Card(dbc.CardBody([
                dbc.Row([dbc.Col(html.Label("Coeficientes a mostrar"), width="auto"),
                         dbc.Col(dcc.Slider(id="mod-top", min=10, max=40, step=5, value=20, marks=None, tooltip={"placement": "bottom"}))]),
                grafico("mod-coef"),
            ])), lg=6, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([grafico("mod-cv")])), lg=7, className="mb-3"),
            dbc.Col(dbc.Card(dbc.CardBody([grafico(F.fig_cv_espacial()),
                                           lectura("En departamentos que no vio al entrenar, el AUC es casi el mismo que en la validación temporal: "
                                                   "el modelo generaliza porque usa atributos del corredor y no identificadores ni coordenadas.")])), lg=5, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([grafico(F.fig_curva_aprendizaje()),
                                           lectura("Las curvas se aplanan hacia las 50 mil filas y la brecha entre entrenamiento y validación casi "
                                                   "desaparece: no hay sobreajuste y más datos no mejorarían un modelo lineal; el límite es su forma.")])), lg=6, className="mb-3"),
            dbc.Col(dbc.Card(dbc.CardBody([
                dbc.Row([dbc.Col(html.Label("Subgrupo"), width="auto"),
                         dbc.Col(dcc.Dropdown(id="mod-sub", clearable=False, value="region_origen",
                                              options=[{"label": "Región de origen", "value": "region_origen"},
                                                       {"label": "Central mayorista", "value": "mercado"},
                                                       {"label": "Grupo de alimentos", "value": "grupo_dom"},
                                                       {"label": "Origen PDET", "value": "pdet_origen"}]))], align="center"),
                grafico("mod-sub-fig"),
            ])), lg=6, className="mb-3"),
        ]),
        dbc.Card(dbc.CardBody([
            html.H5([html.I(className="bi bi-sliders me-2"), "Simulador: probabilidad de interrupción de un corredor"]),
            html.Div("Ajusta las características principales; las demás se fijan en la mediana (o moda) del entrenamiento. "
                     "Usa el pipeline de regresión logística entrenado.", className="small text-muted mb-2"),
            dbc.Row([
                dbc.Col([
                    *[html.Div([html.Label(ETIQUETAS[c], className="small mb-0"),
                                dcc.Slider(id=f"sim-{c}", min=round(MOD["rangos_simulador"][c][0], 2), max=round(MOD["rangos_simulador"][c][1], 2),
                                           value=round(MOD["perfil_base"][c], 2), marks=None, tooltip={"placement": "bottom", "always_visible": False})],
                               className="mb-1") for c in ["n_envios8", "cv8", "activas26", "ratio_t", "dist_km", "festivos_t1"]],
                ], lg=5),
                dbc.Col([
                    html.Label("Grupo de alimentos dominante", className="small mb-0"),
                    dcc.Dropdown(id="sim-grupo_dom", options=MOD["categorias"]["grupo_dom"], value=MOD["perfil_base"]["grupo_dom"], clearable=False, className="mb-2"),
                    html.Label("Central mayorista de destino", className="small mb-0"),
                    dcc.Dropdown(id="sim-mercado", options=MOD["categorias"]["mercado"], value="Cartagena, Bazurto", clearable=False, className="mb-2"),
                    dbc.Checklist(id="sim-pdet", options=[{"label": "Municipio de origen PDET", "value": 1}], value=[], switch=True),
                ], lg=3),
                dbc.Col(html.Div(id="sim-resultado"), lg=4),
            ]),
        ]), className="mb-4"),
    ], fluid="xl", className="pt-3")


@app.callback(Output("mod-umbral", "value"), Input("mod-modelo", "value"))
def cb_umbral_por_modelo(modelo):
    return MOD[modelo]["umbral"]


@app.callback(Output("mod-kpis", "children"), Output("mod-matriz", "figure"), Output("mod-umbral-label", "children"),
              Output("mod-umbral-nota", "children"), Output("mod-lectura-umbral", "children"),
              Input("mod-modelo", "value"), Input("mod-umbral", "value"))
def cb_umbral(modelo, umbral):
    y, p = PRED["y"].to_numpy(), PRED[MODELOS_PROB[modelo]].to_numpy()
    m = F.metricas_umbral(y, p, umbral)
    base = 1 - y.mean()
    kpis = [dbc.Col(tarjeta_kpi(num(m[k], 3), n, i, c), xs=6, md=4, xl=2) for k, n, i, c in [
        ("AUC-ROC", "AUC-ROC", "graph-up", "primary"), ("F1", "F1", "bullseye", "primary"),
        ("recall", "Recall (interrupciones detectadas)", "search", "danger"), ("precision", "Precisión de las alertas", "bell", "warning"),
        ("accuracy", "Exactitud", "check2-circle", "secondary")]]
    kpis.append(dbc.Col(tarjeta_kpi(num(m["alertas"]), "Alertas emitidas", "megaphone", "info"), xs=6, md=4, xl=2))
    elegido = MOD[modelo]["umbral"]
    nota = f"Umbral elegido en entrenamiento: {num(elegido, 2)} (máximo F1 fuera de pliegue)."
    txt = lectura(
        f"Con umbral {num(umbral, 2)} el modelo emite {num(m['alertas'])} alertas en 2025 y detecta {pct(m['recall'])} de las "
        f"{num(m['tp'] + m['fn'])} interrupciones; {pct(m['precision'])} de las alertas son interrupciones reales.",
        f"La exactitud ({pct(m['accuracy'])}) se compara con {pct(base)} de predecir siempre «sin interrupción»: con este "
        "desbalance la exactitud engaña, por eso el umbral se elige por F1 y se reportan AUC y AUC-PR.",
        "Bajar el umbral detecta más interrupciones a costa de más falsas alarmas; para una alerta temprana de abastecimiento "
        "suele preferirse un recall alto.", titulo="Qué significa este umbral")
    return kpis, F.fig_matriz(m, f"Matriz de confusión · {modelo}"), f"Umbral de decisión: {num(umbral, 2)}", nota, txt


@app.callback(Output("mod-roc", "figure"), Input("mod-curvas", "value"))
def cb_roc(modelos):
    return F.fig_roc_pr(modelos or ["Regresión logística"])


@app.callback(Output("mod-coef", "figure"), Output("mod-cv", "figure"), Input("mod-modelo", "value"), Input("mod-top", "value"))
def cb_coef(modelo, top):
    return F.fig_coeficientes(modelo, top), F.fig_cv(modelo)


@app.callback(Output("mod-sub-fig", "figure"), Input("mod-sub", "value"), Input("mod-modelo", "value"))
def cb_sub(dim, modelo):
    return F.fig_subgrupos(dim, modelo)


SIM = ["n_envios8", "cv8", "activas26", "ratio_t", "dist_km", "festivos_t1"]


@app.callback(Output("sim-resultado", "children"),
              [Input(f"sim-{c}", "value") for c in SIM] + [Input("sim-grupo_dom", "value"), Input("sim-mercado", "value"), Input("sim-pdet", "value")])
def cb_simulador(*vals):
    fila = dict(R["modelos"]["perfil_base"])
    fila.update(dict(zip(SIM, vals[:len(SIM)])))
    fila["grupo_dom"], fila["mercado"], fila["pdet_origen"] = vals[len(SIM)], vals[len(SIM) + 1], int(bool(vals[len(SIM) + 2]))
    X = pd.DataFrame([fila])[X_COLS]
    p = float(MODELO.predict_proba(X)[0, 1])
    umbral = MOD["Regresión logística"]["umbral"]
    color = "danger" if p >= umbral else "success"
    return dbc.Card(dbc.CardBody([
        html.Div("Probabilidad de interrupción la próxima semana", className="kpi-etiqueta"),
        html.Div(pct(p), className=f"display-5 fw-bold text-{color}"),
        dbc.Progress(value=100 * p, color=color, className="my-2", style={"height": "10px"}),
        html.Div(("Alerta: supera el umbral " if p >= umbral else "Sin alerta: por debajo del umbral ") + num(umbral, 2), className="small"),
        html.Div(f"Tasa media de entrenamiento: {pct(K['tasa_train'])}", className="small text-muted"),
    ]), className="h-100 sim-card")


# ============================================================================ layout
app.layout = html.Div([
    encabezado,
    dbc.Container(dbc.Tabs(id="tabs", active_tab="contexto", className="tabs-principales", children=[
        dbc.Tab(label="1. Contexto del problema", tab_id="contexto"),
        dbc.Tab(label="2. Análisis exploratorio", tab_id="eda"),
        dbc.Tab(label="3. Modelos base", tab_id="modelos"),
    ]), fluid="xl"),
    html.Div(id="contenido"),
    html.Footer(dbc.Container([
        html.Span("Entregable 2 · Curso de Machine Learning · Doctorado en Ingeniería · "),
        html.A("Informe en Jupyter Book", href=LIBRO, target="_blank"), html.Span(" · "),
        html.A("Código", href=REPO, target="_blank"),
    ], fluid="xl"), className="pie"),
])


@app.callback(Output("contenido", "children"), Input("tabs", "active_tab"))
def cb_tab(tab):
    return {"contexto": tab_contexto, "eda": tab_eda, "modelos": tab_modelos}[tab]()


if __name__ == "__main__":
    # DASH_DEBUG=1 activa la recarga automática al editar (solo para desarrollo local)
    app.run(debug=os.environ.get("DASH_DEBUG") == "1", port=8050)
