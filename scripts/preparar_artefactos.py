"""
Prepara los artefactos que consumen el dashboard (src/app.py) y el Jupyter Book (book/).

Parte del panel de corredores construido en el Entregable 1 (proyecto_sipsa) y deja en data/
archivos pequeños: agregados para el EDA, estadísticos, resultados de los modelos base,
predicciones de prueba y el pipeline logístico serializado. Así el dashboard no necesita cargar
el panel completo (cabe en el plan gratuito de Render) y el libro interpreta exactamente las
mismas cifras que muestra el dashboard.

Uso:
    python scripts/preparar_artefactos.py --proyecto "../proyecto_sipsa"
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score,
                             roc_curve, silhouette_score)
from sklearn.metrics.pairwise import haversine_distances
from sklearn.model_selection import GridSearchCV, GroupKFold, learning_curve
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# Los procesos paralelos de joblib deben poder importar modelo_utils (Winsorizador, etc.)
os.environ["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
import modelo_utils as mu  # noqa: E402

import warnings  # noqa: E402

warnings.filterwarnings("ignore", category=InterpolationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DATA = ROOT / "data"
R = 6371.0
Y = mu.VAR_OBJETIVO
RNG = np.random.default_rng(mu.SEMILLA)


# ------------------------------------------------------------------------------ utilidades
def guardar_json(obj, nombre):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return o.strftime("%Y-%m-%d")
        raise TypeError(type(o))
    (DATA / nombre).write_text(json.dumps(obj, default=conv, ensure_ascii=False, indent=1), encoding="utf-8")


def pesos_knn(lat, lon, k=8):
    X = np.radians(np.column_stack([lat, lon]))
    _, idx = BallTree(X, metric="haversine").query(X, k=k + 1)
    n = len(lat)
    W = np.zeros((n, n))
    W[np.repeat(np.arange(n), k), idx[:, 1:].ravel()] = 1.0 / k
    return W


def moran_i(y, W, perm=999):
    y = np.asarray(y, float)
    z = y - y.mean()
    n, s0 = len(y), W.sum()
    I = n / s0 * (z @ W @ z) / (z @ z)
    sim = np.array([(lambda zp: n / s0 * (zp @ W @ zp) / (zp @ zp))(RNG.permutation(z)) for _ in range(perm)])
    p = (np.sum(sim >= I) + 1) / (perm + 1) if I >= sim.mean() else (np.sum(sim <= I) + 1) / (perm + 1)
    return float(I), float(p)


def lisa(y, W, perm=499):
    y = np.asarray(y, float)
    z = (y - y.mean()) / y.std()
    lag = W @ z
    Ii = z * lag
    n = len(y)
    k = (W > 0).sum(axis=1)
    p = np.empty(n)
    for i in range(n):
        otros = np.delete(z, i)
        sims = np.array([otros[RNG.choice(n - 1, k[i], replace=False)].mean() for _ in range(perm)]) * z[i]
        p[i] = (np.sum(np.abs(sims) >= abs(Ii[i])) + 1) / (perm + 1)
    cuad = np.where(z > 0, np.where(lag > 0, "Alto-Alto", "Alto-Bajo"), np.where(lag > 0, "Bajo-Alto", "Bajo-Bajo"))
    return np.where(p < 0.05, cuad, "No significativo"), z, lag


def gi_star(y, lat, lon, k=8):
    y = np.asarray(y, float)
    _, idx = BallTree(np.radians(np.column_stack([lat, lon])), metric="haversine").query(
        np.radians(np.column_stack([lat, lon])), k=k + 1)
    n, w = len(y), k + 1
    s = np.sqrt((y ** 2).mean() - y.mean() ** 2)
    return (y[idx].sum(axis=1) - y.mean() * w) / (s * np.sqrt((n * w - w ** 2) / (n - 1)))


def cramers_v(t):
    chi2 = stats.chi2_contingency(t, correction=False)[0]
    return float(np.sqrt(chi2 / (t.to_numpy().sum() * (min(t.shape) - 1))))


def delong_rapido(y, s1, s2):
    """Prueba de DeLong para dos AUC correlacionados (algoritmo rápido de Sun y Xu, 2014)."""
    y = np.asarray(y)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()

    def comps(s):
        sx, sy = s[pos], s[neg]
        tx = stats.rankdata(sx)
        ty = stats.rankdata(sy)
        tz = stats.rankdata(np.concatenate([sx, sy]))
        auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        v01 = (tz[:m] - tx) / n
        v10 = 1 - (tz[m:] - ty) / m
        return auc, v01, v10

    a1, v01a, v10a = comps(np.asarray(s1, float))
    a2, v01b, v10b = comps(np.asarray(s2, float))
    sx = np.cov(np.vstack([v01a, v01b]))
    sy = np.cov(np.vstack([v10a, v10b]))
    S = sx / m + sy / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (a1 - a2) / np.sqrt(var)
    return float(a1), float(a2), float(a1 - a2), float(np.sqrt(var)), float(z), float(2 * stats.norm.sf(abs(z)))


def submuestrear_curva(x, y, n=300):
    idx = np.unique(np.linspace(0, len(x) - 1, min(n, len(x))).astype(int))
    return np.asarray(x)[idx], np.asarray(y)[idx]


def nombres_limpios(pipe):
    nombres = pipe.named_steps["prep"].get_feature_names_out()
    return pd.Index(nombres).str.replace(r"^(log|spl|num|bin|cat)__", "", regex=True)


# ------------------------------------------------------------------------------ contexto
def artefactos_contexto(panel, proyecto):
    t0 = time.time()
    divi = pd.read_csv(proyecto / "data/raw/divipola_municipios.csv", dtype=str)
    pdet = pd.read_csv(proyecto / "data/raw/pdet_municipios.csv", dtype=str)
    envios_path = proyecto / "data/interim/sipsa_a_envios.parquet"
    import pyarrow.parquet as pq
    n_cargas = pq.ParquetFile(envios_path).metadata.num_rows
    env = pd.read_parquet(envios_path, columns=["mercado", "fecha", "kg", "cod_mpio"])
    env = env[(env["kg"] > 0) & env["fecha"].notna()]
    env["mercado"] = env["mercado"].astype(str).replace(
        {"Cali, Santa Helena": "Cali, Santa Elena", "Pereira, La 41-Impala": "Pereira, La 41"})
    env["anio"] = env["fecha"].dt.year
    toneladas_mercado = (env.groupby(["mercado", "anio"])["kg"].sum() / 1e3).rename("toneladas").reset_index()
    toneladas_mercado.to_parquet(DATA / "toneladas_mercado_anio.parquet", index=False)

    train, test = mu.particion(panel)
    kpis = {
        "registros_crudos": int(n_cargas),
        "observaciones_panel": int(len(panel)),
        "corredores": int(panel.groupby(["mercado", "cod_mpio"]).ngroups),
        "municipios_origen": int(panel["cod_mpio"].nunique()),
        "centrales": int(panel["mercado"].nunique()),
        "semanas": int(panel["semana"].nunique()),
        "periodo": [panel["semana"].min(), panel["semana_t1"].max()],
        "filas_train": int(len(train)), "filas_test": int(len(test)),
        "tasa_train": float(train[Y].mean()), "tasa_test": float(test[Y].mean()),
        "tasa_total": float(panel[Y].mean()),
        "pdet_oficiales": int(pdet["cod_muni"].nunique()),
        "pdet_en_sipsa": int(len(set(pdet["cod_muni"].str.zfill(5)) & set(panel["cod_mpio"]))),
        "municipios_divipola": int(len(divi)),
        "fin_train": mu.FIN_TRAIN, "inicio_test": mu.INICIO_TEST,
    }

    # Corredores de ejemplo hacia Cartagena (Bazurto), en entrenamiento 2023-2024: el estable es el de mayor
    # volumen; el irregular, el de mayor tasa de interrupción entre los que mueven al menos 5 t/día. Son flujos
    # grandes que agregan muchas cargas, de modo que no exponen la actividad de un productor individual.
    baz = panel[(panel["mercado"] == "Cartagena, Bazurto") & (panel["semana_t1"] >= "2023-01-01") & (panel["semana_t1"] <= mu.FIN_TRAIN)]
    # t_ref: nivel de referencia (promedio móvil de 8 semanas), usado para elegir; t_dia: media de la serie graficada
    g = baz.groupby("cod_mpio").agg(t_ref=("base8", lambda x: x.mean() / 1e3), t_dia=("tasa_t1", lambda x: x.mean() / 1e3),
                                    tasa=(Y, "mean"), n=(Y, "size"), origen=("mpio_origen", "first"),
                                    depto=("depto_origen", "first"), pdet=("pdet_origen", "first"))
    g = g[g["n"] >= 60]
    estable = g["t_ref"].idxmax()
    irregular = g[g["t_ref"] >= 5]["tasa"].idxmax()
    partes, ejemplos = [], []
    for tipo, cod in [("Estable", estable), ("Irregular", irregular)]:
        ej = baz[baz["cod_mpio"] == cod][["semana_t1", "tasa_t1", "base8", Y]].assign(umbral=lambda d: 0.5 * d["base8"], tipo=tipo, cod_mpio=cod)
        partes.append(ej)
        f = g.loc[cod]
        ejemplos.append({"tipo": tipo, "cod_mpio": cod, "origen": f["origen"], "depto": f["depto"], "pdet": int(f["pdet"]),
                         "central": "Cartagena, Bazurto", "t_dia": float(f["t_dia"]), "t_ref": float(f["t_ref"]),
                         "tasa_interrupcion": float(f["tasa"]), "semanas": int(f["n"])})
    pd.concat(partes).to_parquet(DATA / "corredor_ejemplo.parquet", index=False)
    kpis["ejemplos"] = ejemplos
    print(f"  contexto: {time.time() - t0:.0f} s")
    return kpis


# ------------------------------------------------------------------------------ EDA
def artefactos_eda(train, test, proyecto):
    t0 = time.time()
    res = {}
    train = train.assign(region_origen=train["depto_origen"].map(mu.REGION), anio=train["semana_t1"].dt.year)

    # Series semanales por dimensión (n y número de interrupciones)
    partes = []
    for dim in [None, "mercado", "region_origen", "grupo_dom", "pdet_origen"]:
        keys = ["semana_t1"] + ([dim] if dim else [])
        g = train.groupby(keys)[Y].agg(n="size", k="sum").reset_index()
        g["dimension"] = dim or "total"
        g["valor"] = g[dim].astype(str) if dim else "Total"
        partes.append(g[["dimension", "valor", "semana_t1", "n", "k"]])
    pd.concat(partes, ignore_index=True).to_parquet(DATA / "serie_semanal.parquet", index=False)

    # Muestra estratificada por año para gráficos interactivos (histogramas, cajas)
    muestra = train.groupby("anio").sample(n=9000, random_state=mu.SEMILLA)
    cols = mu.NUMERICAS + mu.BINARIAS + mu.CATEGORICAS + [Y, "region_origen", "anio", "base8"]
    muestra[cols].to_parquet(DATA / "muestra_eda.parquet", index=False)

    # Descriptivos completos (todo el entrenamiento)
    num = mu.NUMERICAS + ["base8"]
    d = train[num].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    d["asimetria"] = train[num].skew()
    d["curtosis"] = train[num].kurt()
    q1, q3 = train[num].quantile(0.25), train[num].quantile(0.75)
    iqr = q3 - q1
    d["pct_outliers_iqr"] = ((train[num] < q1 - 1.5 * iqr) | (train[num] > q3 + 1.5 * iqr)).mean() * 100
    d.reset_index(names="variable").to_parquet(DATA / "descriptivos.parquet", index=False)

    # Asociación con el objetivo: Mann-Whitney + correlación biserial de rangos + Holm, AUC univariado, IM
    y = train[Y].to_numpy()
    filas = []
    for c in mu.NUMERICAS + mu.BINARIAS:
        x = train[c].fillna(train[c].median()).to_numpy()
        a, b = x[y == 1], x[y == 0]
        mw = stats.mannwhitneyu(a, b)
        auc = mw.statistic / (len(a) * len(b))
        filas.append({"variable": c, "mediana_interr": np.median(a), "mediana_no": np.median(b),
                      "p": mw.pvalue, "rbc": 2 * auc - 1, "auc_univariado": max(auc, 1 - auc), "tipo": "predictora"})
    ctrl = train.assign(ratio_t1=train["tasa_t1"] / train["base8"])
    for c in ["tasa_t1", "ratio_t1", "tasa_t", "base8"]:
        a = roc_auc_score(y, ctrl[c])
        filas.append({"variable": c, "auc_univariado": max(a, 1 - a),
                      "tipo": "control de fuga (excluida)" if c in ("tasa_t1", "ratio_t1") else "auxiliar"})
    for c in mu.CATEGORICAS:
        tasa = train.groupby(c)[Y].transform("mean")
        filas.append({"variable": c, "auc_univariado": roc_auc_score(y, tasa), "tipo": "predictora (codificación por objetivo)"})
    biv = pd.DataFrame(filas)
    m = biv["p"].notna()
    biv.loc[m, "p_holm"] = multipletests(biv.loc[m, "p"], method="holm")[1]
    s = train.sample(120000, random_state=mu.SEMILLA)
    Xmi = s[mu.NUMERICAS + mu.BINARIAS].fillna(s[mu.NUMERICAS + mu.BINARIAS].median())
    mi = dict(zip(Xmi.columns, mutual_info_classif(Xmi, s[Y], random_state=mu.SEMILLA,
                                                   discrete_features=[c in mu.BINARIAS for c in Xmi.columns])))
    for c in mu.CATEGORICAS:
        mi[c] = mutual_info_classif(s[c].astype("category").cat.codes.to_frame(), s[Y], discrete_features=True,
                                    random_state=mu.SEMILLA)[0]
    biv["info_mutua"] = biv["variable"].map(mi)
    biv.to_parquet(DATA / "asociacion_objetivo.parquet", index=False)

    # Categóricas: chi2 y V de Cramér
    cat = []
    for c in mu.CATEGORICAS + mu.BINARIAS + ["region_origen"]:
        t = pd.crosstab(train[c], train[Y])
        chi2, p, dof, _ = stats.chi2_contingency(t, correction=False)
        cat.append({"variable": c, "categorias": t.shape[0], "chi2": chi2, "gl": dof, "p": p, "v_cramer": cramers_v(t)})
    pd.DataFrame(cat).to_parquet(DATA / "categoricas.parquet", index=False)
    tasas_cat = []
    for c in mu.CATEGORICAS + ["region_origen", "pdet_origen"]:
        g = train.groupby(c)[Y].agg(tasa="mean", n="size").reset_index().rename(columns={c: "categoria"})
        g["variable"] = c
        g["categoria"] = g["categoria"].astype(str)
        tasas_cat.append(g)
    pd.concat(tasas_cat).to_parquet(DATA / "tasa_por_categoria.parquet", index=False)

    # Tasa por deciles de cada predictora numérica (relaciones no lineales, p. ej. la U de ratio_t)
    dec = []
    for c in mu.NUMERICAS:
        q = pd.qcut(train[c], 10, duplicates="drop")
        g = train.groupby(q, observed=True).agg(x=(c, "mean"), tasa=(Y, "mean"), n=(Y, "size")).reset_index(drop=True)
        g["variable"] = c
        g["decil"] = np.arange(1, len(g) + 1)
        dec.append(g)
    pd.concat(dec).to_parquet(DATA / "deciles.parquet", index=False)

    # Correlación de Spearman y VIF
    train[mu.NUMERICAS].corr(method="spearman").to_parquet(DATA / "correlacion_spearman.parquet")
    vs = train.sample(60000, random_state=mu.SEMILLA)[mu.NUMERICAS + mu.BINARIAS].dropna().drop(columns="sh_procesados")
    Xv = (vs - vs.mean()) / vs.std()
    Xv.insert(0, "const", 1.0)
    pd.DataFrame({"variable": Xv.columns[1:],
                  "vif": [variance_inflation_factor(Xv.values, i) for i in range(1, Xv.shape[1])]}) \
        .sort_values("vif", ascending=False).to_parquet(DATA / "vif.parquet", index=False)

    # Efecto de festivos y estacionalidad mensual
    res["festivos"] = train.groupby("festivos_t1")[Y].agg(["mean", "size"]).reset_index().to_dict("records")
    tasa_sem = train.groupby("semana_t1")[Y].mean()
    tm = pd.DataFrame({"tasa": tasa_sem, "mes": tasa_sem.index.month})
    kw = stats.kruskal(*[g["tasa"] for _, g in tm.groupby("mes")])
    res["kruskal_mes"] = {"H": kw.statistic, "p": kw.pvalue}

    # Multivariado: PCA, outliers y clustering de corredores (solo libro)
    ps = train.sample(100000, random_state=mu.SEMILLA)
    cols_pca = [c for c in mu.NUMERICAS if c not in ("sem_sin", "sem_cos")]
    Xp = StandardScaler().fit_transform(ps[cols_pca].fillna(ps[cols_pca].median()))
    pca = PCA(random_state=mu.SEMILLA).fit(Xp)
    Z = pca.transform(Xp)[:, :2]
    idx = RNG.choice(len(Z), 15000, replace=False)
    pd.DataFrame({"pc1": Z[idx, 0], "pc2": Z[idx, 1], "y": ps[Y].to_numpy()[idx]}).to_parquet(DATA / "pca_individuos.parquet", index=False)
    pd.DataFrame(pca.components_[:2].T, index=cols_pca, columns=["PC1", "PC2"]).reset_index(names="variable") \
        .to_parquet(DATA / "pca_cargas.parquet", index=False)
    res["pca"] = {"varianza": pca.explained_variance_ratio_, "n80": int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.8) + 1),
                  "kaiser": int((pca.explained_variance_ > 1).sum()), "p": len(cols_pca)}
    cols_out = ["log_base8", "ratio_t", "ratio_t_1", "cv8", "pend8_rel", "activas26", "n_envios8", "dist_km"]
    mo = train.sample(40000, random_state=mu.SEMILLA)[cols_out + [Y]].dropna()
    Xo = mo[cols_out].assign(n_envios8=np.log1p(mo["n_envios8"]), dist_km=np.log1p(mo["dist_km"]))
    d2 = MinCovDet(random_state=mu.SEMILLA).fit(Xo).mahalanobis(Xo)
    om = d2 > stats.chi2.ppf(0.999, df=len(cols_out))
    oi = IsolationForest(random_state=mu.SEMILLA).fit(Xo).predict(Xo) == -1
    res["outliers"] = {"mahalanobis": om.mean(), "isolation": oi.mean(), "ambos": (om & oi).mean(),
                       "tasa_out_mahal": mo[Y][om].mean(), "tasa_in_mahal": mo[Y][~om].mean(),
                       "envios_out": float(np.median(mo["n_envios8"][om])), "envios_in": float(np.median(mo["n_envios8"][~om]))}
    perfil = train.groupby(["mercado", "cod_mpio"]).agg(
        log_base8=("log_base8", "mean"), cv8=("cv8", "mean"), activas26=("activas26", "mean"), dist_km=("dist_km", "first"),
        n_alimentos8=("n_alimentos8", "mean"), sh_frutas=("sh_frutas", "mean"), sh_verduras=("sh_verduras", "mean"),
        sh_tuberculos=("sh_tuberculos", "mean"), tasa=(Y, "mean"), n=(Y, "size"), pdet=("pdet_origen", "first"))
    perfil = perfil[perfil["n"] >= 20]
    ck = ["log_base8", "cv8", "activas26", "n_alimentos8", "sh_frutas", "sh_verduras", "sh_tuberculos"]
    Xk = StandardScaler().fit_transform(np.column_stack([perfil[ck], np.log1p(perfil["dist_km"])]))
    sil = {k: silhouette_score(Xk, KMeans(k, n_init=10, random_state=mu.SEMILLA).fit_predict(Xk)) for k in range(2, 9)}
    kopt = max(sil, key=sil.get)
    perfil["cluster"] = KMeans(kopt, n_init=10, random_state=mu.SEMILLA).fit_predict(Xk)
    res["clusters"] = {"silueta": sil, "k": kopt,
                       "perfiles": perfil.groupby("cluster").agg(corredores=("n", "size"), tasa=("tasa", "mean"),
                                                                 pdet=("pdet", "mean"), cv8=("cv8", "median"),
                                                                 log_base8=("log_base8", "median"),
                                                                 dist_km=("dist_km", "median")).reset_index().to_dict("records")}

    # Temporal: STL, estacionariedad, ACF/PACF, autocorrelación intra-corredor, punto de cambio, eventos
    serie = tasa_sem.asfreq("W-MON").interpolate()
    stl = STL(serie, period=52, robust=True).fit()
    pd.DataFrame({"semana_t1": serie.index, "observado": stl.observed, "tendencia": stl.trend,
                  "estacional": stl.seasonal, "residuo": stl.resid}).to_parquet(DATA / "stl_tasa.parquet", index=False)
    var_r = stl.resid.var()
    adf = adfuller(serie, autolag="AIC")
    kp = kpss(serie, regression="c", nlags="auto")
    res["temporal"] = {
        "fuerza_tendencia": max(0, 1 - var_r / (stl.trend + stl.resid).var()),
        "fuerza_estacional": max(0, 1 - var_r / (stl.seasonal + stl.resid).var()),
        "adf_estadistico": adf[0], "adf_p": adf[1], "kpss_p": kp[1],
        "acf": acf(serie, nlags=60), "pacf": pacf(serie, nlags=30, method="ywm"),
        "banda_conf": 1.96 / np.sqrt(len(serie)),
    }
    ts = train.sort_values("semana")[["mercado", "cod_mpio", "semana", Y]]
    ac = []
    for k in [1, 2, 3, 4, 8, 13, 26, 52]:
        lag = ts.groupby(["mercado", "cod_mpio"])[Y].shift(k)
        gap = ts["semana"] - ts.groupby(["mercado", "cod_mpio"])["semana"].shift(k)
        ok = lag.notna() & (gap == pd.Timedelta(weeks=k))
        ac.append({"rezago": k, "correlacion": float(np.corrcoef(ts.loc[ok, Y], lag[ok])[0, 1]), "pares": int(ok.sum())})
    res["temporal"]["autocorr_corredor"] = ac
    x = tasa_sem.to_numpy()
    costos = [((x[:k] - x[:k].mean()) ** 2).sum() + ((x[k:] - x[k:].mean()) ** 2).sum() for k in range(26, len(x) - 26)]
    kc = int(np.argmin(costos)) + 26
    res["temporal"]["punto_cambio"] = {"fecha": tasa_sem.index[kc], "antes": x[:kc].mean(), "despues": x[kc:].mean()}
    eventos = {"Confinamiento COVID-19": ("2020-03-23", "2020-05-31"), "Paro nacional 2021": ("2021-04-28", "2021-06-15")}
    res["temporal"]["eventos"] = [{"evento": e, "inicio": a, "fin": b,
                                   "tasa": tasa_sem[(tasa_sem.index >= a) & (tasa_sem.index <= b)].mean(),
                                   "maximo": tasa_sem[(tasa_sem.index >= a) & (tasa_sem.index <= b)].max()}
                                  for e, (a, b) in eventos.items()]
    res["temporal"]["tasa_fuera_eventos"] = tasa_sem[~(((tasa_sem.index >= "2020-03-23") & (tasa_sem.index <= "2020-05-31")) |
                                                       ((tasa_sem.index >= "2021-04-28") & (tasa_sem.index <= "2021-06-15")))].mean()

    # Deriva: PSI entre 2019 y cada año
    def psi(a, b, bins=10):
        q = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
        pa = np.histogram(a, q)[0] / len(a) + 1e-6
        pb = np.histogram(np.clip(b, q[0], q[-1]), q)[0] / len(b) + 1e-6
        return float(((pa - pb) * np.log(pa / pb)).sum())
    ref = train[train["anio"] == 2019]
    deriva = []
    for c in [c for c in mu.NUMERICAS if c not in ("sem_sin", "sem_cos")]:
        for a in range(2020, 2025):
            deriva.append({"variable": c, "anio": a, "psi": psi(ref[c].dropna().values, train.loc[train["anio"] == a, c].dropna().values)})
    pd.DataFrame(deriva).to_parquet(DATA / "deriva_psi.parquet", index=False)

    # Espacial: agregados municipales, Moran, LISA, Gi*, semivariograma, MAUP, interacción espacio-tiempo
    mun = train.groupby("cod_mpio").agg(
        nombre=("mpio_origen", "first"), depto=("depto_origen", "first"), lat=("lat_o", "first"), lon=("lon_o", "first"),
        tasa=(Y, "mean"), n=(Y, "size"), pdet=("pdet_origen", "first"), cv8=("cv8", "mean"),
        corredores=("mercado", "nunique"), toneladas_dia=("base8", "sum"))
    mun["toneladas_dia"] = mun["toneladas_dia"] / 1e3 / train["semana"].nunique()
    mun["region"] = mun["depto"].map(mu.REGION)
    ok = mun["n"] >= 52
    m2 = mun[ok].copy()
    W = pesos_knn(m2["lat"].values, m2["lon"].values, 8)
    I8, p8 = moran_i(m2["tasa"].values, W)
    sens = {k: moran_i(m2["tasa"].values, pesos_knn(m2["lat"].values, m2["lon"].values, k), perm=199)[0] for k in (4, 12)}
    m2["lisa"], m2["z"], m2["rezago"] = lisa(m2["tasa"].values, W)
    m2["gi"] = gi_star(m2["tasa"].values, m2["lat"].values, m2["lon"].values)
    mun = mun.join(m2[["lisa", "z", "rezago", "gi"]])
    mun.reset_index().to_parquet(DATA / "municipios.parquet", index=False)
    D = haversine_distances(np.radians(m2[["lat", "lon"]].values)) * R
    dv = (m2["tasa"].values[:, None] - m2["tasa"].values[None, :]) ** 2 / 2
    iu = np.triu_indices(len(m2), 1)
    bordes = np.array([0, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800])
    cls = np.digitize(D[iu], bordes)
    gam = pd.Series(dv[iu]).groupby(cls).mean()
    semiv = [{"distancia": (bordes[i - 1] + bordes[i]) / 2, "semivarianza": float(gam[i])} for i in gam.index if i < len(bordes)]
    dep = train.groupby("depto_origen").agg(lat=("lat_o", "mean"), lon=("lon_o", "mean"), tasa=(Y, "mean"), n=(Y, "size"))
    dep = dep[dep["n"] >= 200]
    Id, pd_ = moran_i(dep["tasa"].values, pesos_knn(dep["lat"].values, dep["lon"].values, 4))
    por_anio, hot_prev = [], None
    for a in range(2018, 2025):
        dd = train[train["anio"] == a].groupby("cod_mpio").agg(lat=("lat_o", "first"), lon=("lon_o", "first"), tasa=(Y, "mean"), n=(Y, "size"))
        dd = dd[dd["n"] >= 20]
        Ia, pa = moran_i(dd["tasa"].values, pesos_knn(dd["lat"].values, dd["lon"].values, 8), perm=199)
        hot = set(dd.index[gi_star(dd["tasa"].values, dd["lat"].values, dd["lon"].values) > 1.96])
        por_anio.append({"anio": a, "municipios": len(dd), "moran": Ia, "p": pa, "hotspots": len(hot),
                         "jaccard": (len(hot & hot_prev) / len(hot | hot_prev)) if hot_prev else None})
        hot_prev = hot
    anual = train.groupby(["anio", "cod_mpio"])[Y].agg(tasa="mean", n="size").reset_index()
    anual[anual["n"] >= 20].to_parquet(DATA / "tasa_municipio_anio.parquet", index=False)
    reg_t = train.assign(trim=train["semana_t1"].dt.to_period("Q").dt.start_time).groupby(["trim", "region_origen"])[Y].mean().unstack()
    ev = train.groupby(["cod_mpio", "semana_t1"]).agg(tasa=(Y, "mean"), n=(Y, "size"), lat=("lat_o", "first"), lon=("lon_o", "first")).reset_index()
    ev = ev[(ev["n"] >= 3) & (ev["tasa"] >= 0.67)]
    ev = ev.sample(min(3000, len(ev)), random_state=mu.SEMILLA)
    Dsp = haversine_distances(np.radians(ev[["lat", "lon"]].values)) * R
    tt = ev["semana_t1"].values.astype("datetime64[D]").astype(int) / 7
    iu2 = np.triu_indices(len(ev), 1)
    cerca = Dsp[iu2] <= 50
    knox = lambda t: int((cerca & (np.abs(t[:, None] - t[None, :])[iu2] <= 1)).sum())  # noqa: E731
    obs = knox(tt)
    sim = np.array([knox(RNG.permutation(tt)) for _ in range(99)])
    res["espacial"] = {
        "moran_k8": I8, "moran_p": p8, "moran_k4": sens[4], "moran_k12": sens[12],
        "lisa_conteo": m2["lisa"].value_counts().to_dict(), "municipios_analizados": int(len(m2)),
        "semivariograma": semiv, "varianza_total": float(m2["tasa"].var()),
        "maup": {"municipal": I8, "departamental": Id, "p_departamental": pd_, "sd_municipal": float(m2["tasa"].std()), "sd_departamental": float(dep["tasa"].std())},
        "por_anio": por_anio, "correlacion_regional": reg_t.corr().round(3).to_dict(),
        "knox": {"observado": obs, "esperado": float(sim.mean()), "razon": obs / sim.mean(), "p": (np.sum(sim >= obs) + 1) / 100, "eventos": int(len(ev))},
        "tasa_region": train.groupby("region_origen")[Y].agg(["mean", "size"]).reset_index().to_dict("records"),
        "sin_cobertura": int(1122 - mun.shape[0]),
    }

    # GeoJSON municipal simplificado para los mapas del dashboard
    gj = json.loads((proyecto / "data/raw/municipios_geom.geojson").read_text())
    for f in gj["features"]:
        f["properties"] = {"cod": str(f["properties"]["mpcodigo"]).zfill(5), "nombre": f["properties"]["mpnombre"]}
        f["geometry"]["coordinates"] = json.loads(json.dumps(f["geometry"]["coordinates"]),
                                                  parse_float=lambda v: round(float(v), 3))
    (DATA / "municipios.geojson").write_text(json.dumps(gj, separators=(",", ":")))
    print(f"  EDA: {time.time() - t0:.0f} s")
    return res


# ------------------------------------------------------------------------------ modelos
def artefactos_modelos(train, test):
    t0 = time.time()
    res = {}
    Xtr, Xte = train[mu.X_COLS + ["semana_t1"]], test[mu.X_COLS + ["semana_t1"]]
    ytr, yte = train[Y].to_numpy(), test[Y].to_numpy()
    cv = mu.BloquesTemporales(n_splits=5, gap_semanas=8)
    res["pliegues"] = [{"pliegue": i + 1, "train_desde": train["semana_t1"].iloc[a].min(), "train_hasta": train["semana_t1"].iloc[a].max(),
                        "valid_desde": train["semana_t1"].iloc[b].min(), "valid_hasta": train["semana_t1"].iloc[b].max(),
                        "n_train": len(a), "n_valid": len(b)} for i, (a, b) in enumerate(cv.split(train))]

    modelos = {
        "Regresión logística": (Pipeline([("prep", mu.preprocesador()), ("clf", LogisticRegression(max_iter=3000, random_state=mu.SEMILLA))]),
                                {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0]}),
        "SVM lineal": (Pipeline([("prep", mu.preprocesador()), ("clf", LinearSVC(dual=False, max_iter=5000, random_state=mu.SEMILLA))]),
                       {"clf__C": [0.0001, 0.001, 0.01, 0.1, 1.0]}),
    }
    filas_cv, scores_te, ajustados = [], {}, {}
    for nombre, (pipe, grid) in modelos.items():
        t1 = time.time()
        gs = GridSearchCV(pipe, grid, cv=cv, scoring={"roc_auc": "roc_auc", "average_precision": "average_precision"},
                          refit="roc_auc", n_jobs=-1, return_train_score=True)
        gs.fit(Xtr, ytr)
        r = pd.DataFrame(gs.cv_results_)
        for _, fila in r.iterrows():
            for i in range(5):
                filas_cv.append({"modelo": nombre, "C": fila["param_clf__C"], "pliegue": i + 1,
                                 "auc_valid": fila[f"split{i}_test_roc_auc"], "auc_train": fila[f"split{i}_train_roc_auc"],
                                 "ap_valid": fila[f"split{i}_test_average_precision"]})
        C = gs.best_params_["clf__C"]
        # Predicciones fuera de pliegue para el umbral y la calibración (Platt) de la SVM
        oof_s, oof_y = [], []
        for a, b in cv.split(train):
            m = Pipeline([("prep", mu.preprocesador()), ("clf", type(pipe.named_steps["clf"])(**{**pipe.named_steps["clf"].get_params(), "C": C}))])
            m.fit(Xtr.iloc[a], ytr[a])
            oof_s.append(m.decision_function(Xtr.iloc[b]))
            oof_y.append(ytr[b])
        oof_s, oof_y = np.concatenate(oof_s), np.concatenate(oof_y)
        platt = LogisticRegression().fit(oof_s.reshape(-1, 1), oof_y)
        oof_p = platt.predict_proba(oof_s.reshape(-1, 1))[:, 1] if nombre == "SVM lineal" else 1 / (1 + np.exp(-oof_s))
        umbrales = np.round(np.linspace(0.05, 0.8, 76), 2)
        f1s = [f1_score(oof_y, oof_p >= t) for t in umbrales]
        umbral = float(umbrales[int(np.argmax(f1s))])
        best = gs.best_estimator_
        s_te = best.decision_function(Xte)
        p_te = platt.predict_proba(s_te.reshape(-1, 1))[:, 1] if nombre == "SVM lineal" else best.predict_proba(Xte)[:, 1]
        scores_te[nombre] = p_te
        ajustados[nombre] = best
        res[nombre] = {"C": C, "umbral": umbral, "f1_oof": max(f1s), "auc_cv": gs.best_score_,
                       "auc_cv_sd": float(r.loc[gs.best_index_, "std_test_roc_auc"]),
                       "curva_umbral": {"umbral": umbrales, "f1": f1s},
                       "platt": {"a": float(platt.coef_[0, 0]), "b": float(platt.intercept_[0])} if nombre == "SVM lineal" else None,
                       "segundos": time.time() - t1}
        coef = pd.Series(best.named_steps["clf"].coef_[0], index=nombres_limpios(best))
        coef.rename("coeficiente").rename_axis("variable").reset_index().assign(modelo=nombre) \
            .to_parquet(DATA / f"coeficientes_{'logistica' if nombre.startswith('Reg') else 'svm'}.parquet", index=False)
        print(f"  {nombre}: C={C}, AUC CV={gs.best_score_:.4f}, umbral={umbral}, {time.time() - t1:.0f} s")
    pd.DataFrame(filas_cv).to_parquet(DATA / "cv_resultados.parquet", index=False)

    # Líneas base
    dmf = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    dst = DummyClassifier(strategy="stratified", random_state=mu.SEMILLA).fit(Xtr, ytr)
    pred = test[["semana_t1", "mercado", "depto_origen", "cod_mpio", "grupo_dom", "pdet_origen", "lat_o", "lon_o", "ratio_t"]].copy()
    pred["region_origen"] = pred["depto_origen"].map(mu.REGION)
    pred["y"] = yte
    pred["p_logistica"] = scores_te["Regresión logística"]
    pred["p_svm"] = scores_te["SVM lineal"]
    pred["s_persistencia"] = -test["ratio_t"].to_numpy()
    pred["p_dummy_estrat"] = dst.predict_proba(Xte)[:, 1]
    pred["yhat_dummy_estrat"] = dst.predict(Xte)
    pred.to_parquet(DATA / "predicciones_test.parquet", index=False)
    res["Persistencia"] = {"umbral_regla": "ratio_t < 0,5"}
    res["prevalencia_train"] = float(ytr.mean())

    # Métricas en prueba con IC bootstrap por bloques de semana
    def decisiones(nombre):
        if nombre == "Regresión logística":
            return pred["p_logistica"].to_numpy(), (pred["p_logistica"] >= res[nombre]["umbral"]).astype(int).to_numpy()
        if nombre == "SVM lineal":
            return pred["p_svm"].to_numpy(), (pred["p_svm"] >= res[nombre]["umbral"]).astype(int).to_numpy()
        if nombre == "Persistencia":
            return pred["s_persistencia"].to_numpy(), (pred["ratio_t"] < 0.5).astype(int).to_numpy()
        if nombre == "Dummy (estratificado)":
            return pred["p_dummy_estrat"].to_numpy(), pred["yhat_dummy_estrat"].to_numpy()
        return dmf.predict_proba(Xte)[:, 1], dmf.predict(Xte)
    metricas = {"accuracy": lambda y, s, p: accuracy_score(y, p), "precision": lambda y, s, p: precision_score(y, p, zero_division=0),
                "recall": lambda y, s, p: recall_score(y, p), "F1": lambda y, s, p: f1_score(y, p),
                "AUC-ROC": lambda y, s, p: roc_auc_score(y, s), "AUC-PR": lambda y, s, p: average_precision_score(y, s)}
    grupos = pd.Series(np.arange(len(yte))).groupby(pred["semana_t1"].to_numpy()).apply(np.array).to_list()
    idx_boot = [np.concatenate([grupos[i] for i in RNG.integers(0, len(grupos), len(grupos))]) for _ in range(500)]
    filas = []
    nombres_modelos = ["Dummy (clase mayoritaria)", "Dummy (estratificado)", "Persistencia", "Regresión logística", "SVM lineal"]
    for nombre in nombres_modelos:
        s, p = decisiones(nombre)
        for mn, f in metricas.items():
            v = f(yte, s, p)
            boot = [f(yte[i], s[i], p[i]) for i in idx_boot] if mn not in ("AUC-ROC", "AUC-PR") or not nombre.startswith("Dummy (clase") else [v]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            filas.append({"modelo": nombre, "metrica": mn, "valor": v, "ic_inf": lo, "ic_sup": hi})
        if nombre in ("Regresión logística", "SVM lineal"):
            filas.append({"modelo": nombre, "metrica": "Brier", "valor": brier_score_loss(yte, s), "ic_inf": None, "ic_sup": None})
    filas.append({"modelo": "Constante (tasa de train)", "metrica": "Brier", "valor": brier_score_loss(yte, np.full(len(yte), ytr.mean())),
                  "ic_inf": None, "ic_sup": None})
    pd.DataFrame(filas).to_parquet(DATA / "metricas_test.parquet", index=False)

    # Comparaciones: DeLong, diferencia de AUC por bootstrap y McNemar
    comp = []
    pares = [("Regresión logística", "SVM lineal"), ("Regresión logística", "Persistencia"), ("SVM lineal", "Persistencia")]
    for a, b in pares:
        sa, pa_ = decisiones(a)
        sb, pb_ = decisiones(b)
        a1, a2, dauc, se, z, p = delong_rapido(yte, sa, sb)
        db = [roc_auc_score(yte[i], sa[i]) - roc_auc_score(yte[i], sb[i]) for i in idx_boot[:300]]
        ca, cb = pa_ == yte, pb_ == yte
        n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        chi = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if n01 + n10 else 0.0
        comp.append({"modelo_a": a, "modelo_b": b, "auc_a": a1, "auc_b": a2, "delta_auc": dauc, "se_delong": se, "z": z, "p_delong": p,
                     "ic_boot_inf": np.percentile(db, 2.5), "ic_boot_sup": np.percentile(db, 97.5),
                     "solo_a_acierta": n01, "solo_b_acierta": n10, "chi2_mcnemar": chi, "p_mcnemar": float(stats.chi2.sf(chi, 1))})
    comp = pd.DataFrame(comp)
    comp["p_delong_holm"] = multipletests(comp["p_delong"], method="holm")[1]
    comp.to_parquet(DATA / "comparaciones.parquet", index=False)

    # Curvas ROC, PR y calibración
    curvas = []
    for nombre in ["Regresión logística", "SVM lineal", "Persistencia", "Dummy (estratificado)"]:
        s, _ = decisiones(nombre)
        fpr, tpr, _ = roc_curve(yte, s)
        x, yv = submuestrear_curva(fpr, tpr)
        curvas += [{"modelo": nombre, "curva": "ROC", "x": a, "y": b} for a, b in zip(x, yv)]
        pr, rc, _ = precision_recall_curve(yte, s)
        x, yv = submuestrear_curva(rc[::-1], pr[::-1])
        curvas += [{"modelo": nombre, "curva": "PR", "x": a, "y": b} for a, b in zip(x, yv)]
    for nombre, col in [("Regresión logística", "p_logistica"), ("SVM lineal", "p_svm")]:
        fr, mp = calibration_curve(yte, pred[col], n_bins=15, strategy="quantile")
        curvas += [{"modelo": nombre, "curva": "Calibración", "x": a, "y": b} for a, b in zip(mp, fr)]
    pd.DataFrame(curvas).to_parquet(DATA / "curvas.parquet", index=False)

    # Validación por bloques espaciales (departamento de origen)
    esp = []
    for nombre, best in ajustados.items():
        for i, (a, b) in enumerate(GroupKFold(n_splits=5).split(Xtr, ytr, groups=train["depto_origen"])):
            m = Pipeline([("prep", mu.preprocesador()), ("clf", best.named_steps["clf"].__class__(**best.named_steps["clf"].get_params()))])
            m.fit(Xtr.iloc[a], ytr[a])
            s = m.decision_function(Xtr.iloc[b])
            esp.append({"modelo": nombre, "pliegue": i + 1, "departamentos": ", ".join(sorted(train["depto_origen"].iloc[b].unique())),
                        "n": len(b), "auc": roc_auc_score(ytr[b], s), "ap": average_precision_score(ytr[b], s), "prevalencia": ytr[b].mean()})
    pd.DataFrame(esp).to_parquet(DATA / "cv_espacial.parquet", index=False)

    # Curva de aprendizaje (último pliegue temporal como validación)
    a, b = list(cv.split(train))[-1]
    lc = []
    for nombre, best in ajustados.items():
        tam, s_tr, s_va = learning_curve(
            Pipeline([("prep", mu.preprocesador()), ("clf", best.named_steps["clf"].__class__(**best.named_steps["clf"].get_params()))]),
            Xtr, ytr, cv=[(a, b)], train_sizes=[0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0], scoring="roc_auc", shuffle=False, n_jobs=-1)
        lc += [{"modelo": nombre, "filas": int(t), "auc_train": float(x1), "auc_valid": float(x2)} for t, x1, x2 in zip(tam, s_tr.mean(1), s_va.mean(1))]
    pd.DataFrame(lc).to_parquet(DATA / "curva_aprendizaje.parquet", index=False)

    # Diagnóstico de residuos de la logística
    p = pred["p_logistica"].clip(1e-6, 1 - 1e-6)
    r = (yte - p) / np.sqrt(p * (1 - p))
    r_sem = pd.Series(r.to_numpy(), index=pred["semana_t1"]).groupby(level=0).mean()
    lb = acorr_ljungbox(r_sem, lags=[1, 4, 8], return_df=True)
    rm = pd.DataFrame({"r": r.to_numpy(), "y": yte, "cod_mpio": pred["cod_mpio"], "lat": pred["lat_o"], "lon": pred["lon_o"]}) \
        .groupby("cod_mpio").agg(r=("r", "mean"), y=("y", "mean"), n=("r", "size"), lat=("lat", "first"), lon=("lon", "first"))
    rm = rm[rm["n"] >= 20]
    Wr = pesos_knn(rm["lat"].values, rm["lon"].values, 8)
    Ir, pr_ = moran_i(rm["r"].values, Wr)
    Iy, py_ = moran_i(rm["y"].values, Wr)
    rm.reset_index().to_parquet(DATA / "residuos_municipio.parquet", index=False)
    pd.DataFrame({"semana_t1": r_sem.index, "residuo": r_sem.values}).to_parquet(DATA / "residuos_semana.parquet", index=False)
    res["residuos"] = {"ljung_box": lb.reset_index(names="rezago").to_dict("records"), "moran_residuo": Ir, "p_residuo": pr_,
                       "moran_tasa_test": Iy, "p_tasa_test": py_, "acf_residuo": acf(r_sem, nlags=20), "banda": 1.96 / np.sqrt(len(r_sem))}

    # Dependencia parcial de ratio_t y perfil base para el simulador
    base = Xtr.sample(3000, random_state=mu.SEMILLA).copy()
    malla = np.round(np.linspace(0, 3.5, 36), 2)
    pdp = []
    for nombre, best in ajustados.items():
        for v in malla:
            base["ratio_t"] = v
            s = best.decision_function(base)
            pv = 1 / (1 + np.exp(-s)) if nombre.startswith("Reg") else 1 / (1 + np.exp(-(res[nombre]["platt"]["a"] * s + res[nombre]["platt"]["b"])))
            pdp.append({"modelo": nombre, "ratio_t": v, "prob": float(pv.mean())})
    pd.DataFrame(pdp).to_parquet(DATA / "dependencia_ratio_t.parquet", index=False)
    perfil = {c: float(train[c].median()) for c in mu.NUMERICAS + mu.BINARIAS}
    perfil.update({c: train[c].mode().iloc[0] for c in mu.CATEGORICAS})
    res["perfil_base"] = perfil
    res["rangos_simulador"] = {c: [float(train[c].quantile(0.01)), float(train[c].quantile(0.99))] for c in mu.NUMERICAS}
    res["categorias"] = {c: sorted(train[c].dropna().unique().tolist()) for c in mu.CATEGORICAS}
    joblib.dump(ajustados["Regresión logística"], DATA / "modelo_logistico.joblib", compress=3)
    print(f"  modelos: {time.time() - t0:.0f} s")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proyecto", type=Path, default=ROOT.parent / "proyecto_sipsa",
                    help="Carpeta del Entregable 1 (contiene data/processed/panel_corredores.parquet)")
    ap.add_argument("--partes", nargs="+", default=["contexto", "eda", "modelos"],
                    choices=["contexto", "eda", "modelos"], help="Regenerar solo algunas partes (las demás se conservan)")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    panel = pd.read_parquet(args.proyecto / "data/processed/panel_corredores.parquet")
    train, test = mu.particion(panel)
    print(f"Panel {panel.shape}; train {len(train):,}; test {len(test):,}")
    previo = DATA / "resumen.json"
    resumen = json.loads(previo.read_text(encoding="utf-8")) if previo.exists() else {}
    if "contexto" in args.partes:
        resumen["kpis"] = artefactos_contexto(panel, args.proyecto)
    if "eda" in args.partes:
        resumen["eda"] = artefactos_eda(train, test, args.proyecto)
    if "modelos" in args.partes:
        resumen["modelos"] = artefactos_modelos(train, test)
    guardar_json(resumen, "resumen.json")
    print("Listo:", sorted(p.name for p in DATA.iterdir()))


if __name__ == "__main__":
    main()
