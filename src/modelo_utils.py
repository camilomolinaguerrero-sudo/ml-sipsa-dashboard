"""Definiciones compartidas por el script de artefactos, el dashboard y el Jupyter Book.

Reproduce la partición, las variables y el preprocesamiento del Entregable 1, de modo que los
modelos base del Entregable 2 son comparables con los ya reportados. El dashboard importa este
módulo para poder cargar el pipeline serializado (contiene la clase Winsorizador).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, SplineTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SEMILLA = 42
VAR_OBJETIVO = "interrupcion_t1"

# Partición cronológica del Entregable 1: entrenamiento con semanas objetivo hasta el 27/10/2024,
# prueba con las semanas objetivo de 2025 y un hueco de 10 semanas entre ambos.
FIN_TRAIN = pd.Timestamp("2024-10-27")
INICIO_TEST = pd.Timestamp("2025-01-06")

NUMERICAS = ["log_base8", "ratio_t", "ratio_t_1", "cv8", "pend8_rel", "activas8", "activas26",
             "n_envios8", "n_alimentos8", "antig_sem", "dist_km", "frac_caida_mercado_t",
             "frac_caida_nacional_t", "festivos_t1", "sem_sin", "sem_cos",
             "sh_verduras", "sh_frutas", "sh_tuberculos", "sh_granos", "sh_lacteos",
             "sh_carnes", "sh_pescados", "sh_procesados"]
BINARIAS = ["mismo_depto", "local", "pdet_origen"]
CATEGORICAS = ["mercado", "depto_origen", "grupo_dom"]

LOG = ["n_envios8", "n_alimentos8", "dist_km", "antig_sem"]
SPL = ["ratio_t", "ratio_t_1"]
NUM = [c for c in NUMERICAS if c not in LOG + SPL + ["sh_procesados"]]
X_COLS = LOG + SPL + NUM + BINARIAS + CATEGORICAS

REGION = {
    "AMAZONAS": "Amazonía", "CAQUETÁ": "Amazonía", "GUAINÍA": "Amazonía", "GUAVIARE": "Amazonía",
    "PUTUMAYO": "Amazonía", "VAUPÉS": "Amazonía",
    "ANTIOQUIA": "Andina", "BOYACÁ": "Andina", "CALDAS": "Andina", "CUNDINAMARCA": "Andina",
    "BOGOTÁ, D.C.": "Andina", "HUILA": "Andina", "NORTE DE SANTANDER": "Andina", "QUINDÍO": "Andina",
    "RISARALDA": "Andina", "SANTANDER": "Andina", "TOLIMA": "Andina",
    "ATLÁNTICO": "Caribe", "BOLÍVAR": "Caribe", "CESAR": "Caribe", "CÓRDOBA": "Caribe",
    "LA GUAJIRA": "Caribe", "MAGDALENA": "Caribe", "SUCRE": "Caribe",
    "ARCHIPIÉLAGO DE SAN ANDRÉS, PROVIDENCIA Y SANTA CATALINA": "Caribe",
    "ARAUCA": "Orinoquía", "CASANARE": "Orinoquía", "META": "Orinoquía", "VICHADA": "Orinoquía",
    "CAUCA": "Pacífica", "CHOCÓ": "Pacífica", "NARIÑO": "Pacífica", "VALLE DEL CAUCA": "Pacífica",
}

# Nombres legibles de las variables para el dashboard y el libro
ETIQUETAS = {
    "log_base8": "Escala del corredor (log t/día)", "ratio_t": "Abastecimiento en t / promedio 8 sem.",
    "ratio_t_1": "Abastecimiento en t-1 / promedio 8 sem.", "cv8": "Coeficiente de variación (8 sem.)",
    "pend8_rel": "Tendencia relativa (8 sem.)", "activas8": "Semanas activas (de 8)",
    "activas26": "Fracción de semanas activas (26)", "n_envios8": "Cargas por semana (prom. 8)",
    "n_alimentos8": "Productos por semana (prom. 8)", "antig_sem": "Antigüedad del corredor (semanas)",
    "dist_km": "Distancia origen-central (km)", "frac_caida_mercado_t": "Fracción en caída en la central (t)",
    "frac_caida_nacional_t": "Fracción en caída en el país (t)", "festivos_t1": "Festivos en t+1",
    "sem_sin": "Semana del año (seno)", "sem_cos": "Semana del año (coseno)",
    "sh_verduras": "Participación verduras", "sh_frutas": "Participación frutas",
    "sh_tuberculos": "Participación tubérculos", "sh_granos": "Participación granos",
    "sh_lacteos": "Participación lácteos", "sh_carnes": "Participación carnes",
    "sh_pescados": "Participación pescados", "sh_procesados": "Participación procesados",
    "mismo_depto": "Mismo departamento", "local": "Corredor local", "pdet_origen": "Origen PDET",
    "mercado": "Central mayorista", "depto_origen": "Departamento de origen", "grupo_dom": "Grupo dominante",
}


class Winsorizador(BaseEstimator, TransformerMixin):
    """Recorta cada columna en los cuantiles [inferior, superior] aprendidos en fit (solo train)."""

    def __init__(self, inferior=0.01, superior=0.99):
        self.inferior = inferior
        self.superior = superior

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.lim_inf_ = np.nanquantile(X, self.inferior, axis=0)
        self.lim_sup_ = np.nanquantile(X, self.superior, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, float), self.lim_inf_, self.lim_sup_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class BloquesTemporales:
    """Validación cruzada de ventana creciente sobre semanas completas, con hueco entre
    entrenamiento y validación. Requiere la columna `semana_t1` en X."""

    def __init__(self, n_splits=5, gap_semanas=8, columna="semana_t1"):
        self.n_splits = n_splits
        self.gap_semanas = gap_semanas
        self.columna = columna

    def split(self, X, y=None, groups=None):
        semanas = pd.to_datetime(X[self.columna]).to_numpy()
        unicas = np.unique(semanas)
        tss = TimeSeriesSplit(n_splits=self.n_splits, gap=self.gap_semanas)
        for tr_w, va_w in tss.split(unicas):
            yield (np.flatnonzero(np.isin(semanas, unicas[tr_w])),
                   np.flatnonzero(np.isin(semanas, unicas[va_w])))

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def preprocesador():
    """ColumnTransformer del Entregable 1 (cada paso se justifica en el EDA)."""
    return ColumnTransformer([
        ("log", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                          ("win", Winsorizador()), ("esc", StandardScaler())]), LOG),
        ("spl", Pipeline([("imp", SimpleImputer(strategy="median")), ("win", Winsorizador()),
                          ("spl", SplineTransformer(n_knots=5, degree=3, knots="quantile")),
                          ("esc", StandardScaler())]), SPL),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("win", Winsorizador()),
                          ("esc", StandardScaler())]), NUM),
        ("bin", "passthrough", BINARIAS),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01), CATEGORICAS),
    ])


def particion(p: pd.DataFrame):
    """(train, test) según la semana objetivo t+1."""
    return p[p["semana_t1"] <= FIN_TRAIN].copy(), p[p["semana_t1"] >= INICIO_TEST].copy()


# ---------------------------------------------------------------- modelos adicionales (Entregable 2, ampliación)
from sklearn.base import ClassifierMixin  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

# Nombre del modelo -> columna de probabilidades en data/predicciones_test.parquet
COLUMNA_MODELO = {
    "Regresión logística": "p_logistica", "SVM lineal": "p_svm", "Lasso (L1)": "p_lasso",
    "k-NN": "p_knn", "Naive Bayes": "p_nb", "Random Forest": "p_rf", "XGBoost": "p_xgb",
    "SVM RBF (Nyström)": "p_svm_rbf", "PCA + logística": "p_pca_log", "Logística por segmentos": "p_segmentos",
}


def preprocesador_denso():
    """Mismo preprocesamiento, con salida densa (Naive Bayes, PCA y Nyström no aceptan matrices dispersas)."""
    ct = preprocesador()
    ct.set_params(sparse_threshold=0)
    return ct


class LogisticaSegmentada(BaseEstimator, ClassifierMixin):
    """Agrupa las filas con k-means según el perfil del corredor (escala y regularidad, calculados con
    información hasta t) y ajusta una regresión logística con el pipeline completo en cada segmento."""

    def __init__(self, k=2, C=0.01, columnas=("log_base8", "cv8", "activas26", "n_alimentos8", "dist_km"),
                 n_kmeans=60000, random_state=SEMILLA):
        self.k = k
        self.C = C
        self.columnas = columnas
        self.n_kmeans = n_kmeans
        self.random_state = random_state

    def _z(self, X, ajustar=False):
        Z = X[list(self.columnas)].astype(float).copy()
        for c in ("n_alimentos8", "dist_km"):
            if c in Z:
                Z[c] = np.log1p(Z[c].clip(lower=0))
        if ajustar:
            self.mediana_ = Z.median()
            Z = Z.fillna(self.mediana_)
            self.media_, self.desv_ = Z.mean(), Z.std().replace(0, 1)
        Z = Z.fillna(self.mediana_)
        return ((Z - self.media_) / self.desv_).to_numpy()

    def fit(self, X, y):
        y = np.asarray(y)
        Z = self._z(X, ajustar=True)
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(Z), min(self.n_kmeans, len(Z)), replace=False)
        self.km_ = KMeans(self.k, n_init=10, random_state=self.random_state).fit(Z[idx])
        etiquetas = self.km_.predict(Z)
        self.modelos_ = {}
        for g in range(self.k):
            m = etiquetas == g
            self.modelos_[g] = Pipeline([("prep", preprocesador()),
                                         ("clf", LogisticRegression(C=self.C, max_iter=3000, random_state=self.random_state))]
                                        ).fit(X[m], y[m])
        self.classes_ = np.array([0, 1])
        return self

    def segmento(self, X):
        return self.km_.predict(self._z(X))

    def predict_proba(self, X):
        etiquetas = self.segmento(X)
        p = np.empty(len(X))
        for g, modelo in self.modelos_.items():
            m = etiquetas == g
            if m.any():
                p[m] = modelo.predict_proba(X[m])[:, 1]
        return np.column_stack([1 - p, p])

    def decision_function(self, X):
        p = np.clip(self.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
        return np.log(p / (1 - p))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
