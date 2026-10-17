"""
Modelos adicionales del curso sobre el mismo panel, la misma partición y la misma validación.

Se suman a la regresión logística y la SVM lineal (preparar_artefactos.py), sin modificar sus resultados:
Lasso (L1), k-NN, Naive Bayes, Random Forest, XGBoost, SVM RBF (aproximación de Nyström), PCA + logística
y logística por segmentos (k-means). Para cada modelo:

1. Búsqueda de hiperparámetros con validación cruzada por bloques de semanas (hueco de 8 semanas).
2. Umbral de decisión que maximiza F1 en las predicciones fuera de pliegue; calibración de Platt para los
   modelos que no entregan probabilidades.
3. Ajuste final con todo el entrenamiento y evaluación una sola vez en la prueba (2025).
4. Validación por departamentos no vistos (GroupKFold) e importancia por permutación.

Cada modelo se guarda en data/modelos_adicionales/<slug>.joblib; si el script se interrumpe, al volver a
ejecutarlo omite los modelos ya terminados. Al final integra todo en los artefactos que usan el dashboard y
el libro (metricas_test, comparaciones, curvas, cv_resultados, cv_espacial, predicciones_test, resumen.json).

Uso:
    python scripts/modelos_adicionales.py --proyecto "../proyecto_sipsa"            # todos
    python scripts/modelos_adicionales.py --solo xgb rf                            # algunos
    python scripts/modelos_adicionales.py --consolidar                             # solo integrar
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from statsmodels.stats.multitest import multipletests
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
import modelo_utils as mu  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
from preparar_artefactos import delong_rapido, guardar_json, submuestrear_curva  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DATA = ROOT / "data"
DIR_MOD = DATA / "modelos_adicionales"
Y = mu.VAR_OBJETIVO
SEM = mu.SEMILLA


def _sub(idx, n, semilla=SEM):
    """Submuestra aleatoria (reproducible) de índices de entrenamiento, para modelos costosos."""
    if n is None or len(idx) <= n:
        return idx
    return np.sort(np.random.default_rng(semilla).choice(idx, n, replace=False))


# ------------------------------------------------------------------------------ catálogo de modelos
def catalogo():
    pre = mu.preprocesador
    den = mu.preprocesador_denso
    return {
        "lasso": dict(
            nombre="Lasso (L1)", proba=True, n_sub=None, jobs_pliegues=5, aprendizaje=False,
            grid=[{"C": c} for c in (0.0005, 0.002, 0.01, 0.05)],
            fabrica=lambda p: Pipeline([("prep", pre()), ("clf", LogisticRegression(
                C=p["C"], l1_ratio=1, solver="liblinear", max_iter=2000, random_state=SEM))]),
            etiqueta=lambda p: f"C={p['C']:g}"),
        "knn": dict(
            nombre="k-NN", proba=True, n_sub=100_000, jobs_pliegues=1, aprendizaje=False, n_perm=5000,
            grid=[{"k": k} for k in (50, 150, 300)],
            fabrica=lambda p: Pipeline([("prep", den()), ("clf", KNeighborsClassifier(
                n_neighbors=p["k"], algorithm="brute", n_jobs=-1))]),
            etiqueta=lambda p: f"k={p['k']}"),
        "nb": dict(
            nombre="Naive Bayes", proba=True, n_sub=None, jobs_pliegues=5, aprendizaje=False,
            grid=[{"vs": v} for v in (1e-9, 1e-5, 1e-2)],
            fabrica=lambda p: Pipeline([("prep", den()), ("clf", GaussianNB(var_smoothing=p["vs"]))]),
            etiqueta=lambda p: f"var_smoothing={p['vs']:g}"),
        "rf": dict(
            nombre="Random Forest", proba=True, n_sub=None, jobs_pliegues=1, aprendizaje=True,
            grid=[{"hoja": h} for h in (20, 100)],
            fabrica=lambda p: Pipeline([("prep", den()), ("clf", RandomForestClassifier(
                n_estimators=150, min_samples_leaf=p["hoja"], max_features="sqrt", max_samples=0.25,
                n_jobs=-1, random_state=SEM))]),
            etiqueta=lambda p: f"min_samples_leaf={p['hoja']}"),
        "xgb": dict(
            nombre="XGBoost", proba=True, n_sub=None, jobs_pliegues=1, aprendizaje=True,
            grid=[{"lr": lr, "prof": d} for lr in (0.05, 0.1) for d in (4, 6)],
            fabrica=lambda p: Pipeline([("prep", den()), ("clf", XGBClassifier(
                n_estimators=400, learning_rate=p["lr"], max_depth=p["prof"], subsample=0.8,
                colsample_bytree=0.8, min_child_weight=50, tree_method="hist", n_jobs=10,
                random_state=SEM, eval_metric="auc"))]),
            etiqueta=lambda p: f"lr={p['lr']:g}, prof={p['prof']}"),
        "svm_rbf": dict(
            nombre="SVM RBF (Nyström)", proba=False, n_sub=150_000, jobs_pliegues=3, aprendizaje=False,
            grid=[{"gamma": g, "C": c} for g in (0.005, 0.02) for c in (0.01, 0.1)],
            fabrica=lambda p: Pipeline([("prep", den()),
                                        ("nys", Nystroem(kernel="rbf", gamma=p["gamma"], n_components=300, random_state=SEM)),
                                        ("clf", LinearSVC(C=p["C"], dual=False, max_iter=5000, random_state=SEM))]),
            etiqueta=lambda p: f"gamma={p['gamma']:g}, C={p['C']:g}"),
        "pca_log": dict(
            nombre="PCA + logística", proba=True, n_sub=None, jobs_pliegues=5, aprendizaje=False,
            grid=[{"n": n} for n in (10, 20, 40)],
            fabrica=lambda p: Pipeline([("prep", den()), ("pca", PCA(n_components=p["n"], random_state=SEM)),
                                        ("clf", LogisticRegression(C=0.01, max_iter=3000, random_state=SEM))]),
            etiqueta=lambda p: f"componentes={p['n']}"),
        "segmentos": dict(
            nombre="Logística por segmentos", proba=True, n_sub=None, jobs_pliegues=5, aprendizaje=False,
            grid=[{"k": k} for k in (2, 3)],
            fabrica=lambda p: mu.LogisticaSegmentada(k=p["k"], C=0.01),
            etiqueta=lambda p: f"segmentos={p['k']}"),
    }


def puntajes(est, X, proba):
    return est.predict_proba(X)[:, 1] if proba else est.decision_function(X)


# ------------------------------------------------------------------------------ entrenamiento de un modelo
def entrenar(slug, cfg, train, test):
    t0 = time.time()
    Xtr, Xte = train[mu.X_COLS], test[mu.X_COLS]
    ytr, yte = train[Y].to_numpy(), test[Y].to_numpy()
    splits = list(mu.BloquesTemporales(n_splits=5, gap_semanas=8).split(train))
    proba, n_sub = cfg["proba"], cfg["n_sub"]

    def un_pliegue(params, i, a, b):
        a2 = _sub(a, n_sub, SEM + i)
        est = cfg["fabrica"](params).fit(Xtr.iloc[a2], ytr[a2])
        s_va = puntajes(est, Xtr.iloc[b], proba)
        muestra_tr = _sub(a2, 50_000, SEM + 100 + i)   # AUC de entrenamiento sobre una muestra (control de sobreajuste)
        s_tr = puntajes(est, Xtr.iloc[muestra_tr], proba)
        return i, s_va, roc_auc_score(ytr[muestra_tr], s_tr)

    filas, oof = [], {}
    for ci, params in enumerate(cfg["grid"]):
        t1 = time.time()
        res = Parallel(n_jobs=cfg["jobs_pliegues"])(delayed(un_pliegue)(params, i, a, b) for i, (a, b) in enumerate(splits))
        s_all = []
        for i, s_va, auc_tr in sorted(res, key=lambda r: r[0]):
            b = splits[i][1]
            filas.append({"modelo": cfg["nombre"], "params": cfg["etiqueta"](params), "pliegue": i + 1,
                          "auc_valid": roc_auc_score(ytr[b], s_va), "ap_valid": average_precision_score(ytr[b], s_va),
                          "auc_train": auc_tr})
            s_all.append(s_va)
        oof[ci] = np.concatenate(s_all)
        auc_m = np.mean([f["auc_valid"] for f in filas if f["params"] == cfg["etiqueta"](params)])
        print(f"    {cfg['nombre']} {cfg['etiqueta'](params)}: AUC validación {auc_m:.4f} ({time.time() - t1:.0f} s)", flush=True)
    cvdf = pd.DataFrame(filas)
    medias = cvdf.groupby("params")["auc_valid"].mean()
    mejor_etq = medias.idxmax()
    ci_best = [cfg["etiqueta"](p) for p in cfg["grid"]].index(mejor_etq)
    best = cfg["grid"][ci_best]
    oof_s = oof[ci_best]
    oof_y = np.concatenate([ytr[b] for _, b in splits])

    # Calibración (Platt) para modelos sin probabilidades y umbral por F1 fuera de pliegue
    platt = None
    if not proba:
        lr = LogisticRegression().fit(oof_s.reshape(-1, 1), oof_y)
        platt = {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}
    a_prob = (lambda s: 1 / (1 + np.exp(-(platt["a"] * s + platt["b"])))) if platt else (lambda s: s)
    oof_p = a_prob(oof_s)
    umbrales = np.round(np.linspace(0.05, 0.8, 76), 2)
    f1s = [f1_score(oof_y, oof_p >= t) for t in umbrales]
    umbral = float(umbrales[int(np.argmax(f1s))])

    # Ajuste final y prueba
    idx_final = _sub(np.arange(len(train)), n_sub, SEM + 999)
    final = cfg["fabrica"](best).fit(Xtr.iloc[idx_final], ytr[idx_final])
    p_te = a_prob(puntajes(final, Xte, proba))

    # Departamentos no vistos
    esp = []
    for i, (a, b) in enumerate(GroupKFold(n_splits=5).split(Xtr, ytr, groups=train["depto_origen"])):
        a2 = _sub(a, n_sub, SEM + 200 + i)
        est = cfg["fabrica"](best).fit(Xtr.iloc[a2], ytr[a2])
        s = puntajes(est, Xtr.iloc[b], proba)
        esp.append({"modelo": cfg["nombre"], "pliegue": i + 1,
                    "departamentos": ", ".join(sorted(train["depto_origen"].iloc[b].unique())), "n": len(b),
                    "auc": roc_auc_score(ytr[b], s), "ap": average_precision_score(ytr[b], s), "prevalencia": ytr[b].mean()})

    # Curva de aprendizaje (solo ensambles de árboles; los lineales ya la tienen)
    lc = []
    if cfg["aprendizaje"]:
        a, b = splits[-1]
        tam, s_tr, s_va = learning_curve(cfg["fabrica"](best), Xtr, ytr, cv=[(a, b)],
                                         train_sizes=[0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0],
                                         scoring="roc_auc", shuffle=False, n_jobs=1)
        lc = [{"modelo": cfg["nombre"], "filas": int(t), "auc_train": float(x1), "auc_valid": float(x2)}
              for t, x1, x2 in zip(tam, s_tr.mean(1), s_va.mean(1))]

    # Importancia por permutación (caída del AUC al barajar cada variable), sobre una muestra de prueba
    n_perm = cfg.get("n_perm", 20_000)
    ip = np.random.default_rng(SEM).choice(len(test), n_perm, replace=False)
    pi = permutation_importance(final, Xte.iloc[ip], yte[ip], scoring="roc_auc", n_repeats=3,
                                random_state=SEM, n_jobs=1 if slug in ("knn", "xgb", "rf") else 4)
    imp = pd.DataFrame({"modelo": cfg["nombre"], "variable": mu.X_COLS, "importancia": pi.importances_mean,
                        "sd": pi.importances_std})

    extra = {}
    if slug == "lasso":
        coef = final.named_steps["clf"].coef_[0]
        nombres = final.named_steps["prep"].get_feature_names_out()
        extra["coef_no_nulos"] = int((coef != 0).sum())
        extra["coef_total"] = int(len(coef))
        extra["coeficientes"] = pd.DataFrame({"variable": pd.Index(nombres).str.replace(r"^(log|spl|num|bin|cat)__", "", regex=True),
                                              "coeficiente": coef})
    if slug == "segmentos":
        seg = final.segmento(Xtr)
        extra["segmentos"] = pd.DataFrame({"segmento": seg, "y": ytr, "cv8": Xtr["cv8"].to_numpy(),
                                           "n_envios8": Xtr["n_envios8"].to_numpy()}).groupby("segmento").agg(
            filas=("y", "size"), tasa=("y", "mean"), cv8=("cv8", "median"), n_envios8=("n_envios8", "median")).reset_index()

    res = {"slug": slug, "nombre": cfg["nombre"], "params": mejor_etq, "umbral": umbral, "f1_oof": max(f1s),
           "auc_cv": float(medias.max()), "auc_cv_sd": float(cvdf[cvdf["params"] == mejor_etq]["auc_valid"].std()),
           "curva_umbral": {"umbral": umbrales.tolist(), "f1": [float(x) for x in f1s]}, "platt": platt,
           "n_entrenamiento": int(len(idx_final)), "p_test": p_te, "cv": cvdf, "espacial": pd.DataFrame(esp),
           "aprendizaje": pd.DataFrame(lc), "importancia": imp, "extra": extra, "segundos": time.time() - t0}
    print(f"  {cfg['nombre']}: {mejor_etq}, AUC CV {res['auc_cv']:.4f}, AUC prueba {roc_auc_score(yte, p_te):.4f}, "
          f"umbral {umbral}, {res['segundos'] / 60:.1f} min", flush=True)
    return res


def importancia_lineales(train, test):
    """Importancia por permutación de la logística y la SVM lineal (para comparar con los demás modelos)."""
    Xtr, Xte = train[mu.X_COLS], test[mu.X_COLS]
    ytr, yte = train[Y].to_numpy(), test[Y].to_numpy()
    r = json.loads((DATA / "resumen.json").read_text(encoding="utf-8"))["modelos"]
    logit = joblib.load(DATA / "modelo_logistico.joblib")
    svm = Pipeline([("prep", mu.preprocesador()), ("clf", LinearSVC(C=r["SVM lineal"]["C"], dual=False, max_iter=5000,
                                                                   random_state=SEM))]).fit(Xtr, ytr)
    ip = np.random.default_rng(SEM).choice(len(test), 20_000, replace=False)
    partes = []
    for nombre, est in [("Regresión logística", logit), ("SVM lineal", svm)]:
        pi = permutation_importance(est, Xte.iloc[ip], yte[ip], scoring="roc_auc", n_repeats=3, random_state=SEM, n_jobs=4)
        partes.append(pd.DataFrame({"modelo": nombre, "variable": mu.X_COLS, "importancia": pi.importances_mean,
                                    "sd": pi.importances_std}))
    return pd.concat(partes)


# ------------------------------------------------------------------------------ integración en los artefactos
def consolidar(test):
    r = json.loads((DATA / "resumen.json").read_text(encoding="utf-8"))
    pred = pd.read_parquet(DATA / "predicciones_test.parquet")
    yte = pred["y"].to_numpy()
    assert len(pred) == len(test) and (pred["semana_t1"].to_numpy() == test["semana_t1"].to_numpy()).all()
    res = {s: joblib.load(DIR_MOD / f"{s}.joblib") for s in catalogo() if (DIR_MOD / f"{s}.joblib").exists()}
    print("Integrando:", ", ".join(v["nombre"] for v in res.values()))

    # Predicciones
    for s, v in res.items():
        pred[mu.COLUMNA_MODELO[v["nombre"]]] = v["p_test"]
    pred.to_parquet(DATA / "predicciones_test.parquet", index=False)

    # Métricas con IC bootstrap por bloques de semana (los mismos remuestreos para todos los modelos nuevos)
    rng = np.random.default_rng(SEM)
    grupos = pd.Series(np.arange(len(yte))).groupby(pred["semana_t1"].to_numpy()).apply(np.array).to_list()
    idx_boot = [np.concatenate([grupos[i] for i in rng.integers(0, len(grupos), len(grupos))]) for _ in range(500)]
    metricas = {"accuracy": lambda y, s, p: accuracy_score(y, p), "precision": lambda y, s, p: precision_score(y, p, zero_division=0),
                "recall": lambda y, s, p: recall_score(y, p), "F1": lambda y, s, p: f1_score(y, p),
                "AUC-ROC": lambda y, s, p: roc_auc_score(y, s), "AUC-PR": lambda y, s, p: average_precision_score(y, s)}
    met = pd.read_parquet(DATA / "metricas_test.parquet")
    nuevos = [v["nombre"] for v in res.values()]
    met = met[~met["modelo"].isin(nuevos)]
    filas = []
    for v in res.values():
        s = v["p_test"]
        p = (s >= v["umbral"]).astype(int)
        for mn, f in metricas.items():
            boot = [f(yte[i], s[i], p[i]) for i in idx_boot]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            filas.append({"modelo": v["nombre"], "metrica": mn, "valor": f(yte, s, p), "ic_inf": lo, "ic_sup": hi})
        filas.append({"modelo": v["nombre"], "metrica": "Brier", "valor": brier_score_loss(yte, s), "ic_inf": np.nan, "ic_sup": np.nan})
    pd.concat([met, pd.DataFrame(filas)], ignore_index=True).to_parquet(DATA / "metricas_test.parquet", index=False)

    # Comparaciones con la regresión logística: DeLong, bootstrap pareado de la diferencia de AUC y McNemar
    comp = pd.read_parquet(DATA / "comparaciones.parquet")
    comp = comp[~comp["modelo_b"].isin(nuevos) & ~comp["modelo_a"].isin(nuevos)]
    s_log = pred["p_logistica"].to_numpy()
    d_log = (s_log >= r["modelos"]["Regresión logística"]["umbral"]).astype(int)
    filas = []
    for v in res.values():
        s = v["p_test"]
        d = (s >= v["umbral"]).astype(int)
        a1, a2, dauc, se, z, pv = delong_rapido(yte, s, s_log)
        db = [roc_auc_score(yte[i], s[i]) - roc_auc_score(yte[i], s_log[i]) for i in idx_boot[:300]]
        ca, cb = d == yte, d_log == yte
        n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        chi = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if n01 + n10 else 0.0
        filas.append({"modelo_a": v["nombre"], "modelo_b": "Regresión logística", "auc_a": a1, "auc_b": a2, "delta_auc": dauc,
                      "se_delong": se, "z": z, "p_delong": pv, "ic_boot_inf": np.percentile(db, 2.5),
                      "ic_boot_sup": np.percentile(db, 97.5), "solo_a_acierta": n01, "solo_b_acierta": n10,
                      "chi2_mcnemar": chi, "p_mcnemar": float(stats.chi2.sf(chi, 1))})
    comp = pd.concat([comp, pd.DataFrame(filas)], ignore_index=True)
    comp["p_delong_holm"] = multipletests(comp["p_delong"], method="holm")[1]
    comp.to_parquet(DATA / "comparaciones.parquet", index=False)

    # Curvas ROC, PR y calibración
    cur = pd.read_parquet(DATA / "curvas.parquet")
    cur = cur[~cur["modelo"].isin(nuevos)]
    filas = []
    for v in res.values():
        s = v["p_test"]
        fpr, tpr, _ = roc_curve(yte, s)
        x, yv = submuestrear_curva(fpr, tpr)
        filas += [{"modelo": v["nombre"], "curva": "ROC", "x": a, "y": b} for a, b in zip(x, yv)]
        pr, rc, _ = precision_recall_curve(yte, s)
        x, yv = submuestrear_curva(rc[::-1], pr[::-1])
        filas += [{"modelo": v["nombre"], "curva": "PR", "x": a, "y": b} for a, b in zip(x, yv)]
        fr, mp = calibration_curve(yte, np.clip(s, 0, 1), n_bins=15, strategy="quantile")
        filas += [{"modelo": v["nombre"], "curva": "Calibración", "x": a, "y": b} for a, b in zip(mp, fr)]
    pd.concat([cur, pd.DataFrame(filas)], ignore_index=True).to_parquet(DATA / "curvas.parquet", index=False)

    # Validación cruzada temporal (con etiqueta de hiperparámetros para todos los modelos)
    cv = pd.read_parquet(DATA / "cv_resultados.parquet")
    cv = cv[~cv["modelo"].isin(nuevos)]
    if "params" not in cv or cv["params"].isna().any():
        cv["params"] = cv.apply(lambda f: f["params"] if isinstance(f.get("params"), str) else f"C={f['C']:g}", axis=1)
    pd.concat([cv, *[v["cv"] for v in res.values()]], ignore_index=True).to_parquet(DATA / "cv_resultados.parquet", index=False)

    for archivo, clave in [("cv_espacial", "espacial"), ("curva_aprendizaje", "aprendizaje")]:
        t = pd.read_parquet(DATA / f"{archivo}.parquet")
        t = t[~t["modelo"].isin(nuevos)]
        pd.concat([t, *[v[clave] for v in res.values() if len(v[clave])]], ignore_index=True).to_parquet(
            DATA / f"{archivo}.parquet", index=False)

    imp_path = DATA / "importancia_permutacion.parquet"
    imp_prev = pd.read_parquet(imp_path) if imp_path.exists() else pd.DataFrame(columns=["modelo"])
    imp_prev = imp_prev[~imp_prev["modelo"].isin(nuevos)]
    pd.concat([imp_prev, *[v["importancia"] for v in res.values()]], ignore_index=True).to_parquet(imp_path, index=False)

    for v in res.values():
        if "coeficientes" in v["extra"]:
            v["extra"]["coeficientes"].assign(modelo=v["nombre"]).to_parquet(DATA / "coeficientes_lasso.parquet", index=False)
        if "segmentos" in v["extra"]:
            v["extra"]["segmentos"].to_parquet(DATA / "segmentos_perfil.parquet", index=False)

    for v in res.values():
        r["modelos"][v["nombre"]] = {k: v[k] for k in ("params", "umbral", "f1_oof", "auc_cv", "auc_cv_sd", "curva_umbral",
                                                       "platt", "n_entrenamiento", "segundos")}
        r["modelos"][v["nombre"]]["extra"] = {k: x for k, x in v["extra"].items() if not isinstance(x, pd.DataFrame)}
    r["modelos"]["orden"] = ["Regresión logística", "SVM lineal"] + nuevos
    guardar_json(r, "resumen.json")
    print("Artefactos actualizados.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proyecto", type=Path, default=ROOT.parent / "proyecto_sipsa")
    ap.add_argument("--solo", nargs="+", choices=list(catalogo()), help="Entrenar solo estos modelos")
    ap.add_argument("--rehacer", action="store_true", help="Volver a entrenar aunque exista el resultado")
    ap.add_argument("--consolidar", action="store_true", help="Solo integrar resultados ya guardados")
    args = ap.parse_args()
    DIR_MOD.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(args.proyecto / "data/processed/panel_corredores.parquet")
    train, test = mu.particion(panel)
    del panel
    if not args.consolidar:
        if not (DATA / "importancia_permutacion.parquet").exists():
            t = time.time()
            importancia_lineales(train, test).to_parquet(DATA / "importancia_permutacion.parquet", index=False)
            print(f"  importancia de los modelos lineales: {time.time() - t:.0f} s", flush=True)
        for slug, cfg in catalogo().items():
            if args.solo and slug not in args.solo:
                continue
            destino = DIR_MOD / f"{slug}.joblib"
            if destino.exists() and not args.rehacer:
                print(f"  {cfg['nombre']}: ya entrenado (se omite)")
                continue
            print(f"Entrenando {cfg['nombre']}...", flush=True)
            joblib.dump(entrenar(slug, cfg, train, test), destino, compress=3)
    consolidar(test)


if __name__ == "__main__":
    main()
