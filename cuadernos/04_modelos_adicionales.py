# %% [markdown]
# # 4. Modelos adicionales y comparación
#
# El capítulo 3 dejó dos preguntas abiertas: si la forma lineal limitaba el desempeño (la curva de
# aprendizaje se aplanaba desde unas 50 mil filas) y qué aportan los demás modelos del curso. Aquí se
# entrenan ocho modelos más sobre el mismo panel y con el mismo esquema, y se comparan con la regresión
# logística. El dashboard los incluye en el selector de la pestaña **3. Modelos base**.

# %%
import sys

sys.path.insert(0, "../src")
import numpy as np
import pandas as pd
import plotly.io as pio

import figuras as F
import modelo_utils as mu

pio.renderers.default = "notebook_connected"
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
pd.set_option("display.width", 170)
R = F.resumen()
M = R["modelos"]
MODELOS = F.modelos_entrenados()
MET, COMP = F.tabla("metricas_test"), F.tabla("comparaciones")

# %% [markdown]
# ## 4.1 Los modelos y por qué se incluyeron
#
# | Modelo | Idea | Hiperparámetros explorados | Observaciones |
# |---|---|---|---|
# | Lasso (L1) | Logística con penalización L1, que anula coeficientes poco útiles | C ∈ {0,0005; 0,002; 0,01; 0,05} | Selección de variables; solver `liblinear` con `l1_ratio = 1` (API de scikit-learn 1.9) |
# | k-NN | Proporción de interrupciones entre los k vecinos más parecidos | k ∈ {50, 150, 300} | Submuestra de 100 mil filas: con 620 mil, cada predicción compara contra todo el entrenamiento |
# | Naive Bayes | Bayes gaussiano con independencia condicional | `var_smoothing` ∈ {10⁻⁹, 10⁻⁵, 10⁻²} | Línea base probabilística; su supuesto de independencia no se cumple aquí |
# | Random Forest | Promedio de 150 árboles sobre submuestras de filas y variables | `min_samples_leaf` ∈ {20, 100} | `max_samples` = 25 %, `max_features` = raíz cuadrada |
# | XGBoost | Gradient boosting: 400 árboles en secuencia | tasa de aprendizaje ∈ {0,05; 0,1} × profundidad ∈ {4, 6} | `subsample` = `colsample_bytree` = 0,8; `min_child_weight` = 50; algoritmo `hist` |
# | SVM RBF (Nyström) | SVM con kernel gaussiano aproximado con 300 componentes | γ ∈ {0,005; 0,02} × C ∈ {0,01; 0,1} | Submuestra de 150 mil filas; probabilidades por Platt |
# | PCA + logística | Logística sobre componentes principales | componentes ∈ {10, 20, 40} | Prueba si reducir dimensiones ayuda |
# | Logística por segmentos | K-means por escala y regularidad + una logística por segmento | segmentos ∈ {2, 3} | Retoma los dos perfiles de corredor del EDA |
#
# Todos usan el mismo preprocesamiento (capítulo 3), la misma validación por bloques de semanas con hueco
# de 8 semanas para elegir hiperparámetros, el mismo criterio de umbral (máximo F1 fuera de pliegue) y se
# evaluaron una sola vez en la prueba de 2025. La logística y la SVM lineal conservan los resultados del
# capítulo 3.

# %% [markdown]
# ## 4.2 Validación cruzada

# %%
cv = F.tabla("cv_resultados")
tabla_cv = (cv.groupby(["modelo", "params"], sort=False)
              .agg(auc_valid=("auc_valid", "mean"), sd=("auc_valid", "std"), auc_train=("auc_train", "mean"))
              .reset_index())
tabla_cv["elegido"] = [p == F._params_elegidos(m) for m, p in zip(tabla_cv["modelo"], tabla_cv["params"])]
tabla_cv["brecha"] = tabla_cv["auc_train"] - tabla_cv["auc_valid"]
tabla_cv

# %%
F.fig_validaciones()

# %% [markdown]
# **Interpretación.** En la validación por bloques temporales el orden ya anticipa lo que se verá en la
# prueba: XGBoost obtiene el mayor AUC medio (0,791), seguido de Random Forest (0,786) y la SVM RBF (0,785);
# los modelos lineales y sus variantes (Lasso, PCA + logística, logística por segmentos) quedan entre 0,780 y
# 0,783; k-NN en 0,777 y Naive Bayes en 0,753. En departamentos no vistos se mantiene el mismo orden (XGBoost
# 0,792; Random Forest 0,787; logística 0,780): ningún modelo depende de memorizar ubicaciones.
#
# La columna *brecha* (AUC de entrenamiento menos AUC de validación) separa dos familias. En los lineales es
# de unas tres milésimas, sin sobreajuste. En los ensambles de árboles es mayor (0,026 en el XGBoost elegido y
# 0,032 en el Random Forest), señal de que ajustan detalles del entrenamiento que no se repiten; aun así su
# AUC de validación es el más alto, de modo que esa flexibilidad compensa. En el Random Forest y el k-NN el AUC
# de entrenamiento es optimista por construcción (el árbol o el vecino incluyen la propia fila), así que su
# brecha se lee con cautela.

# %%
F.fig_cv("XGBoost")

# %%
F.fig_cv("k-NN")

# %% [markdown]
# **Hiperparámetros.** En XGBoost la combinación más profunda y rápida (tasa 0,1, profundidad 6) tiene
# la mayor brecha (0,043) y un AUC de validación menor: es sobreajuste. Se eligió tasa 0,05 con profundidad 6.
# En k-NN el AUC crece con k (0,772 con 50 vecinos; 0,777 con 300) y la brecha se cierra: promediar más
# vecinos suaviza la predicción. Varios modelos eligieron el valor extremo de su rejilla (k = 300 en k-NN,
# C = 0,05 en Lasso, `var_smoothing` = 0,01 en Naive Bayes, C = 0,1 en la SVM RBF). Las rejillas se
# mantuvieron pequeñas por costo computacional; ampliarlas es un paso pendiente, aunque las diferencias
# entre valores vecinos son de milésimas.

# %% [markdown]
# ## 4.3 Desempeño en la prueba (2025)

# %%
F.fig_comparacion_modelos("AUC-ROC")

# %%
F.fig_comparacion_modelos("AUC-PR")

# %%
t = MET[MET["metrica"].isin(["AUC-ROC", "AUC-PR", "F1", "recall", "precision", "accuracy", "Brier"])].copy()
t["valor (IC 95 %)"] = [f"{v:.3f} ({a:.3f}–{b:.3f})" if pd.notna(a) else f"{v:.3f}" for v, a, b in zip(t["valor"], t["ic_inf"], t["ic_sup"])]
orden = MET[MET["metrica"] == "AUC-ROC"].sort_values("valor", ascending=False)["modelo"]
t.pivot(index="modelo", columns="metrica", values="valor (IC 95 %)").reindex(orden)[["AUC-ROC", "AUC-PR", "F1", "recall", "precision", "accuracy", "Brier"]]

# %%
c = COMP[COMP["modelo_b"] == "Regresión logística"].copy()
c["relevancia"] = c["delta_auc"].map(F.escala_dauc)
c.sort_values("delta_auc", ascending=False)[["modelo_a", "auc_a", "delta_auc", "ic_boot_inf", "ic_boot_sup", "z", "p_delong_holm",
                                              "relevancia", "solo_a_acierta", "solo_b_acierta", "p_mcnemar"]]

# %% [markdown]
# **Interpretación.** En la prueba de 2025:
#
# * **XGBoost** obtiene el mejor desempeño: AUC-ROC 0,806 (IC 95 %: 0,802-0,809) y AUC-PR 0,569, frente a
#   0,793 y 0,539 de la logística. La mejora de AUC es de 0,013 (IC bootstrap: 0,011-0,015), significativa
#   según DeLong (z = 22,8) y **pequeña** según la escala del curso. Es el único modelo que supera el umbral
#   de lo trivial.
# * **Random Forest** (+0,006) y la **SVM RBF** (+0,005) también superan a la logística, pero con diferencias
#   triviales. La **logística por segmentos** gana 0,002, igual de trivial.
# * **Lasso** iguala a la logística (diferencia menor que 0,0001): con la penalización elegida conserva 87 de
#   los 96 coeficientes, así que es casi el mismo modelo.
# * **PCA + logística** (−0,001) y **k-NN** (−0,006) quedan levemente por debajo; **Naive Bayes** (−0,028) es el
#   más débil, con una diferencia pequeña pero negativa.
# * Todos superan con amplitud a la persistencia (0,584) y a los Dummy (0,50).
#
# **McNemar.** Con los umbrales elegidos, XGBoost y la logística tienen una exactitud casi igual (0,700 y
# 0,701) y McNemar no encuentra diferencias en los aciertos (p = 0,29). La ganancia de XGBoost está en *qué*
# detecta: con la misma precisión (0,456) recupera más interrupciones (recall 0,794 frente a 0,763). En 2025
# emite 46.902 alertas, de las cuales 21.399 son interrupciones reales, 841 más que la logística, y deja sin
# detectar 5.547 en lugar de 6.388. Para Lasso, McNemar sí da diferencias (p ≈ 0) aunque su AUC es idéntico:
# su umbral elegido es 0,25 y no 0,26, y ese pequeño cambio mueve muchas decisiones.

# %%
F.fig_roc_pr(["Regresión logística", "XGBoost", "Random Forest", "k-NN", "Naive Bayes", "Persistencia"])

# %% [markdown]
# **Interpretación.** La curva ROC de XGBoost queda por encima de la de la logística en todo el rango: con
# una tasa de falsos positivos de 0,10 detecta el 42,0 % de las interrupciones, frente a 38,3 %. En la curva
# precisión-recall la ventaja es mayor entre las alertas de más riesgo: con recall 0,2 la precisión sube de
# 0,64 a 0,68 y con recall 0,5, de 0,54 a 0,57; con recall alto (0,9) ambas convergen (0,39 y 0,40). Es decir,
# XGBoost ordena mejor sobre todo los casos más claros, útil si los recursos solo alcanzan para atender las
# primeras alertas. Las curvas de Random Forest y k-NN quedan cerca de la de la logística; Naive Bayes, por
# debajo.

# %% [markdown]
# ## 4.4 Calibración

# %%
F.fig_calibracion(("Regresión logística", "XGBoost", "Random Forest", "k-NN", "Naive Bayes"))

# %% [markdown]
# **Interpretación.** XGBoost, Random Forest y k-NN producen probabilidades bien calibradas: la mayor
# distancia a la diagonal es de 0,03 a 0,04, y el Brier de XGBoost (0,150) es el mejor de todos (logística:
# 0,154; constante: 0,193). La SVM RBF, calibrada con Platt, sobreestima un poco en el extremo alto (0,71
# predicho frente a 0,66 observado). **Naive Bayes está muy mal calibrado**: cuando predice probabilidades
# cercanas a 1, solo se interrumpe el 53 %, y su Brier (0,266) es peor que el de predecir siempre la tasa
# media (0,193). La causa es su supuesto de independencia: con variables tan correladas como cargas,
# productos y semanas activas, cuenta varias veces la misma evidencia y empuja las probabilidades a los
# extremos. Por eso su umbral óptimo de F1 quedó en 0,05.

# %% [markdown]
# ## 4.5 ¿Qué variables usa cada modelo?

# %%
F.fig_importancia_comparada()

# %%
F.fig_importancia("XGBoost", 15)

# %% [markdown]
# **Interpretación.** La importancia por permutación mide cuánto cae el AUC en una muestra de prueba al
# barajar cada variable, de modo que es comparable entre modelos. Todos se apoyan en el mismo núcleo de
# variables de **regularidad, tamaño y dinámica reciente** (`activas26`, `cv8`, `n_envios8`, `ratio_t`,
# `ratio_t_1`), pero con énfasis distintos:
#
# * La **logística** concentra casi todo en el número de cargas (`n_envios8`), porque entre variables muy
#   correlacionadas un modelo lineal regularizado tiende a cargar el peso en una de ellas.
# * **XGBoost** reparte la importancia entre semanas activas (la primera), variabilidad y cargas, y da más peso
#   a la central de destino que los modelos lineales.
# * **Random Forest** y **k-NN** se apoyan sobre todo en la dinámica reciente (`ratio_t` y `ratio_t_1`), la
#   relación en U que los modelos lineales solo representan gracias a los splines.
#
# Que la importancia cambie de modelo a modelo entre variables correlacionadas no es una contradicción:
# describe cómo usa cada modelo una información compartida, no cuál variable «causa» la interrupción.

# %%
lasso = F.tabla("coeficientes_lasso")
print(f"Lasso conserva {M['Lasso (L1)']['extra']['coef_no_nulos']} de {M['Lasso (L1)']['extra']['coef_total']} coeficientes")
lasso.assign(nombre=[F.etiqueta_variable(v) for v in lasso["variable"]]).query("coeficiente == 0")[["nombre"]].head(40)

# %%
F.fig_coeficientes("Lasso (L1)", 20)

# %% [markdown]
# **Interpretación.** Con la penalización elegida, Lasso anula 9 de los 96 coeficientes: dos bases del
# spline de `ratio_t_1`, la participación de tubérculos, la marca de mismo departamento, la central de
# Montería, tres departamentos (Bolívar, Santander, Tolima) y el grupo dominante de frutas. Son variables
# redundantes con otras que el modelo conserva. Penalizaciones más fuertes eliminan más variables pero bajan
# el AUC de validación (0,779 con C = 0,0005), por eso la validación eligió la más débil de la rejilla. La
# conclusión práctica: casi todas las variables aportan algo y no hay un subconjunto pequeño que baste.

# %%
F.tabla("segmentos_perfil")

# %% [markdown]
# **Interpretación.** La validación eligió tres segmentos. K-means separa corredores **irregulares y
# pequeños** (cv8 mediano 1,12; 1,3 cargas semanales; 50,5 % de semanas con interrupción), **intermedios**
# (cv8 0,55; 5 cargas; 23,9 %) y **grandes y regulares** (cv8 0,25; 36 cargas; 6,4 %). Una logística distinta
# por segmento mejora el AUC en solo 0,002: los perfiles son muy distintos, pero el modelo único ya los
# distinguía a través de las mismas variables con que se forman los segmentos.

# %% [markdown]
# ## 4.6 Generalización, aprendizaje y subgrupos

# %%
F.tabla("cv_espacial").groupby("modelo", sort=False)[["auc", "ap"]].agg(["mean", "std"])

# %%
F.fig_curva_aprendizaje()

# %%
F.tabla("curva_aprendizaje").assign(brecha=lambda d: d["auc_train"] - d["auc_valid"])

# %% [markdown]
# **Interpretación.** Las curvas de aprendizaje separan con claridad las dos familias. En los modelos
# lineales el AUC de validación se aplana desde unas 50 mil filas (0,781 a 0,784). En **XGBoost** sigue
# subiendo con más datos: 0,786 con 100 mil filas, 0,791 con 200 mil y 0,796 con 500 mil, mientras la brecha
# con el entrenamiento se cierra de 0,154 a 0,012. Random Forest también mejora, aunque menos (0,786 a 0,790).
# Esto confirma la lectura del capítulo 3: el techo de la logística lo ponía su forma, y un modelo flexible
# aprovecha los datos que el lineal ya no podía usar. Sugiere también que XGBoost se beneficiaría de más
# historia o de más árboles con parada temprana.

# %%
from sklearn.metrics import roc_auc_score

pred = F.tabla("predicciones_test")
filas = []
for dim in ["region_origen", "grupo_dom", "pdet_origen"]:
    for v, g in pred.groupby(dim):
        if len(g) < 300:
            continue
        fila = {"dimension": dim, "grupo": str(v), "n": len(g)}
        for m in ["Regresión logística", "XGBoost", "Random Forest"]:
            fila[m] = roc_auc_score(g["y"], g[mu.COLUMNA_MODELO[m]])
        filas.append(fila)
sub = pd.DataFrame(filas)
sub["XGBoost − logística"] = sub["XGBoost"] - sub["Regresión logística"]
sub

# %% [markdown]
# **Interpretación.** La mejora de XGBoost no se concentra en un grupo: supera a la logística en las cinco
# regiones (entre +0,009 en la Amazonía y +0,014 en el Caribe), en los ocho grupos de alimentos (entre +0,008 en
# verduras y +0,017 en lácteos y tubérculos), en los corredores PDET y no PDET (+0,014 y +0,013) y en 30 de las
# 32 centrales (en las otras dos el AUC es igual). Los grupos difíciles siguen siéndolo: pescados (0,689),
# granos (0,722) y la Orinoquía (0,764) son los de menor AUC también con XGBoost.

# %% [markdown]
# ## 4.7 Costo computacional

# %%
pd.DataFrame({m: {"hiperparámetros elegidos": F._params_elegidos(m), "filas de entrenamiento": M[m].get("n_entrenamiento", 622244),
                  "minutos (búsqueda, validación e importancia)": M[m].get("segundos", np.nan) / 60}
              for m in MODELOS if m not in ("Regresión logística", "SVM lineal")}).T

# %% [markdown]
# **Interpretación.** Gracias al algoritmo de histogramas, XGBoost tardó menos de 3 minutos en toda la
# búsqueda, la validación espacial, la curva de aprendizaje y la importancia; Random Forest, algo más de 2.
# k-NN fue el más lento (6 minutos) aun con una submuestra de 100 mil filas, porque cada predicción compara
# contra todo el conjunto de entrenamiento; la SVM RBF necesitó la aproximación de Nyström y una submuestra de
# 150 mil filas porque una SVM con kernel exacto no escala a 620 mil filas.
#
# ## 4.8 Conclusión del capítulo
#
# * **XGBoost es el mejor modelo** (AUC 0,806; AUC-PR 0,569; Brier 0,150), con una mejora pequeña pero
#   consistente sobre la logística en todas las regiones, grupos de alimentos y validaciones, y con las
#   probabilidades mejor calibradas. Detecta 841 interrupciones más en 2025 con la misma precisión.
# * **La logística sigue siendo una línea base muy competitiva**: siete de los nueve modelos alternativos quedan
#   a menos de 0,01 de ella, y es la más fácil de interpretar y de desplegar (es la que usa el simulador del
#   dashboard).
# * **Naive Bayes no es adecuado** para este problema por la fuerte correlación entre predictoras, y **k-NN** no
#   compensa su costo.
# * La mejora de los modelos flexibles viene de captar interacciones y la dinámica reciente sin necesidad de
#   splines definidos a mano, y crece con la cantidad de datos. Para la tesis, esto respalda probar modelos
#   más expresivos (incluido deep learning), siempre comparados contra un XGBoost bien ajustado y no solo
#   contra la logística.
