# %% [markdown]
# # 3. Modelos base
#
# Este capítulo acompaña la pestaña **3. Modelos base** del dashboard. Documenta el preprocesamiento, el
# esquema de validación, la elección de hiperparámetros y umbrales, y la evaluación en el conjunto de
# prueba (todo 2025) de dos modelos lineales frente a líneas base triviales.
#
# **¿Por qué regresión logística y SVM lineal?** El enunciado pide un modelo base logístico o una SVR
# lineal. La SVR es para objetivos continuos; como la interrupción es binaria, se usa su contraparte de
# clasificación, la máquina de vectores de soporte lineal (`LinearSVC`), junto con la regresión logística.
# Ambas comparten el mismo preprocesamiento y difieren en la función de pérdida: logística (log-loss) y
# bisagra cuadrática (*squared hinge*), respectivamente.

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
pd.set_option("display.width", 160)
R = F.resumen()
K, M = R["kpis"], R["modelos"]
PRED = F.tabla("predicciones_test")
MET = F.tabla("metricas_test")

# %% [markdown]
# ## 3.1 Preprocesamiento
#
# Todo el preprocesamiento vive en un `Pipeline` de scikit-learn y se ajusta solo con los datos de
# entrenamiento de cada pliegue, de modo que ninguna estadística de validación o prueba se filtra al
# modelo. Cada decisión remite a un resultado del EDA:
#
# | Paso | Variables | Decisión | Resultado del EDA que la motiva |
# |---|---|---|---|
# | Imputación | numéricas | mediana de entrenamiento | único faltante: `ratio_t_1` (0,03 %), semanas sin encuesta |
# | Logaritmo | `n_envios8`, `n_alimentos8`, `dist_km`, `antig_sem` | `log1p` | asimetrías de 8,0, 4,0 y 1,7 |
# | Winsorización | numéricas continuas | recorte en percentiles 1 y 99 de entrenamiento | colas largas con valores genuinos |
# | Splines cúbicos | `ratio_t`, `ratio_t_1` | 5 nodos en cuantiles | relación en U con la interrupción |
# | Escalado | numéricas | `StandardScaler` | escalas distintas; la penalización exige escalas comparables |
# | Codificación | central, departamento, grupo dominante | one-hot; categorías < 1 % como «infrecuentes» | 6-8 categorías raras por variable |
# | Composición | participaciones por grupo | se omite `sh_procesados` | suman 1 (colinealidad exacta) |
# | Regularización | todas | L2 con C elegido por validación temporal | correlaciones altas entre tamaño y regularidad |
# | Excluidas | `tasa_t1`, `cod_mpio`, días de encuesta de t+1 | fuera del modelo | auditoría de fuga |

# %%
pd.Series({"numéricas con log": mu.LOG, "con splines": mu.SPL, "numéricas restantes": mu.NUM,
           "binarias": mu.BINARIAS, "categóricas": mu.CATEGORICAS}).to_frame("variables")

# %% [markdown]
# ## 3.2 Esquema de validación
#
# **Principal (futuro de corredores conocidos).** Validación cruzada de ventana creciente sobre
# semanas completas: `TimeSeriesSplit` aplicado a las semanas, con 5 pliegues y un hueco de 8 semanas
# entre entrenamiento y validación. Es el mismo tipo de partición que separa entrenamiento y prueba.
#
# **Secundario (territorios nuevos).** `GroupKFold` con el departamento de origen como grupo: el modelo
# se evalúa en departamentos que no vio al entrenar, lo que controla la autocorrelación espacial
# detectada en el EDA.

# %%
pd.DataFrame(M["pliegues"])

# %% [markdown]
# ## 3.3 Hiperparámetro de regularización

# %%
F.fig_cv("Regresión logística")

# %%
F.fig_cv("SVM lineal")

# %%
cv = F.tabla("cv_resultados")
cv.groupby(["modelo", "C"])[["auc_train", "auc_valid", "ap_valid"]].mean()

# %% [markdown]
# **Interpretación.** El AUC de validación apenas cambia con C: en la logística va de 0,7809 a 0,7810 y
# en la SVM de 0,7831 a 0,7834. Con 600 mil filas y unas 90 columnas los datos dominan sobre la
# penalización, así que la regularización casi no importa. Se eligió el C con mayor AUC medio (logística
# C = 0,01; SVM C = 0,001). La brecha entre entrenamiento y validación es de unas tres milésimas en ambos
# modelos: no hay sobreajuste. El AUC es estable entre pliegues (0,777 a 0,786), incluido el pliegue 2, cuya
# validación contiene el confinamiento y el paro nacional.

# %% [markdown]
# ## 3.4 Umbral de decisión y calibración de la SVM
#
# La logística entrega probabilidades; la SVM entrega distancias al hiperplano. Para que ambas puedan
# usarse con un umbral de probabilidad, las puntuaciones de la SVM se calibraron con el método de Platt
# (una regresión logística sobre la puntuación), ajustado con las predicciones **fuera de pliegue** del
# entrenamiento. El umbral de cada modelo se eligió maximizando el F1 en esas mismas predicciones fuera
# de pliegue, nunca con la prueba.

# %%
F.fig_umbral("Regresión logística")

# %%
F.fig_umbral("SVM lineal")

# %%
pd.DataFrame({m: {"C": M[m]["C"], "umbral": M[m]["umbral"], "F1 fuera de pliegue": M[m]["f1_oof"],
                  "AUC validación (media)": M[m]["auc_cv"], "AUC validación (d.e.)": M[m]["auc_cv_sd"]}
              for m in ["Regresión logística", "SVM lineal"]}).T

# %%
pd.Series(M["SVM lineal"]["platt"]).to_frame("Platt: p = 1 / (1 + exp(-(a·s + b)))")

# %% [markdown]
# **Interpretación.** El F1 fuera de pliegue es máximo con umbrales bajos (0,26 en la logística y 0,24 en
# la SVM), muy por debajo de 0,5, porque la clase positiva es minoritaria (28 %): para detectar la
# mayoría de las interrupciones hay que alertar desde probabilidades moderadas. En el dashboard el
# deslizador arranca en estos umbrales y permite explorar el resto.

# %% [markdown]
# ## 3.5 Líneas base triviales
#
# * `DummyClassifier(strategy="most_frequent")`: siempre predice «sin interrupción».
# * `DummyClassifier(strategy="stratified")`: predice al azar con la proporción de clases de entrenamiento.
# * **Persistencia**, la línea base natural en series de tiempo: alerta si el corredor ya estaba por debajo
#   del 50 % de su promedio en la semana actual (`ratio_t < 0,5`).

# %% [markdown]
# ## 3.6 Desempeño en la prueba (2025)

# %%
F.fig_roc_pr(["Regresión logística", "SVM lineal", "Persistencia", "Dummy (estratificado)"])

# %%
tabla = MET[MET["metrica"] != "Brier"].copy()
tabla["valor (IC 95 %)"] = [f"{v:.3f} ({a:.3f}–{b:.3f})" for v, a, b in zip(tabla["valor"], tabla["ic_inf"], tabla["ic_sup"])]
tabla.pivot(index="modelo", columns="metrica", values="valor (IC 95 %)")[["AUC-ROC", "AUC-PR", "F1", "recall", "precision", "accuracy"]]

# %% [markdown]
# **Interpretación.** Los intervalos de confianza al 95 % se obtienen con **bootstrap por bloques de
# semana** (500 réplicas): se remuestrean semanas completas para respetar la dependencia entre corredores
# de una misma semana. En 2025:
#
# * La **regresión logística** alcanza AUC-ROC 0,793 (0,789-0,797) y AUC-PR 0,539; la **SVM lineal**, 0,795
#   (0,792-0,799) y 0,543. Ambas superan por amplio margen a los Dummy (AUC 0,50; AUC-PR 0,26, que es la
#   prevalencia) y a la persistencia (AUC 0,584; AUC-PR 0,389).
# * Que la persistencia rinda tan poco es informativo: saber que el corredor ya cayó esta semana ayuda
#   poco; lo que más anticipa la interrupción es su tamaño y su regularidad.
# * Con los umbrales elegidos, la logística detecta 76,3 % de las interrupciones (recall) con una
#   precisión de 45,6 % y F1 de 0,571; la SVM, 77,3 %, 45,5 % y 0,573.
# * La **exactitud** (0,70) es menor que la del Dummy de clase mayoritaria (0,74), y no es un error: con
#   umbrales bajos el modelo acepta falsas alarmas para detectar las interrupciones, que es lo que importa
#   en una alerta temprana. Con desbalance, la exactitud no sirve para elegir modelos.

# %%
cmp = F.tabla("comparaciones")
cmp[["modelo_a", "modelo_b", "auc_a", "auc_b", "delta_auc", "ic_boot_inf", "ic_boot_sup", "z", "p_delong_holm",
     "solo_a_acierta", "solo_b_acierta", "chi2_mcnemar", "p_mcnemar"]]

# %% [markdown]
# **Comparación entre modelos.** Se aplicaron tres pruebas complementarias:
#
# * **DeLong** (algoritmo rápido de Sun y Xu, 2014), para AUC correlacionados sobre las mismas filas, con
#   corrección de Holm por las tres comparaciones.
# * **Bootstrap pareado** de la diferencia de AUC, con los mismos índices para ambos modelos.
# * **McNemar**, sobre las decisiones con los umbrales elegidos.
#
# La SVM supera a la logística por **0,002** puntos de AUC (IC 95 % bootstrap: 0,0018 a 0,0026). Con
# 103 mil filas la prueba de DeLong declara significativa incluso esa diferencia (z = −10,9), pero según la
# escala de relevancia del curso (ΔAUC < 0,01 trivial; 0,01-0,03 pequeña; 0,03-0,05 moderada; > 0,05
# importante) es **trivial**. McNemar tampoco encuentra diferencias en las decisiones (χ² = 2,3; p = 0,13):
# los dos modelos aciertan y fallan casi en las mismas filas. En cambio, la ventaja de ambos sobre la
# persistencia (+0,21 de AUC) es **importante** y significativa en las tres pruebas.

# %%
for m, col in [("Regresión logística", "p_logistica"), ("SVM lineal", "p_svm")]:
    mt = F.metricas_umbral(PRED["y"].to_numpy(), PRED[col].to_numpy(), M[m]["umbral"])
    F.fig_matriz(mt, f"Matriz de confusión · {m} (umbral {M[m]['umbral']:.2f})".replace(".", ",")).show()

# %% [markdown]
# **Interpretación.** Con el umbral de 0,26, la logística emite 45.126 alertas en 2025: 20.558 son
# interrupciones reales (verdaderos positivos) y 24.568 falsas alarmas; deja sin detectar 6.388 de las
# 26.946 interrupciones. La SVM produce una matriz casi idéntica. En el dashboard el deslizador recalcula
# esta matriz para cualquier umbral: subirlo a 0,5 reduce las alertas a menos de la mitad y eleva la
# precisión, pero deja escapar alrededor de dos tercios de las interrupciones.

# %%
F.fig_calibracion()

# %%
MET[MET["metrica"] == "Brier"][["modelo", "valor"]]

# %% [markdown]
# **Interpretación.** Las probabilidades de ambos modelos están bien calibradas en el rango donde hay más
# datos: en el decil de probabilidad 0,43 la frecuencia observada es 0,43. En el extremo alto hay una leve
# sobreestimación (0,69 predicho frente a 0,65 observado en la logística), coherente con la menor tasa de
# 2025 (26,0 % frente a 28,2 % en entrenamiento). El Brier de ambos (0,154) mejora 20 % sobre la predicción
# constante (0,193). Que las probabilidades estén calibradas permite usarlas para priorizar recursos, no
# solo para clasificar.

# %% [markdown]
# ## 3.7 Interpretación de los coeficientes

# %%
F.fig_coeficientes("Regresión logística", 25)

# %%
F.fig_coeficientes("SVM lineal", 25)

# %%
cl = F.tabla("coeficientes_logistica").set_index("variable")["coeficiente"]
cont = cl[[c for c in cl.index if c in mu.LOG + mu.NUM + mu.BINARIAS]]
pd.DataFrame({"coeficiente": cont, "odds ratio por d.e.": np.exp(cont)}).sort_values("coeficiente")

# %% [markdown]
# **Interpretación.** Los dos modelos coinciden en el orden de importancia, aunque la escala de sus
# coeficientes es distinta (log-odds en la logística y pesos del hiperplano en la SVM):
#
# * **Tamaño y regularidad dominan.** Una desviación estándar más de cargas semanales (en log) reduce a la
#   mitad los odds de interrupción (OR 0,49); más semanas activas en el semestre (OR 0,72) y una tendencia
#   reciente creciente (OR 0,85) también protegen; la variabilidad (`cv8`) los aumenta (OR 1,28).
# * **Signos que se leen en bloque.** `n_alimentos8` y `activas8` tienen coeficiente positivo pese a que su
#   relación individual con la interrupción es negativa. Es un efecto de supresión por colinealidad
#   (correlación 0,91 con `n_envios8` y 0,83 con `activas26`): a igual número de cargas, más productos
#   distintos significa cargas más pequeñas y diversas.
# * **Contexto.** A igualdad de lo demás, los corredores hacia las plazas minoristas de Bogotá (Samper
#   Mendoza, Las Flores) tienen más riesgo y los que llegan a Armenia, Granabastos, Villavicencio o Neiva,
#   menos; los de tubérculos, lácteos y carnes son más estables. La distancia (OR 1,07) y los festivos de la
#   semana siguiente (OR 1,07) aumentan levemente el riesgo.
# * **PDET.** El coeficiente es casi nulo (OR 0,99). La mayor tasa bruta de los corredores PDET (33 % frente a
#   28 %) se explica por sus características: son más pequeños, irregulares y lejanos. Para la tesis esto
#   indica que su fragilidad tiene causas logísticas (escala y regularidad) que una intervención puede
#   atacar: consolidar cargas, centros de acopio, regularidad de despachos.

# %%
F.fig_dependencia()

# %% [markdown]
# **Interpretación.** La dependencia parcial muestra que los splines reproducen la U del EDA en ambos
# modelos: la probabilidad media de interrupción es máxima (≈ 36 %) cuando la semana actual trajo entre 10 y
# 30 % del promedio, mínima (≈ 24 %) cerca del nivel habitual y sube de nuevo a ≈ 26 % tras un pico.

# %% [markdown]
# ## 3.8 Generalización a territorios no vistos

# %%
F.fig_cv_espacial()

# %%
F.tabla("cv_espacial").groupby("modelo")[["auc", "ap"]].agg(["mean", "std"])

# %% [markdown]
# **Interpretación.** Evaluados en grupos de departamentos que no vieron al entrenar, los modelos obtienen
# AUC 0,780 ± 0,010 (logística) y 0,783 ± 0,010 (SVM), prácticamente lo mismo que en la validación temporal
# (0,781 y 0,783). El modelo generaliza a territorios nuevos porque no usa identificadores ni coordenadas,
# sino atributos del corredor. La mayor dispersión entre pliegues (el más bajo, 0,768, incluye Córdoba,
# Magdalena y Santander) refleja la heterogeneidad regional del EDA.

# %% [markdown]
# ## 3.9 Curva de aprendizaje

# %%
F.fig_curva_aprendizaje()

# %%
F.tabla("curva_aprendizaje").assign(brecha=lambda d: d["auc_train"] - d["auc_valid"])

# %% [markdown]
# **Interpretación.** Con 10 mil filas la brecha entre entrenamiento y validación es de 0,08 (sobreajuste
# por pocos datos); desde 50 mil filas cae por debajo de 0,01 y el AUC de validación solo mejora de 0,781 a
# 0,784 (logística) al pasar de 100 mil a 500 mil filas. La muestra es más que suficiente para un modelo
# lineal; el límite es su **forma**, no la cantidad de datos. Los modelos no lineales de los siguientes
# entregables (k-NN, Random Forest, XGBoost, SVM con kernel) tienen margen para mejorar.

# %% [markdown]
# ## 3.10 Desempeño por subgrupo

# %%
F.fig_subgrupos("region_origen")

# %%
F.fig_subgrupos("grupo_dom")

# %%
F.fig_subgrupos("pdet_origen")

# %% [markdown]
# **Interpretación.** El desempeño no es homogéneo. Por región, el AUC va de 0,752 en la Orinoquía a 0,797
# en la región Andina; por grupo de alimentos, de 0,675 en pescados a 0,831 en verduras. Los corredores
# PDET tienen un AUC algo menor (0,770 frente a 0,795). Los grupos donde el modelo rinde peor son los de
# mayor tasa de interrupción y menor volumen: allí el comportamiento es más errático y depende de factores
# que el SIPSA no registra. Por central, el AUC va de 0,62 en la plaza Samper Mendoza a 0,89 en Ipiales
# (selector *Subgrupo* del dashboard).

# %% [markdown]
# ## 3.11 Diagnóstico de residuos (detalle)

# %%
F.fig_residuos()

# %%
pd.DataFrame(M["residuos"]["ljung_box"])

# %%
pd.Series({"I de Moran del residuo municipal": M["residuos"]["moran_residuo"], "p": M["residuos"]["p_residuo"],
           "I de Moran de la tasa observada en prueba": M["residuos"]["moran_tasa_test"]}).to_frame("valor")

# %% [markdown]
# **Interpretación.** Se usan los residuos de Pearson, $r_i = (y_i-\hat p_i)/\sqrt{\hat p_i(1-\hat p_i)}$. El
# residuo medio semanal no tiene autocorrelación (Ljung-Box p = 0,14, 0,68 y 0,81 en los rezagos 1, 4 y 8):
# los rezagos y ventanas capturaron la dependencia temporal. La autocorrelación espacial baja de I = 0,17 en
# la tasa observada a I = 0,045 en el residuo, pero sigue siendo significativa (p = 0,008): queda información
# espacial sin capturar, probablemente los choques locales que detectó la prueba de Knox. Un rezago espacial
# de las interrupciones recientes es la extensión natural.

# %% [markdown]
# ## 3.12 Verificaciones de la nota crítica
#
# El enunciado del curso pide revisar el modelo si la exactitud supera 80-90 %. Aquí queda en 0,70-0,77, por
# debajo de ese rango, y aun así se verificó lo que la nota exige:
#
# * **Fuga.** Ninguna predictora admitida supera AUC univariado 0,753 y las variables construidas con $t+1$
#   están excluidas. En el Entregable 1 se comprobó que quitar las variables en observación `frac_caida_*`
#   deja el AUC de prueba en 0,793.
# * **Línea base.** La exactitud del modelo es comparable a la del Dummy de clase mayoritaria (0,74); por
#   eso se reportan métricas insensibles al desbalance.
# * **Estructura temporal y espacial.** La validación respeta el tiempo (bloques con hueco) y se evaluó
#   aparte con bloques espaciales.
# * **Coherencia.** El AUC de prueba (0,793) es coherente con el de validación (0,781).
# * **Dificultad.** El problema no es trivial: el modelo ordena bien el riesgo, pero su precisión en la clase
#   positiva es moderada.

# %% [markdown]
# ## 3.13 Simulador del dashboard
#
# La última sección de la pestaña permite ajustar seis características de un corredor hipotético (cargas
# por semana, variabilidad, semanas activas, dinámica reciente, distancia y festivos), su grupo de
# alimentos, la central de destino y la condición PDET. Las demás variables se fijan en la mediana (o la
# moda) del entrenamiento, y el pipeline logístico serializado (`data/modelo_logistico.joblib`) calcula la
# probabilidad de interrupción y la compara con el umbral. Sirve para explorar el modelo, no para
# pronosticar corredores reales, porque las variables restantes no se ajustan al caso.

# %%
import joblib

perfil = pd.DataFrame([M["perfil_base"]])[mu.X_COLS]
modelo = joblib.load(F.DATA / "modelo_logistico.joblib")
print(f"Probabilidad del perfil mediano: {modelo.predict_proba(perfil)[0, 1]:.3f}")
