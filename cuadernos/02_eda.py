# %% [markdown]
# # 2. Análisis exploratorio
#
# Este capítulo acompaña la pestaña **2. Análisis exploratorio** del dashboard y sigue su orden: objetivo
# y tiempo, variables y relaciones, espacio y fuga de datos. Cada figura del dashboard aparece aquí con
# su interpretación; los recuadros **Detalle** contienen análisis que no se muestran en pantalla.
#
# Todo el EDA usa solo el conjunto de entrenamiento (semanas objetivo de 2018 a octubre de 2024). La
# prueba se reservó antes de cualquier decisión de modelado.

# %%
import sys

sys.path.insert(0, "../src")
import numpy as np
import pandas as pd
import plotly.io as pio

import figuras as F
from modelo_utils import ETIQUETAS

pio.renderers.default = "notebook_connected"
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
pd.set_option("display.width", 160)
R = F.resumen()
K, E = R["kpis"], R["eda"]
T, S = E["temporal"], E["espacial"]

# %% [markdown]
# ## 2.1 Variable objetivo y tiempo
#
# ### Serie semanal de la tasa de interrupción

# %%
F.fig_serie("total")

# %%
pd.DataFrame(T["eventos"]).assign(tasa_fuera=T["tasa_fuera_eventos"])

# %% [markdown]
# **Interpretación.** La tasa semanal oscila alrededor de 28 % (28,2 % en entrenamiento) sin tendencia
# marcada, pero con picos que afectan a muchos corredores a la vez. El más alto coincide con el paro
# nacional de 2021: en la semana del 3 de mayo se interrumpió 55 % de los corredores y el promedio del
# periodo fue 35,3 %. El confinamiento de 2020 elevó la media a 31,8 %, frente a 27,9 % fuera de ambos
# eventos. Son choques comunes que ninguna variable individual del corredor puede anticipar; tienen dos
# consecuencias para el modelado: la validación debe ser cronológica y los intervalos de confianza deben
# remuestrear semanas completas, no filas.

# %%
F.fig_serie("region_origen", ["Andina", "Caribe", "Pacífica", "Orinoquía", "Amazonía"])

# %%
pd.DataFrame(S["correlacion_regional"]).round(2)

# %% [markdown]
# **Interpretación.** Al desagregar por región de origen (selector *Desagregar por* del dashboard), la
# región Andina, que aporta dos de cada tres filas, tiene la tasa más baja (26,4 %), mientras el Caribe,
# la Orinoquía y la Amazonía están por encima de 34 %. Las series trimestrales de la región Andina, el
# Pacífico, la Amazonía y la Orinoquía están correlacionadas entre sí; el Caribe sigue una dinámica
# propia, con correlaciones bajas o negativas frente a las demás regiones. Esto justifica incluir el
# departamento de origen como predictora y evaluar el modelo por región.

# %%
F.fig_serie("mercado", ["Bogotá, D.C., Corabastos", "Medellín, Central Mayorista de Antioquia",
                        "Cartagena, Bazurto", "Bogotá, D.C., Plaza Las Flores"])

# %% [markdown]
# **Interpretación.** Por central, las diferencias de nivel son grandes y persistentes: las plazas
# minoristas de Bogotá (Las Flores, Samper Mendoza) superan 45 %, mientras Ipiales y Armenia están cerca
# de 19 %. Las grandes centrales (Corabastos, Central Mayorista de Antioquia) tienen series más suaves
# porque agregan muchos corredores grandes. La central de destino también entra al modelo.

# %% [markdown]
# ### Estacionalidad y calendario

# %%
F.fig_estacionalidad()

# %%
pd.DataFrame(E["festivos"]).rename(columns={"mean": "tasa", "size": "n"})

# %% [markdown]
# **Interpretación.** Las diferencias entre meses son significativas (Kruskal-Wallis H = 82,5,
# p ≈ 5·10⁻¹³): marzo (mediana 29,9 %), abril (29,2 %) y diciembre (28,9 %) tienen las medianas más
# altas, y febrero (26,2 %) la más baja. Los picos aislados de enero corresponden a la primera semana del
# año. La causa visible son los festivos: sin festivos en la semana siguiente la tasa es 27,6 %; con uno, 29,0 %; con
# dos, 37,8 %, y con tres (Semana Santa, fin de año), 40,0 %. Como el calendario se conoce de antemano,
# se incluyen como predictoras el número de festivos de $t+1$ y la semana del año codificada con seno y
# coseno, que capta el ciclo anual sin el salto artificial entre la semana 52 y la 1.

# %% [markdown]
# :::{admonition} Detalle: descomposición STL y estacionariedad
# :class: note
# La descomposición STL (periodo de 52 semanas) separa una tendencia débil (fuerza 0,08) de una
# estacionalidad anual moderada (fuerza 0,41). La serie es estacionaria: la prueba ADF rechaza la raíz
# unitaria (estadístico −12,0; p ≈ 3·10⁻²²) y la KPSS no rechaza la estacionariedad en nivel (p ≥ 0,10).
# Definir la interrupción como una caída **relativa** al propio corredor eliminó la tendencia que sí
# tiene el volumen de carga.
# :::

# %%
F.fig_stl()

# %%
pd.Series(T["punto_cambio"]).to_frame("punto de cambio en la media")

# %% [markdown]
# El único punto de cambio en la media (segmentación binaria) se ubica en mayo de 2023, con una
# reducción pequeña (28,6 % → 26,7 %) que coincide con la ampliación del operativo del DANE a nuevas
# centrales. No altera la definición del problema, pero explica por qué la tasa de prueba (26,0 %) es
# algo menor que la de entrenamiento (28,2 %).

# %% [markdown]
# ### Dependencia temporal

# %%
F.fig_acf()

# %%
F.fig_autocorr_corredor()

# %% [markdown]
# **Interpretación.** La ACF de la tasa nacional vale 0,44 en el rezago 1 y cae dentro de la banda de
# significancia hacia la cuarta semana; entre los rezagos 5 y 50 solo el 8 y el 12 la rozan (|0,12|), y la
# ACF reaparece con fuerza en los rezagos 51-53 (0,19 a 0,26), la huella del ciclo anual. La PACF está
# dominada por el rezago 1 (0,44); los rezagos 8 y 9 apenas superan la banda (−0,12 y 0,11). Es un patrón
# cercano a un AR(1) más un componente estacional. Dentro de cada corredor, la interrupción de una semana se correlaciona 0,28 con la de la
# semana previa, 0,10 con la de ocho semanas antes y se estabiliza en 0,11-0,13 a 13, 26 y 52 semanas,
# lo que refleja diferencias persistentes entre corredores. **Decisiones:** usar el nivel de la semana
# actual y la anterior relativos al promedio (`ratio_t`, `ratio_t_1`) y ventanas de 8 y 26 semanas, y
# validar con bloques de semanas separados por un hueco.

# %% [markdown]
# :::{admonition} Detalle: deriva temporal
# :class: note
# El índice de estabilidad poblacional (PSI) entre 2019 y cada año posterior es menor que 0,01 en las
# predictoras que describen el nivel, la regularidad, la dinámica y la composición del corredor, y menor
# que 0,05 en la fracción de corredores en caída de la central. Hay dos excepciones con causas conocidas:
#
# * `antig_sem` (semanas desde la primera aparición del corredor) deriva **por construcción**, porque
#   crece con el calendario: su mediana pasa de 75 semanas en 2019 a 328 en 2024 (PSI de 4,6 a 6,3). En la
#   prueba todos los corredores son más antiguos que en el entrenamiento, por lo que su coeficiente
#   extrapola; entra al modelo en logaritmo y su peso es pequeño (−0,04 en log-odds por desviación
#   estándar). En los próximos entregables conviene reemplazarla por una versión acotada (por ejemplo,
#   antigüedad recortada a 52 semanas).
# * `frac_caida_nacional_t` (PSI de 0,24 a 2,80), porque es un único valor por semana que cambia con cada
#   choque; se dejó en observación y el Entregable 1 comprobó que el modelo no depende de ella.
# :::

# %%
F.tabla("deriva_psi").pivot(index="variable", columns="anio", values="psi").round(3)

# %% [markdown]
# ## 2.2 Variables y relaciones con la interrupción
#
# ### Distribuciones por clase y tasa por deciles
#
# El dashboard permite elegir cualquier predictora numérica. Aquí se interpretan las cuatro más
# informativas.

# %%
F.fig_distribucion("cv8")

# %%
F.fig_deciles("cv8")

# %% [markdown]
# **Coeficiente de variación de 8 semanas (`cv8`).** Es la predictora con mayor tamaño de efecto
# (|rbc| = 0,51; AUC univariado 0,753). Su mediana es 0,93 en las semanas previas a una interrupción y
# 0,48 en las demás. La relación es monótona y fuerte: la tasa de interrupción pasa de 2,3 % en el decil
# más estable a 55,1 % en el más variable. Un corredor que ya oscila mucho tiene más probabilidad de caer
# por debajo de la mitad de su promedio.

# %%
F.fig_distribucion("n_envios8", escala_log=True)

# %%
F.fig_deciles("n_envios8")

# %% [markdown]
# **Cargas por semana (`n_envios8`).** La distribución original es muy asimétrica (asimetría 8,0,
# curtosis 90; 12,9 % de valores fuera de 1,5·IQR), por eso el dashboard ofrece la escala logarítmica y el
# modelo usa `log1p`. Los corredores que se interrumpen tienen una mediana de 1,9 cargas semanales frente a
# 6,9 de los demás (|rbc| = 0,49): con una o dos cargas por semana basta que un camión no salga para que el
# corredor «se interrumpa».

# %%
F.fig_deciles("ratio_t")

# %% [markdown]
# **Dinámica reciente (`ratio_t`).** La relación tiene forma de **U**: la tasa de interrupción es 52,7 %
# en el decil más bajo (semanas en que el corredor ya trajo poco o nada), baja a 13,9 % cuando el
# abastecimiento está cerca de su promedio (ratio ≈ 1) y vuelve a subir a 35,4 % tras un pico (ratio ≈ 2,6),
# porque los picos suelen ser seguidos de un regreso brusco. Un modelo lineal en `ratio_t` no puede
# representar una U; por eso el pipeline usa **splines cúbicos** para esta variable y la anterior.

# %%
F.fig_deciles("activas26")

# %% [markdown]
# **Regularidad semestral (`activas26`).** Como la mitad de las filas corresponde a corredores activos
# todas las semanas del último semestre, los deciles se funden en seis grupos. Esos corredores plenamente
# regulares tienen una tasa de 12,6 %; la tasa sube a 30,5 % cuando falta carga en 1 de cada 10 semanas y
# llega a 57,4 % en los corredores activos solo un tercio del tiempo (|rbc| = 0,47).

# %% [markdown]
# :::{admonition} Detalle: estadísticos descriptivos de todas las predictoras
# :class: note
# La tabla resume el conjunto de entrenamiento completo. Escala (`base8`), cargas (`n_envios8`) y
# productos (`n_alimentos8`) son muy asimétricas y se transforman con `log1p`; las razones de cambio tienen
# entre 4 y 5 % de valores fuera de 1,5·IQR, que son picos reales y se recortan en los percentiles 1 y 99
# (winsorización aprendida en entrenamiento) en lugar de eliminarse.
# :::

# %%
F.tabla("descriptivos").set_index("variable")[["mean", "50%", "std", "min", "max", "asimetria", "curtosis", "pct_outliers_iqr"]].round(3)

# %% [markdown]
# ### Variables categóricas

# %%
F.fig_tasa_categoria("grupo_dom")

# %%
F.fig_tasa_categoria("pdet_origen")

# %%
F.tabla("categoricas").round(4)

# %% [markdown]
# **Interpretación.** El grupo de alimentos dominante es la categórica más asociada con la interrupción
# (V de Cramér = 0,13): los corredores de pescados (45,2 %), granos (41,2 %) y procesados (36,2 %) se
# interrumpen más que los de verduras (21,9 %), carnes (23,4 %) y tubérculos (24,1 %), productos de
# abastecimiento continuo. Los corredores que nacen en municipios **PDET** se interrumpen más (33,4 %
# frente a 27,6 %), aunque la asociación es débil (V = 0,04). El capítulo 3 muestra que esa diferencia
# desaparece al controlar por tamaño, regularidad y distancia. Con más de 600 mil filas todas las pruebas
# χ² son significativas; lo informativo es la V de Cramér.

# %% [markdown]
# ### Asociación con el objetivo

# %%
F.fig_asociacion("rbc")

# %%
F.fig_asociacion("info_mutua")

# %%
F.tabla("asociacion_objetivo").query("tipo == 'predictora'").sort_values("auc_univariado", ascending=False)[
    ["variable", "mediana_interr", "mediana_no", "rbc", "p_holm", "auc_univariado", "info_mutua"]].head(15)

# %% [markdown]
# **Interpretación.** Las pruebas de Mann-Whitney, corregidas por comparaciones múltiples con Holm, dan
# p ≈ 0 para casi todas las variables; por eso se ordenan por tamaño del efecto (correlación biserial de
# rangos, |rbc|) y por información mutua, que capta relaciones no lineales. Las variables de **tamaño y
# regularidad** (`cv8`, `n_envios8`, `activas26`, `n_alimentos8`, `activas8`, `log_base8`) dominan con
# |rbc| entre 0,40 y 0,51 y también encabezan la información mutua. La diferencia más notoria entre ambas
# medidas es la **distancia** (`dist_km`): su efecto monótono es pequeño (|rbc| = 0,16), pero es la cuarta
# variable por información mutua (0,077), señal de una relación no lineal que además cambia según la
# región de la central. La **dinámica reciente** (`ratio_t`, `ratio_t_1`, `pend8_rel`) tiene un efecto
# intermedio, y el calendario y la mayoría de las participaciones por grupo aportan poco por sí solos.

# %% [markdown]
# ### Correlación y multicolinealidad

# %%
F.fig_correlacion()

# %%
F.tabla("vif").round(2).head(10)

# %% [markdown]
# **Interpretación.** Las medidas de tamaño y regularidad están muy correlacionadas: `n_envios8` y
# `n_alimentos8` (ρ = 0,91), `activas8` y `activas26` (0,83), `cv8` y `activas8` (−0,82). Aun así, ningún
# VIF supera 5 (el máximo es 4,5, en `activas8`) una vez se omite una de las participaciones por grupo, que
# suman 1 por construcción. Se conservan todas y el modelo usa regularización L2; al interpretar los
# coeficientes se leen por bloques, no de forma aislada.

# %% [markdown]
# :::{admonition} Detalle: análisis multivariado (PCA, outliers y subpoblaciones)
# :class: note
# **PCA.** La primera componente explica 18 % de la varianza y se necesitan 12 de 22 para llegar a 80 %:
# la información está repartida. PC1 es un eje de tamaño y regularidad y PC2 de dinámica reciente; las
# interrupciones se concentran en un lado de PC1 pero se superponen mucho con la clase negativa, lo que
# anticipa un desempeño moderado de un clasificador lineal.
#
# **Outliers multivariados.** La distancia de Mahalanobis robusta marca 47 % de las filas y el Isolation
# Forest 15 %. Una proporción tan alta indica una mezcla de subpoblaciones, no errores: los «atípicos»
# son corredores pequeños (mediana de 1,6 cargas semanales frente a 12,6) con 45 % de interrupciones,
# frente a 13 % en el núcleo. No se eliminan porque son la población de mayor riesgo.
#
# **Subpoblaciones.** K-means sobre el perfil de cada corredor elige k = 2 (silueta 0,30): 2.141 corredores
# pequeños, irregulares y lejanos (51 % de semanas con interrupción, 13,6 % PDET) y 992 consolidados (13,5 %).
# :::

# %%
F.fig_pca()

# %%
pd.DataFrame(E["clusters"]["perfiles"]).round(3)

# %% [markdown]
# ## 2.3 Componente espacial

# %%
F.fig_mapa_tasa("tasa")

# %% [markdown]
# **Interpretación.** El mapa muestra la tasa de interrupción de los 779 municipios con al menos 52
# corredor-semanas. Las tasas municipales más altas, en promedio por departamento, están en La Guajira
# (51 %), Cesar (47 %) y Sucre (46 %), seguidas de Cauca, Córdoba, Tolima y Nariño (37 a 40 %); las más
# bajas, en Norte de Santander, Antioquia, Santander y el Eje Cafetero.

# %%
F.fig_mapa_tasa("lisa")

# %%
F.fig_mapa_tasa("gi")

# %%
pd.Series({"I de Moran (k = 8)": S["moran_k8"], "p (999 permutaciones)": S["moran_p"], "I con k = 4": S["moran_k4"],
           "I con k = 12": S["moran_k12"], "municipios analizados": S["municipios_analizados"]}).to_frame("valor")

# %% [markdown]
# **Interpretación.** La tasa municipal tiene autocorrelación espacial positiva y significativa: I de
# Moran = 0,247 (p = 0,001) con una matriz de 8 vecinos más cercanos, y el resultado se mantiene con 4
# (0,276) y 12 vecinos (0,223). Se usan vecinos más cercanos y no contigüidad porque los municipios de
# origen no cubren todo el territorio y su densidad es muy desigual. El LISA identifica 58 municipios
# Alto-Alto (la mitad en el Caribe, sobre todo en Cesar, y grupos en Cundinamarca, Santander, Cauca y
# Nariño) y 88 Bajo-Bajo (Norte de Santander, Antioquia, Santander y Caldas). El estadístico Gi* de
# Getis-Ord coincide: los hotspots más numerosos están en Cesar (17), Cundinamarca (13, en Gualivá y Rionegro),
# Santander (12, en Vélez y Socorro), Nariño (9, la mayoría en la zona andina), Sucre (9), Cauca (8) y La
# Guajira (8); los coldspots, en Norte de Santander (30), Antioquia (20), Santander (18) y Caldas (13).
# Santander tiene a la vez hotspots y coldspots: la escala municipal revela contrastes que el promedio
# departamental oculta. **Consecuencia:** como los
# municipios cercanos se parecen, el capítulo 3 valida también con bloques espaciales (departamentos no
# vistos).

# %%
F.fig_mapa_tasa("cv8")

# %% [markdown]
# **Interpretación.** El mapa de variabilidad (`cv8`) reproduce buena parte del patrón de la tasa: parte
# de la autocorrelación espacial de las interrupciones se explica porque los corredores pequeños e
# irregulares también se agrupan en el espacio.

# %% [markdown]
# :::{admonition} Detalle: dispersión de Moran, semivariograma, escala y cobertura
# :class: note
# El diagrama de dispersión de Moran muestra la pendiente que define el I global. El semivariograma
# crece hasta unos 125-350 km y luego se estabiliza en la varianza total; el efecto pepita es alto
# (≈ 0,018 de una meseta de ≈ 0,026): buena parte de la variación es propia de cada municipio o corredor.
# Agregado por departamento, el I de Moran cae a 0,01 (p = 0,31): la dependencia opera a escala municipal
# y subregional (problema de la unidad de área modificable). Hay 210 municipios sin corredores regulares:
# en número absoluto, sobre todo en la región Andina (75, muchos de ellos pequeños municipios de Boyacá y
# Antioquia) y el Pacífico (46, la mitad en Chocó); en proporción, la Amazonía es la menos cubierta (67 %
# de sus municipios sin corredores), seguida del Pacífico (26 %). Es un sesgo de cobertura espacial.
# :::

# %%
F.fig_moran_dispersion()

# %%
F.fig_semivariograma()

# %%
pd.Series(S["maup"]).to_frame("valor")

# %% [markdown]
# ### Interacción espacio-tiempo

# %%
pd.DataFrame(S["por_anio"]).round(3)

# %%
pd.Series(S["knox"]).to_frame("prueba de Knox (≤ 50 km y ≤ 1 semana)")

# %% [markdown]
# **Interpretación.** La autocorrelación espacial es significativa todos los años, pero se debilita en
# 2020-2021 (I = 0,19 y 0,18), cuando los choques nacionales generalizaron las interrupciones, y se
# recupera en 2024 (0,22). Los hotspots se desplazan: el índice de Jaccard entre años consecutivos va de
# 0,37 a 0,52. La prueba de Knox indica que los eventos de interrupción masiva (municipio-semana con al
# menos dos tercios de sus corredores interrumpidos) ocurren juntos en espacio y tiempo 1,57 veces más de
# lo esperado bajo independencia (p = 0,01): choques locales que afectan a varios municipios vecinos a la
# vez. En el dashboard esta razón aparece como indicador junto al mapa.

# %% [markdown]
# ## 2.4 Auditoría de fuga de datos

# %%
F.fig_asociacion("auc_univariado")

# %%
F.tabla("asociacion_objetivo").sort_values("auc_univariado", ascending=False)[["variable", "tipo", "auc_univariado"]].head(10)

# %% [markdown]
# **Interpretación.** El AUC univariado de cada variable es un detector simple de fuga: un valor cercano a
# 1 indica que la variable contiene la respuesta. Las variables de control construidas con la semana
# $t+1$ lo confirman: `tasa_t1 / base8` tiene AUC = 1,00 (define el objetivo) y `tasa_t1`, 0,93; ambas
# están excluidas. Ninguna predictora admitida supera 0,753 (`cv8`), un valor coherente con una relación
# real. El identificador del corredor (`cod_mpio`) se excluye para que el modelo no memorice la tasa
# histórica de cada corredor; la central y el departamento se conservan como contexto (AUC con
# codificación por objetivo de 0,56-0,57). Los días de encuesta de $t+1$ solo definen la población en
# riesgo y nunca entran como predictora. Las variables `frac_caida_*` agregan la semana $t$ de todos los
# corredores (no usan $t+1$) y quedan en observación.
#
# **Lista final.** Predictoras: 24 numéricas, 3 binarias y 3 categóricas (tabla del capítulo 3).
# Excluidas: `tasa_t1`, `tasa_t1 / base8`, `cod_mpio`, días de encuesta de $t+1$. En observación:
# `frac_caida_mercado_t`, `frac_caida_nacional_t`.
