# %% [markdown]
# # 1. Contexto del problema
#
# Este capítulo acompaña la pestaña **1. Contexto del problema** del dashboard. Reproduce cada figura
# de esa pestaña con las mismas funciones (`src/figuras.py`) y las mismas cifras (`data/`), las
# interpreta y añade los detalles que no caben en pantalla.

# %%
import sys

sys.path.insert(0, "../src")
import pandas as pd
import plotly.io as pio

import figuras as F

pio.renderers.default = "notebook_connected"
R = F.resumen()
K = R["kpis"]
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

# %% [markdown]
# ## 1.1 El problema y su relación con la tesis
#
# La propuesta doctoral *Arquitectura de gobernanza para la gestión de proyectos de logística
# inteligente: optimización multicriterio basada en deep learning para la resiliencia alimentaria en
# territorios PDET y ZOMAC* necesita un componente que anticipe riesgos de desabastecimiento. Las
# centrales mayoristas concentran el flujo de alimentos hacia las ciudades y reciben carga de cientos
# de municipios. Cuando un **corredor** (municipio de origen → central) se interrumpe, la oferta de la
# ciudad se resiente y el municipio pierde su principal mercado.
#
# **Pregunta.** Con la información disponible al cierre de una semana, ¿se puede anticipar qué
# corredores sufrirán una interrupción la semana siguiente?
#
# Un modelo que ordene bien ese riesgo permitiría a una secretaría de agricultura o a una central
# priorizar apoyo logístico (acopio, transporte, rutas alternas) antes de que la caída ocurra. Este
# entregable presenta la línea base de ese modelo en un dashboard interactivo.

# %% [markdown]
# ## 1.2 Definición de la variable objetivo
#
# Para cada corredor $c$ y semana $t$, sea $r_{c,t}$ el abastecimiento en kilogramos dividido entre
# los días en que el DANE encuestó la central esa semana. La variable objetivo es
#
# $$
# y_{c,t+1}=\mathbb{1}\left[\,r_{c,t+1}<0{,}5\cdot\bar r_{c,\,t-7:t}\,\right],
# $$
#
# donde $\bar r_{c,\,t-7:t}$ es el promedio de las ocho semanas anteriores. Normalizar por días de
# encuesta evita confundir una semana con menos días de recolección con una caída real. Solo se
# evalúan **corredores regulares**: los que tuvieron carga en al menos 4 de las 8 semanas previas y
# cuya central fue encuestada en $t+1$. El panel resultante es un problema de **clasificación binaria
# sobre un panel espacio-temporal** (corredores × semanas, con coordenadas de origen y destino).

# %%
kpis = pd.Series({
    "Cargas registradas por el DANE (2018-2025)": K["registros_crudos"],
    "Corredor-semanas en el panel": K["observaciones_panel"],
    "Corredores": K["corredores"],
    "Municipios de origen": K["municipios_origen"],
    "Centrales mayoristas": K["centrales"],
    "Semanas": K["semanas"],
    "Filas de entrenamiento (semanas objetivo hasta 27/10/2024)": K["filas_train"],
    "Filas de prueba (2025)": K["filas_test"],
    "Municipios PDET oficiales": K["pdet_oficiales"],
    "Municipios PDET con corredores regulares": K["pdet_en_sipsa"],
})
kpis.to_frame("valor").style.format("{:,.0f}")

# %%
print(f"Tasa de interrupción: total {K['tasa_total']:.3f}; entrenamiento {K['tasa_train']:.3f}; prueba {K['tasa_test']:.3f}")

# %% [markdown]
# **Interpretación de los indicadores.** Las 15,5 millones de cargas se agregan en 744.921
# corredor-semanas de 4.512 corredores. Esas filas no son independientes: un mismo corredor aparece
# muchas semanas y los corredores vecinos se parecen, de modo que el tamaño efectivo de la muestra
# está entre el número de corredores y el de filas. Por eso la validación y los intervalos de
# confianza se construyen por bloques de semanas (capítulo 3). La interrupción ocurre en 27,9 % de
# las corredor-semanas: es un desbalance moderado (2,5 a 1) que vuelve engañosa la exactitud y obliga a
# reportar AUC, AUC-PR, F1 y recall.
#
# De los 170 municipios PDET, 122 tienen al menos un corredor regular hacia alguna central. En el
# Entregable 1 se reportó que 160 aparecen alguna vez como origen; la diferencia son los municipios
# que envían carga de forma esporádica y nunca alcanzan la regularidad que exige la población en riesgo.

# %% [markdown]
# ## 1.3 Dónde nacen los corredores

# %%
F.fig_mapa_origenes()

# %% [markdown]
# **Interpretación.** La oferta que llega a las centrales se concentra en el altiplano
# cundiboyacense, Antioquia, el Eje Cafetero y Nariño, que forman las burbujas más grandes. Los
# municipios PDET (naranja) son numerosos en el Caribe, el Pacífico y el sur, pero sus burbujas son
# pequeñas: aportan muchos corredores con poco volumen. Esa diferencia anticipa uno de los resultados
# centrales del EDA, que los corredores pequeños son los que más se interrumpen. Amplias zonas de la
# Orinoquía, la Amazonía y el Pacífico no tienen corredores regulares: el modelo no aprende nada sobre
# ellas, una limitación de cobertura del SIPSA-A.

# %%
F.fig_toneladas_mercado(2024)

# %% [markdown]
# **Interpretación.** En 2024 Corabastos (Bogotá) recibió 32 % del volumen nacional, 2,3 veces lo que
# recibió la segunda central, y las cinco primeras (Corabastos, la Central Mayorista de Antioquia,
# Centroabastos, Barranquillita y Bazurto) concentraron 63 %. Un modelo ajustado por filas estará
# dominado por los corredores hacia estas centrales; por eso el capítulo 3 evalúa el desempeño por
# central y por región, no solo en el agregado. Con el selector de año del dashboard se ve que las
# cuatro primeras mantienen su orden entre 2018 y 2025, mientras el quinto puesto cambia: Cavasa (Cali)
# en 2018, Cenabastos (Cúcuta) en 2021 y Bazurto (Cartagena) en 2024-2025.

# %% [markdown]
# ## 1.4 Dos corredores de ejemplo

# %%
F.fig_corredor_ejemplo("Estable")

# %%
F.fig_corredor_ejemplo("Irregular")

# %%
pd.DataFrame(K["ejemplos"])[["tipo", "origen", "depto", "central", "t_dia", "tasa_interrupcion", "semanas"]]

# %% [markdown]
# **Interpretación.** Los dos corredores llegan a Bazurto (Cartagena) y se eligieron con reglas
# explícitas sobre 2023-2024: el estable es el de mayor volumen (Tunja, 91 t/día) y el irregular, el
# de mayor tasa de interrupción entre los que mueven al menos 5 t/día (Palmira, 7,4 t/día). El primero
# casi nunca cae por debajo del umbral (2,1 % de las semanas); el segundo se interrumpe en 39,5 % de
# las semanas, con caídas a cero seguidas de recuperación. La figura muestra cómo funciona la
# definición: cada cruz roja es una semana en que la carga quedó por debajo de la línea punteada (la
# mitad del promedio móvil). Se eligieron corredores grandes porque agregan muchas cargas y no
# exponen la actividad de un productor individual.

# %% [markdown]
# ## 1.5 Detalles que no aparecen en el dashboard
#
# **Construcción del panel.** Los 18 archivos de microdatos del DANE (2018-2025) cambiaron cuatro
# veces de encabezados, de codificación de texto y de formato numérico; el Entregable 1 los
# normalizó, unificó dos nombres de central escritos de forma distinta, excluyó las importaciones
# (4,7 % del volumen) y agregó las cargas por corredor y semana. Todas las predictoras se calculan con
# ventanas que terminan en la semana $t$.
#
# **Partición.** La prueba son las semanas objetivo de 2025 y el entrenamiento, las semanas objetivo
# hasta el 27 de octubre de 2024. Entre ambos se deja un hueco de 10 semanas, mayor que la ventana de 8
# semanas de las predictoras, para que ninguna fila de prueba use semanas que fueron objetivo en
# entrenamiento.
#
# **Ética.** SIPSA-A no contiene datos personales. El dashboard solo muestra agregados municipales y
# corredores de gran volumen. Un modelo que etiquete corredores PDET como frágiles podría reforzar
# estigmas territoriales; sus resultados deben usarse para priorizar apoyo, no para excluir
# territorios de mercados o inversiones.
