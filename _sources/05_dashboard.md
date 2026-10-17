# 5. Desarrollo y despliegue del dashboard

Este capítulo documenta cómo se construyó el dashboard, qué hace cada pestaña y cómo se publicó en
Render siguiendo la guía del curso
([Render y Dash Tools](https://lihkir.github.io/DataVizPython/dash_dataviz.html#render-y-dash-tool) y el
[tutorial de Charming Data](https://www.youtube.com/watch?v=XWJBJoV5yww)).

## 5.1 Arquitectura

El dashboard no entrena modelos ni carga el panel completo de 744 mil filas. Un script previo calcula
todo lo que se muestra y lo guarda en archivos pequeños; el dashboard y este libro leen esos mismos
archivos y dibujan las figuras con las mismas funciones. Esa separación tiene tres ventajas: la app
arranca en segundos, cabe en los 512 MB de memoria del plan gratuito de Render, y cada cifra que se
interpreta en el libro es idéntica a la que aparece en pantalla.

```text
proyecto_sipsa/ (Entregable 1)               entregable2_dashboard/ (este repositorio)
└── panel_corredores.parquet  ──►  scripts/preparar_artefactos.py   (EDA, logística, SVM lineal)
                                   scripts/modelos_adicionales.py   (8 modelos más del curso)
                                        │  validan con bloques temporales y evalúan en 2025
                                        ▼
                                   data/  (parquet, json, joblib; ~10 MB)
                                        │
                     ┌──────────────────┴──────────────────┐
                     ▼                                     ▼
             src/figuras.py  ◄── funciones de figuras ──►  cuadernos/*.py → book/*.ipynb
                     │
                     ▼
             src/app.py (Dash)  ──►  Render: cd src && gunicorn app:server
```

| Archivo | Función |
|---|---|
| `scripts/preparar_artefactos.py` | Lee el panel del Entregable 1, reproduce la partición cronológica, entrena y valida la logística y la SVM lineal, y precalcula agregados, estadísticos, curvas y predicciones de prueba |
| `scripts/modelos_adicionales.py` | Entrena y valida Lasso, k-NN, Naive Bayes, Random Forest, XGBoost, SVM RBF, PCA + logística y logística por segmentos con el mismo esquema, y los integra a los artefactos |
| `src/modelo_utils.py` | Variables, partición, `Winsorizador`, `BloquesTemporales` y preprocesador; lo importan el script, la app y el libro |
| `src/figuras.py` | Todas las figuras de Plotly; una sola definición para el dashboard y el libro |
| `src/app.py` | Aplicación Dash: diseño, pestañas y callbacks; expone `server = app.server` |
| `src/assets/estilo.css` | Estilo visual (Dash carga automáticamente la carpeta `assets/`) |
| `data/` | Artefactos que consume la app |
| `render.yaml`, `requirements.txt` | Configuración del despliegue en Render |

## 5.2 Contenido de las pestañas

**1. Contexto del problema.** Planteamiento, relación con la tesis, definición formal de la variable
objetivo, indicadores de tamaño de los datos, mapa interactivo de los municipios de origen
(diferenciando los PDET), toneladas recibidas por central con selector de año y un corredor de ejemplo
que muestra cómo se marca una interrupción.

**2. Análisis exploratorio.** Cuatro subpestañas:

* *Objetivo y tiempo*: serie semanal de la tasa de interrupción, desagregable por central, región,
  grupo de alimentos u origen PDET (selector de dimensión y de valores); estacionalidad mensual y
  efecto de festivos; ACF, PACF y autocorrelación dentro de cada corredor.
* *Variables y relaciones*: para cualquier predictora numérica, su distribución por clase (escala
  original o logarítmica) y la tasa de interrupción por deciles, con un resumen numérico que se
  actualiza; tasa por categoría; tamaño del efecto o información mutua; matriz de correlación.
* *Espacio*: mapa municipal con cuatro capas (tasa, clusters LISA, hotspots Gi* y variabilidad), con
  el I de Moran y la razón de Knox.
* *Fuga de datos*: AUC univariado de todas las variables, con las variables de control que delatan la
  fuga, y la tabla de decisiones de la auditoría.

**3. Modelos base.** Una comparación de los diez modelos (AUC-ROC o AUC-PR con intervalos de confianza);
un selector de modelo y un deslizador de umbral que recalcula en vivo la matriz de confusión, las
métricas y el número de alertas sobre las 103.466 predicciones de 2025; curvas ROC y PR con las líneas
base; tabla de métricas con intervalos bootstrap y comparaciones de DeLong y McNemar contra la logística;
calibración; coeficientes (modelos lineales) o importancia por permutación (todos); validación temporal y
espacial; curva de aprendizaje; AUC por subgrupo, y un simulador que calcula la probabilidad de
interrupción de un corredor hipotético con el pipeline logístico entrenado.

## 5.3 Interactividad: callbacks

Cada elemento interactivo es un `@app.callback` que recibe el valor de un control (`Input`) y devuelve
una figura o un bloque de texto (`Output`). Los principales:

| Control | Salida | Cálculo |
|---|---|---|
| Pestaña activa | contenido | construye solo la pestaña visible (`suppress_callback_exceptions=True`) |
| Año (contexto) | barras por central | filtra `toneladas_mercado_anio` |
| Dimensión y valores (EDA) | serie semanal | agrega `serie_semanal` y suaviza con media móvil de 13 semanas |
| Variable y escala (EDA) | histograma, deciles y resumen | usa la muestra estratificada y las tablas precalculadas |
| Capa del mapa (EDA) | coropleta | `choropleth_map` con el GeoJSON municipal |
| Métrica de comparación | AUC de los diez modelos con IC | `fig_comparacion_modelos` sobre `metricas_test` |
| Modelo | umbral inicial, descripción y calibración | lee los hiperparámetros elegidos y las curvas de calibración |
| Modelo y umbral | indicadores, matriz, lectura | `metricas_umbral` sobre las predicciones de prueba |
| Modelo, tipo de interpretación y número de variables | coeficientes o importancia, validación cruzada | coeficientes (lineales), importancia por permutación (todos) y resultados por pliegue |
| Subgrupo | AUC por subgrupo | calcula el AUC por grupo con al menos 300 filas |
| Simulador | probabilidad | `predict_proba` del pipeline logístico serializado |

## 5.4 Despliegue en Render

Se siguió el procedimiento del curso. La estructura que Render espera es la que genera Dash Tools:

```text
ml-sipsa-dashboard/
├── render.yaml          configuración del servicio (Blueprint)
├── requirements.txt     dependencias con versiones exactas
├── data/                artefactos
└── src/
    └── app.py           app = Dash(...); server = app.server
```

1. **Comprobar la app en local**: `python src/app.py` y abrir `http://127.0.0.1:8050`.
2. **Variable `server`**: Render ejecuta `gunicorn app:server`, por lo que `app.py` define
   `server = app.server` justo después de crear la aplicación.
3. **`requirements.txt`**: versiones exactas, incluida `scikit-learn`, que debe coincidir con la versión
   con que se serializó el pipeline (si no, `joblib` puede fallar al cargarlo), y `gunicorn`.
4. **`render.yaml`**: servicio web gratuito, `buildCommand: pip install -r requirements.txt`,
   `startCommand: cd src && gunicorn app:server --workers 1 --timeout 120` y la variable
   `PYTHON_VERSION`, que Render exige con la versión completa (por ejemplo `3.11.16`). Se usa un solo
   proceso de gunicorn porque el plan gratuito tiene 512 MB de memoria.
5. **GitHub**: el código se sube a un repositorio público.
6. **Render**: *New → Blueprint*, seleccionar el repositorio y *Apply*. Render instala las dependencias,
   arranca el servicio y asigna una URL `https://<nombre>.onrender.com`.
7. **Actualizaciones**: cada `git push` a `main` dispara un nuevo despliegue (o se usa *Deploy latest
   commit*).

En el plan gratuito el servicio se suspende tras unos 15 minutos sin visitas; la primera carga
posterior tarda alrededor de un minuto mientras se reactiva.

## 5.5 Reproducibilidad

```bash
python3.11 -m venv ~/.venvs/ml-sipsa-jb
~/.venvs/ml-sipsa-jb/bin/pip install -r requirements-dev.txt
# 1) artefactos (requiere el panel del Entregable 1); el segundo script tarda unas dos horas
~/.venvs/ml-sipsa-jb/bin/python scripts/preparar_artefactos.py --proyecto "../proyecto_sipsa"
~/.venvs/ml-sipsa-jb/bin/python scripts/modelos_adicionales.py --proyecto "../proyecto_sipsa"
# 2) dashboard local
~/.venvs/ml-sipsa-jb/bin/python src/app.py
# 3) libro
for n in 01_contexto 02_eda 03_modelos_base 04_modelos_adicionales; do
  ~/.venvs/ml-sipsa-jb/bin/python scripts/py2nb.py cuadernos/$n.py book/$n.ipynb --ejecutar
done
~/.venvs/ml-sipsa-jb/bin/jupyter-book build book/
```

Semilla aleatoria global: `42`. Versión de Python: 3.11.
