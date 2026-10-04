# Entregable 2 · Dashboard de interrupciones en corredores de abastecimiento (SIPSA-A)

Dashboard en **Dash y Plotly** con tres pestañas (contexto del problema, análisis exploratorio y modelos
base) e informe del desarrollo en **Jupyter Book**.

**Autores:** Camilo Molina Guerrero y Lina Margarita Buelvas · Doctorado en Ingeniería, curso de Machine Learning

- **Dashboard:** https://ml-sipsa-dashboard.onrender.com
- **Informe (Jupyter Book):** https://camilomolinaguerrero-sudo.github.io/ml-sipsa-dashboard/
- **Entregable 1:** https://camilomolinaguerrero-sudo.github.io/ml-sipsa-abastecimiento/

## Estructura

```
ml-sipsa-dashboard/
├── render.yaml                 despliegue en Render (Blueprint)
├── requirements.txt            dependencias del dashboard (versiones exactas)
├── requirements-dev.txt        + artefactos, cuadernos y Jupyter Book
├── src/
│   ├── app.py                  aplicación Dash (server = app.server)
│   ├── figuras.py              figuras de Plotly compartidas con el libro
│   ├── modelo_utils.py         partición, variables, Winsorizador, BloquesTemporales, pipeline
│   └── assets/estilo.css       estilo
├── data/                       artefactos (agregados, métricas, predicciones, pipeline logístico)
├── scripts/
│   ├── preparar_artefactos.py  entrena y valida los modelos y precalcula todo lo que se muestra
│   └── py2nb.py                convierte cuadernos/*.py en .ipynb y los ejecuta
├── cuadernos/                  fuente de los capítulos del libro (formato # %%)
└── book/                       Jupyter Book 1.x (_config.yml, _toc.yml, capítulos)
```

## Ejecutar en local

```bash
python3.11 -m venv ~/.venvs/ml-sipsa-jb
~/.venvs/ml-sipsa-jb/bin/pip install -r requirements-dev.txt
~/.venvs/ml-sipsa-jb/bin/python src/app.py          # http://127.0.0.1:8050
```

Para regenerar los artefactos se necesita el panel del Entregable 1
(`proyecto_sipsa/data/processed/panel_corredores.parquet`):

```bash
~/.venvs/ml-sipsa-jb/bin/python scripts/preparar_artefactos.py --proyecto "../proyecto_sipsa"
for n in 01_contexto 02_eda 03_modelos_base; do
  ~/.venvs/ml-sipsa-jb/bin/python scripts/py2nb.py cuadernos/$n.py book/$n.ipynb --ejecutar
done
~/.venvs/ml-sipsa-jb/bin/jupyter-book build book/
~/.venvs/ml-sipsa-jb/bin/ghp-import -n -p -f book/_build/html   # publica el libro en GitHub Pages
```

## Despliegue en Render

Siguiendo la guía del curso (Dash Tools + Render): el repositorio contiene `render.yaml`,
`requirements.txt` y `src/app.py` con `server = app.server`. En Render: *New → Blueprint*, elegir este
repositorio y *Apply*. Cada `git push` a `main` vuelve a desplegar.

Semilla aleatoria: 42 · Python 3.11.16
