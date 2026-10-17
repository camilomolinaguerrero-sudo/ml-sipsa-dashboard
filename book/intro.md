# Dashboard de interrupciones en corredores de abastecimiento de alimentos

**Segundo entregable del proyecto de Machine Learning** · Doctorado en Ingeniería

**Autores:** Camilo Molina Guerrero y Lina Margarita Buelvas

Dashboard en Dash y Plotly con tres pestañas (contexto, análisis exploratorio y modelos base, con diez
modelos comparados) e informe del desarrollo

---

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} 📊 Dashboard
:link: https://ml-sipsa-dashboard.onrender.com
Aplicación interactiva desplegada en Render
:::

:::{grid-item-card} 💻 Código
:link: https://github.com/camilomolinaguerrero-sudo/ml-sipsa-dashboard
Repositorio con la app, los datos y este libro
:::

:::{grid-item-card} 📘 Entregable 1
:link: https://camilomolinaguerrero-sudo.github.io/ml-sipsa-abastecimiento/
Base de datos, EDA completo y modelo base
:::
::::

## De qué trata

La tesis doctoral en la que se inscribe el proyecto propone una arquitectura de gobernanza para la
logística alimentaria en territorios PDET y ZOMAC, y necesita anticipar riesgos de desabastecimiento.
Con los microdatos del componente de abastecimiento del SIPSA (DANE), que registran cada carga que
llega a las 32 principales centrales mayoristas del país, se construyó un panel semanal de corredores
(municipio de origen → central). La pregunta es si se puede anticipar, con la información disponible
al cierre de una semana, qué corredores sufrirán una **interrupción** la semana siguiente (una caída de
más de la mitad frente a su promedio de 8 semanas).

Este entregable presenta el análisis y los modelos base en un **dashboard interactivo** y documenta en
este libro el desarrollo: la interpretación de cada resultado que se muestra en pantalla y los
detalles metodológicos que no caben en el dashboard.

## Resultados principales

| Aspecto | Resultado |
|---|---|
| Datos | 15,5 millones de cargas (2018-2025) → 744.921 corredor-semanas de 4.512 corredores |
| Objetivo | 27,9 % de las corredor-semanas presentan interrupción en la semana siguiente |
| Partición | Cronológica: entrenamiento hasta octubre de 2024 y prueba con todo 2025, separados por un hueco de 10 semanas |
| Línea base trivial | Dummy: AUC 0,50; persistencia: AUC 0,584 |
| Regresión logística | AUC-ROC 0,793 (IC 95 %: 0,789-0,797); AUC-PR 0,539; recall 0,76 con umbral 0,26 |
| SVM lineal | AUC-ROC 0,795 (IC 95 %: 0,792-0,799); diferencia con la logística trivial (0,002) |
| **XGBoost (mejor modelo)** | AUC-ROC 0,806 (IC 95 %: 0,802-0,809); AUC-PR 0,569; mejora pequeña (+0,013) y consistente sobre la logística |
| Otros modelos del curso | Random Forest 0,799; SVM RBF 0,798; logística por segmentos 0,795; Lasso 0,793; PCA + logística 0,792; k-NN 0,787; Naive Bayes 0,765 |
| Departamentos no vistos | AUC 0,792 (XGBoost), 0,780 (logística) y 0,783 (SVM lineal) |

Los corredores pequeños e irregulares son los que más se interrumpen. Los que nacen en municipios PDET
se interrumpen más (33 % frente a 28 %), pero la diferencia desaparece al controlar por tamaño,
regularidad y distancia: su fragilidad es logística.

## Cómo leer este libro

Cada capítulo corresponde a una pestaña del dashboard y reproduce sus figuras con las mismas funciones
y los mismos datos:

1. **Contexto del problema**: pregunta, definición de la variable objetivo, datos y corredores de ejemplo.
2. **Análisis exploratorio**: objetivo y tiempo, variables y relaciones, espacio y auditoría de fuga.
3. **Modelos base**: preprocesamiento, validación, umbrales, métricas con intervalos de confianza,
   comparaciones, coeficientes, validación espacial, curva de aprendizaje y residuos.
4. **Modelos adicionales y comparación**: Lasso, k-NN, Naive Bayes, Random Forest, XGBoost, SVM RBF, PCA +
   logística y logística por segmentos frente a la logística.
5. **Desarrollo y despliegue del dashboard**: arquitectura, callbacks y despliegue en Render.
6. **Conclusiones y limitaciones**.

Las figuras de este libro son interactivas, igual que en el dashboard: se puede pasar el cursor para
ver valores, hacer zoom y ocultar series desde la leyenda.
