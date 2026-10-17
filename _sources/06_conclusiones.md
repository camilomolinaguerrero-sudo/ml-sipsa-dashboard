# 6. Conclusiones y limitaciones

## Conclusiones

1. **El dashboard cumple su función de comunicar el problema y los modelos.** En tres pestañas presenta
   el contexto, el análisis exploratorio y los modelos base, con controles que permiten explorar la
   serie por central o región, cualquier predictora, cuatro capas espaciales, el umbral de decisión y un
   corredor hipotético. Todo lo que muestra sale de los mismos artefactos que interpreta este libro.
2. **La interrupción tiene estructura aprovechable.** Depende sobre todo del tamaño y la regularidad del
   corredor (los corredores con pocas cargas y alta variabilidad se interrumpen hasta 55 % de las
   semanas), de su dinámica reciente (relación en U), del calendario (festivos, Semana Santa, fin de año)
   y de la región. Tiene dependencia temporal (0,28 a una semana dentro de cada corredor) y espacial
   (I de Moran 0,25), y los eventos masivos se agrupan en espacio y tiempo.
3. **Los modelos base son sólidos y honestos.** La regresión logística (AUC 0,793) y la SVM lineal (AUC
   0,795) superan con amplitud a los Dummy (0,50) y a la persistencia (0,58), generalizan a departamentos
   no vistos (0,78), están bien calibradas y no muestran sobreajuste. Entre ellas la diferencia es
   trivial (0,002 de AUC; McNemar p = 0,13): la forma lineal, no la función de pérdida, fija el techo.
4. **Un modelo flexible rompe ese techo, aunque por poco.** De los ocho modelos adicionales del curso,
   XGBoost es el mejor (AUC 0,806; AUC-PR 0,569; Brier 0,150): mejora a la logística en 0,013, una diferencia
   pequeña según la escala del curso pero consistente en todas las regiones, grupos de alimentos y
   validaciones, y detecta 841 interrupciones más en 2025 con la misma precisión. Su curva de aprendizaje
   sigue subiendo con más datos, a diferencia de la de los lineales. Random Forest y la SVM RBF mejoran de
   forma trivial; Lasso, PCA + logística y la logística por segmentos no aportan; k-NN es costoso y algo
   peor, y Naive Bayes falla por la correlación entre predictoras.
5. **Hallazgo para la tesis.** Los corredores que nacen en municipios PDET se interrumpen más, pero el
   efecto PDET desaparece al controlar por escala, regularidad y distancia. Las intervenciones que
   consoliden cargas y regularicen despachos atacan la causa de su fragilidad.

## Limitaciones

* **Cobertura.** SIPSA-A observa 32 centrales urbanas. No mide mercados locales, plazas municipales ni
  venta directa, más relevantes en los territorios PDET; 210 municipios no tienen corredores regulares.
* **Definición del objetivo.** El umbral de 50 % y la ventana de 8 semanas son decisiones del
  investigador; otros valores cambiarían la prevalencia y el desempeño.
* **Choques no observados.** Clima, orden público, cierres viales y precios no están en el modelo y
  explican los picos de 2020 y 2021.
* **Desempeño desigual.** El AUC es menor en corredores de pescados y granos, en la Orinoquía y en las
  plazas minoristas de Bogotá.
* **Despliegue gratuito.** El plan gratuito de Render suspende el servicio tras unos 15 minutos sin
  visitas; la primera carga posterior tarda alrededor de un minuto.

## Siguientes pasos

* Ajuste más fino de XGBoost (más árboles con parada temprana temporal, rejillas más amplias) y
  modelos más expresivos para la tesis, incluido deep learning, comparados contra XGBoost.
* Variables de vecindad espacial y fuentes externas (IDEAM, INVÍAS) para los choques locales.
* Análisis de sensibilidad del umbral y de la ventana que definen la interrupción.

## Referencias

* DANE (2018-2025). *Sistema de Información de Precios y Abastecimiento del Sector Agropecuario,
  componente Abastecimiento de Alimentos (SIPSA-A)*. <https://microdatos.dane.gov.co/index.php/catalog/697>
* Agencia de Renovación del Territorio. *Municipios PDET*. <https://www.datos.gov.co/d/idrk-ba8y>
* Rubio, L. *Machine Learning* (notas del curso). <https://lihkir.github.io/MachineLearning/>
* Rubio, L. *Visualización de datos con Python: Dash y despliegue en Render*.
  <https://lihkir.github.io/DataVizPython/dash_dataviz.html>
* Sun, X. y Xu, W. (2014). Fast implementation of DeLong's algorithm for comparing the areas under
  correlated receiver operating characteristic curves. *IEEE Signal Processing Letters*, 21(11),
  1389-1393.
* Plotly Technologies. *Dash documentation*. <https://dash.plotly.com/>
