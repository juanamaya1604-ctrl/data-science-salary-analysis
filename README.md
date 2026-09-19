# Data Science Salary Analysis

## Análisis y predicción de salarios en Data Science

Proyecto desarrollado para aplicar conceptos de Ciencia de Datos mediante
Python, análisis exploratorio, visualización de datos y Machine Learning.

### Objetivo

Analizar qué factores están asociados con los salarios de profesionales de
Data Science y evaluar si es posible predecir el salario utilizando
características profesionales y laborales.

## Dataset

Se utilizó el dataset **Data Science Salaries 2023**, compuesto originalmente por:

- 3,755 registros
- 11 variables
- Información sobre experiencia, cargo, modalidad de trabajo, ubicación,
  tamaño de empresa y salario.

Durante la preparación de los datos se identificaron **1,171 registros
duplicados**. Después de eliminarlos, el análisis se realizó sobre
**2,584 registros únicos**.

## 🔎 Principales hallazgos

### Experiencia
El salario promedio aumenta considerablemente según el nivel de experiencia.

- Entry-level: **$72,649 USD**
- Mid-level: **$101,829 USD**
- Senior-level: **$153,897 USD**
- Executive-level: **$191,078 USD**

### Cargo
Entre los cargos analizados, **Director of Data Science** presentó uno de los
salarios promedio más altos, con aproximadamente **$195,141 USD anuales**.

### Modalidad de trabajo
Los salarios promedio observados fueron:

- Presencial: **$143,690 USD**
- Remoto: **$131,822 USD**
- Híbrido: **$78,487 USD**

Estas diferencias representan asociaciones observadas en el dataset y no
necesariamente relaciones causales.

## Modelo predictivo

Se entrenó un modelo de **Regresión Lineal**, utilizando una división
aproximada de 80% de los datos para entrenamiento y 20% para prueba.

Resultados:

- **MAE:** $40,002
- **RMSE:** $53,230
- **R²:** 0.352

El modelo explica aproximadamente el **35.2% de la variación salarial** en los
datos de prueba, lo que indica que captura parte de los patrones salariales,
aunque existen otros factores que no están representados en el dataset.

## Tecnologías utilizadas

- Python
- pandas
- NumPy
- Matplotlib
- scikit-learn
- Visual Studio Code

##  Estructura del proyecto

- `src/` — código del análisis y modelo predictivo
- `figures/` — visualizaciones generadas
- `ds_salaries.csv` — dataset utilizado
- `Presentacion_Proyecto_Ciencia_Datos.pdf` — presentación final

##  Autores

**Juan David Pardo Abella**  
**Juan Esteban Amaya Agudelo**

