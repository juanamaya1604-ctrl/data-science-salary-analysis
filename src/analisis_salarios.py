import pandas as pd
from pathlib import Path

# Encontrar la carpeta principal del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Ruta del dataset
DATA_PATH = BASE_DIR / "data" / "ds_salaries.csv"

# Leer el dataset
df = pd.read_csv(DATA_PATH)

# Mostrar información básica
print("¡Dataset cargado correctamente!")
print("Número de filas:", df.shape[0])
print("Número de columnas:", df.shape[1])

print("\nColumnas del dataset:")
print(df.columns.tolist())

print("\nPrimeras 5 filas:")
print(df.head())
# ==========================================================
# ETAPA 1: EXPLORACIÓN Y LIMPIEZA DE LOS DATOS
# ==========================================================

print("\n--- EXPLORACIÓN DE LOS DATOS ---")

# Revisar valores faltantes
print("\nValores faltantes por columna:")
print(df.isnull().sum())

# Revisar filas duplicadas
duplicados = df.duplicated().sum()
print("\nNúmero de filas duplicadas:", duplicados)

# Eliminar duplicados
df = df.drop_duplicates().copy()

print("Número de filas después de eliminar duplicados:", df.shape[0])

# Estadísticas básicas de los salarios
print("\n--- ESTADÍSTICAS DE SALARIO ---")
print(df["salary_in_usd"].describe())

# Convertir códigos de experiencia a nombres fáciles de entender
experience_map = {
    "EN": "Entry-level",
    "MI": "Mid-level",
    "SE": "Senior-level",
    "EX": "Executive-level"
}

df["experience_name"] = df["experience_level"].map(experience_map)

# Mostrar salario promedio según nivel de experiencia
salario_experiencia = (
    df.groupby("experience_name")["salary_in_usd"]
    .mean()
    .sort_values()
)

print("\nSalario promedio por nivel de experiencia:")
print(salario_experiencia)

print("\n¡Limpieza y exploración completadas!")
# ==========================================================
# ETAPA 2: VISUALIZACIÓN DE LOS DATOS
# ==========================================================

import matplotlib.pyplot as plt

# Orden correcto de los niveles de experiencia
orden_experiencia = [
    "Entry-level",
    "Mid-level",
    "Senior-level",
    "Executive-level"
]

# Calcular salario promedio
grafico_experiencia = (
    df.groupby("experience_name")["salary_in_usd"]
    .mean()
    .reindex(orden_experiencia)
)

# Crear gráfico
plt.figure(figsize=(10, 6))

bars = plt.bar(
    grafico_experiencia.index,
    grafico_experiencia.values
)

plt.title(
    "Salario promedio por nivel de experiencia",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Nivel de experiencia")
plt.ylabel("Salario promedio anual (USD)")

# Mostrar el valor encima de cada barra
for bar in bars:
    altura = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        altura + 3000,
        f"${altura:,.0f}",
        ha="center",
        fontweight="bold"
    )

plt.tight_layout()

# Guardar gráfico
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

plt.savefig(
    FIGURES_DIR / "01_salario_experiencia.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\n¡Primer gráfico creado correctamente!")
# ==========================================================
# GRÁFICO 2: CARGOS CON MAYOR SALARIO PROMEDIO
# ==========================================================

# Contar cuántos registros tiene cada cargo
conteo_cargos = df["job_title"].value_counts()

# Seleccionar cargos con al menos 10 registros
cargos_validos = conteo_cargos[conteo_cargos >= 10].index

# Calcular salario promedio de esos cargos
top_cargos = (
    df[df["job_title"].isin(cargos_validos)]
    .groupby("job_title")["salary_in_usd"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .sort_values()
)

print("\nTop 10 cargos con mayor salario promedio:")
print(top_cargos.sort_values(ascending=False))

# Crear gráfico horizontal
plt.figure(figsize=(11, 7))

bars = plt.barh(
    top_cargos.index,
    top_cargos.values
)

plt.title(
    "Top 10 cargos con mayor salario promedio",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Salario promedio anual (USD)")
plt.ylabel("Cargo")

# Mostrar salario al lado de cada barra
for bar in bars:
    ancho = bar.get_width()

    plt.text(
        ancho + 2000,
        bar.get_y() + bar.get_height() / 2,
        f"${ancho:,.0f}",
        va="center",
        fontweight="bold"
    )

plt.tight_layout()

# Guardar gráfico
plt.savefig(
    FIGURES_DIR / "02_top_cargos.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\n¡Segundo gráfico creado correctamente!")
# ==========================================================
# GRÁFICO 3: MODALIDAD DE TRABAJO Y SALARIO
# ==========================================================

# Convertir remote_ratio a nombres fáciles de entender
modalidad_map = {
    0: "Presencial",
    50: "Híbrido",
    100: "Remoto"
}

df["modalidad_trabajo"] = df["remote_ratio"].map(modalidad_map)

# Calcular salario promedio por modalidad
salario_modalidad = (
    df.groupby("modalidad_trabajo")["salary_in_usd"]
    .mean()
    .reindex(["Presencial", "Híbrido", "Remoto"])
)

print("\nSalario promedio según modalidad de trabajo:")
print(salario_modalidad)

# Crear gráfico
plt.figure(figsize=(9, 6))

bars = plt.bar(
    salario_modalidad.index,
    salario_modalidad.values
)

plt.title(
    "Salario promedio según modalidad de trabajo",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Modalidad de trabajo")
plt.ylabel("Salario promedio anual (USD)")

# Mostrar valores sobre las barras
for bar in bars:
    altura = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        altura + 2500,
        f"${altura:,.0f}",
        ha="center",
        fontweight="bold"
    )

plt.tight_layout()

# Guardar gráfico
plt.savefig(
    FIGURES_DIR / "03_salario_modalidad.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\n¡Tercer gráfico creado correctamente!")
# ==========================================================
# ETAPA 3: MODELO PREDICTIVO - REGRESIÓN LINEAL
# ==========================================================

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

print("\n--- MODELO PREDICTIVO ---")

# Variable que queremos predecir
y = df["salary_in_usd"]

# Variables que utilizaremos para hacer la predicción
variables_modelo = [
    "work_year",
    "experience_level",
    "employment_type",
    "job_title",
    "employee_residence",
    "remote_ratio",
    "company_location",
    "company_size"
]

X = df[variables_modelo]

# IMPORTANTE:
# No utilizamos 'salary' ni 'salary_currency'
# porque contienen información directa sobre el salario real.
# Usarlas produciría data leakage.

# Separar variables numéricas y categóricas
variables_numericas = [
    "work_year",
    "remote_ratio"
]

variables_categoricas = [
    "experience_level",
    "employment_type",
    "job_title",
    "employee_residence",
    "company_location",
    "company_size"
]

# Preparar los datos
preprocesador = ColumnTransformer(
    transformers=[
        ("numericas", StandardScaler(), variables_numericas),
        (
            "categoricas",
            OneHotEncoder(handle_unknown="ignore"),
            variables_categoricas
        )
    ]
)

# Crear modelo de regresión lineal
modelo = Pipeline(
    steps=[
        ("preprocesamiento", preprocesador),
        ("regresion", LinearRegression())
    ]
)

# Dividir datos:
# 80% para entrenar el modelo
# 20% para probarlo
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Datos para entrenamiento:", X_train.shape[0])
print("Datos para prueba:", X_test.shape[0])

# Entrenar modelo
modelo.fit(X_train, y_train)

# Realizar predicciones
predicciones = modelo.predict(X_test)

# Evaluar modelo
mae = mean_absolute_error(y_test, predicciones)
rmse = np.sqrt(mean_squared_error(y_test, predicciones))
r2 = r2_score(y_test, predicciones)

print("\n--- RESULTADOS DEL MODELO ---")
print(f"MAE: ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R²: {r2:.3f}")
# ==========================================================
# GRÁFICO 4: SALARIOS REALES VS. PREDICHOS
# ==========================================================

plt.figure(figsize=(8, 7))

plt.scatter(
    y_test,
    predicciones,
    alpha=0.5
)

# Crear línea de predicción perfecta
valor_minimo = min(y_test.min(), predicciones.min())
valor_maximo = max(y_test.max(), predicciones.max())

plt.plot(
    [valor_minimo, valor_maximo],
    [valor_minimo, valor_maximo],
    linestyle="--"
)

plt.title(
    "Salarios reales vs. salarios predichos",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Salario real (USD)")
plt.ylabel("Salario predicho (USD)")

plt.tight_layout()

# Guardar gráfico
plt.savefig(
    FIGURES_DIR / "04_reales_vs_predichos.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\n¡Modelo predictivo y visualización completados!")