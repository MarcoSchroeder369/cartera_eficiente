import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import joblib
import math
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation

# Ignorar warnings de ajuste (fit) en modelos de sklearn
import warnings
warnings.filterwarnings("ignore", category=FitFailedWarning)

# Empresas chilenas en yahoo-finance
empresas_chilenas = [
    'SECURITY.SN', 'SQM-B.SN', 'LTM.SN', 'CMPC.SN', 'CHILE.SN', 'ENTEL.SN', 'ENELAM.SN', 'IAM.SN',
    'BSANTANDER.SN', 'VAPORES.SN', 'PARAUCO.SN', 'CCU.SN', 'MALLPLAZA.SN', 'SONDA.SN', 'COLBUN.SN', 'FALABELLA.SN',
    'ANDINA-B.SN', 'CONCHATORO.SN', 'BCI.SN', 'COPEC.SN', 'RIPLEY.SN', 'ILC.SN', 'CAP.SN', 'AGUAS-A.SN', 'ECL.SN',
    'CENCOSUD.SN', 'ENELCHILE.SN'
]

# Obtener datos de las empresas en el período específico
start_date = '2021-01-01'
end_date = '2023-02-28'

# Crear DataFrame para almacenar datos
data = pd.DataFrame()

# Obtener datos de las empresas
for x in empresas_chilenas:
    stock = yf.Ticker(x)
    precio_cierre = stock.history(start=start_date, end=end_date)['Close']
    data[x] = precio_cierre

# Calcular los retornos
data_retorno = data.pct_change().dropna()

# Seleccionar las 'n_empresas' con mejor rendimiento
n_empresas = 11

# Filtrar empresas que tengan rendimientos disponibles
empresas_con_rendimientos = data_retorno.mean().dropna().index

# Tomar las 'n_empresas' primeras empresas con mejor rendimiento ponderado
acciones_mejor_rendimiento = data_retorno[empresas_con_rendimientos].mean().sort_values(ascending=False).head(n_empresas)

# Mostrar las 'n_empresas' acciones con mejor rendimiento ponderado
print(f"\nLas {n_empresas} acciones con mejor rendimiento ponderado:")
print(acciones_mejor_rendimiento)

# Seleccionar las 'n_empresas' con mejor rendimiento
portafolio = acciones_mejor_rendimiento.index.tolist()

# Mostrar datos
print(f"\nEmpresas seleccionadas para el portafolio ({n_empresas}):")
print(portafolio)

# Obtener precios y retornos de acciones
stock_precio = data[portafolio]
stock_retorno = stock_precio.pct_change().dropna()

# Calcular rendimientos esperados y riesgos
esperados_stock_retorno = stock_retorno.mean()
stock_riesgo_individual = stock_retorno.std()

# Obtener matriz de covarianza
stock_retorno_esperados_cov_matriz = stock_retorno.cov()

# Mostrar datos
print("Rendimientos esperados:")
print(esperados_stock_retorno)

print("Riesgo individual:")
print(stock_riesgo_individual)

print("Matriz de covarianza de rendimientos de activos:")
display(stock_retorno_esperados_cov_matriz)

# Función para graficar series temporales
def graficar_series_temporales(data, columnas, max_acciones=15):
    num_acciones = min(max_acciones, len(columnas))
    plt.figure(figsize=(15, 8))
    for accion in columnas[:num_acciones]:
        plt.plot(data.index, data[accion], label=accion)
    plt.title('Rendimiento de Acciones')
    plt.xlabel('Fecha')
    plt.ylabel('Rendimiento')
    plt.legend()
    plt.show()

# Graficar series temporales
graficar_series_temporales(stock_precio, portafolio)

# Función para calcular el portafolio sub-óptimo de Markowitz
def portafolio_suboptimo_inversiones(portafolio):
    return np.ones(len(portafolio)) / len(portafolio)

portafolio = portafolio[:len(stock_retorno.columns)]

# Ejecutar la función para el portafolio sub-óptimo
sub_optimo_portafolio = portafolio_suboptimo_inversiones(portafolio)

# Tratamiento en np para operaciones posteriores
sub_optimo_portafolio = np.expand_dims(sub_optimo_portafolio, axis=0)

# Calcular rendimiento, varianza y riesgo del portafolio sub-óptimo
rendimiento_suboptimo = np.dot(sub_optimo_portafolio, esperados_stock_retorno)
varianza_suboptima = np.dot(sub_optimo_portafolio, np.dot(stock_retorno_esperados_cov_matriz, sub_optimo_portafolio.T))
riesgo_suboptimo = math.sqrt(varianza_suboptima)

# Mostrar datos del portafolio sub-óptimo
print('Portafolio sub-óptimo de Markowitz:')
print(f'Rendimiento esperado: {rendimiento_suboptimo[0] * 100:.2f}%')
print(f'Varianza: {varianza_suboptima[0, 0] * 100:.2f}%')
print(f'Riesgo del portafolio: {riesgo_suboptimo * 100:.2f}%')

# Modelo Markowitz para el portafolio óptimo
# Calcular los rendimientos y riesgos esperados usando pypfopt
mu = expected_returns.mean_historical_return(stock_precio)
S = risk_models.sample_cov(stock_precio)
ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
ratios = ef.min_volatility()
ratio_limpio = pd.Series(ratios)
ratio_limpio.index = portafolio # type: ignore
portafolio_optimo = np.expand_dims(ratio_limpio, axis=0)

# Limitar a dos cifras significativas en las ponderaciones
ratio_limpio = ratio_limpio.round(4)
portafolio_optimo = np.expand_dims(ratio_limpio, axis=0)

# Calcular rendimiento, varianza y riesgo del portafolio óptimo
retorno_esperado_optimo = np.dot(portafolio_optimo, mu)[0]
portafolio_optimo_var = np.dot(portafolio_optimo, np.dot(S, portafolio_optimo.transpose()))[0, 0]
portafolio_optimo_riesgo = math.sqrt(portafolio_optimo_var)

# Mostrar datos del portafolio óptimo
print('\nPortafolio óptimo de Markowitz:')
print(f'Rendimiento esperado: {retorno_esperado_optimo * 100:.2f}%')
print(f'Varianza: {portafolio_optimo_var * 100:.2f}%')
print(f'Riesgo del portafolio: {portafolio_optimo_riesgo * 100:.2f}%')

# Mostrar la asignación de ponderaciones en cada activo
print('\nDe la inversión total, el modelo sugiere invertir las siguientes proporciones en cada activo:')
display(ratio_limpio * 100)

# Supongamos que ya tienes el DataFrame 'ratio_limpio' con las ponderaciones base
# Define las ponderaciones base obtenidas del portafolio óptimo de Markowitz
ponderaciones_base = ratio_limpio

# Define las ponderaciones para cada perfil de riesgo
ponderaciones_agresivo = ponderaciones_base * 1.2  # Ajuste para un perfil agresivo
ponderaciones_moderado = ponderaciones_base * 1.0  # Sin cambios para un perfil moderado
ponderaciones_conservador = ponderaciones_base * 0.8  # Ajuste para un perfil conservador

# Dependiendo del perfil de riesgo del cliente, selecciona las ponderaciones adecuadas
perfil_riesgo_cliente = 'agresivo'  # Puedes cambiar esto según el perfil del cliente

if perfil_riesgo_cliente == 'agresivo':
    ponderaciones_cliente = ponderaciones_agresivo
elif perfil_riesgo_cliente == 'moderado':
    ponderaciones_cliente = ponderaciones_moderado
elif perfil_riesgo_cliente == 'conservador':
    ponderaciones_cliente = ponderaciones_conservador
else:
    raise ValueError("Perfil de riesgo no válido")

# Muestra las ponderaciones para cada perfil de riesgo
print("Ponderaciones ajustadas para un perfil AGRESIVO:")
print(ponderaciones_agresivo)

print("\nPonderaciones ajustadas para un perfil MODERADO:")
print(ponderaciones_moderado)

print("\nPonderaciones ajustadas para un perfil CONSERVADOR:")
print(ponderaciones_conservador)

# Simulación de rendimientos y riesgos
num_simulaciones = 100
simulaciones = np.random.multivariate_normal(mu, S, num_simulaciones)
simulacion_rendimientos = np.dot(simulaciones, portafolio_optimo.transpose())
simulacion_riesgos = np.std(simulaciones, axis=1)

# Crear DataFrame con simulaciones
df = pd.DataFrame({
    'Rendimientos': simulacion_rendimientos.ravel(),
    'Riesgos': simulacion_riesgos.ravel(),
})

# Calcular y mostrar métricas adicionales (Sharpe Ratio, R2, MAE)
sharpe_ratio_optimo = (retorno_esperado_optimo - 0.09) / portafolio_optimo_riesgo
r2_optimo = np.corrcoef(simulacion_rendimientos.flatten(), df['Rendimientos'].values.flatten())[0, 1] ** 2 # type: ignore
mae_optimo = median_absolute_error(simulacion_rendimientos.flatten(), df['Rendimientos'].values.flatten()) # type: ignore

print('\nMétricas adicionales del portafolio óptimo:')
print(f'Sharpe Ratio: {sharpe_ratio_optimo:.4f}')
print(f'R2: {r2_optimo:.4f}')
print(f'MAE: {mae_optimo:.4f}')

# Serializar resultados obtenidos
joblib.dump((mu, S, ratio_limpio), 'resultados_markowitz.joblib')

# Cargar resultados serializados
mu, S, ratio_limpio = joblib.load('resultados_markowitz.joblib')

# Crear variable objetivo (0: Bajo, 1: Alto) usando la mediana
df['Clasificacion'] = np.where(df['Rendimientos'] > df['Rendimientos'].median(), 1, 0)

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X = df[['Rendimientos', 'Riesgos']]
y = df['Clasificacion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Grilla de hiperparámetros para modelos de ML
parametros_rf = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}

parametros_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2']
}

parametros_svr = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# Modelos de regresión con grillas de hiperparámetros
from sklearn.linear_model import LogisticRegression

modelos = {
    'Regresión Logística': (LogisticRegression(), parametros_lr),
    'RandomForestRegressor': (RandomForestRegressor(), parametros_rf),
    'SVR': (SVR(), parametros_svr)
}

# Definir las métricas para modelos de regresión
metricas_regresion = {
    'MAE': median_absolute_error,
    'MSE': mean_squared_error,
    'RMSE': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    'R2 Score': r2_score
}

# Evaluar y mostrar métricas para cada modelo
from sklearn.model_selection import GridSearchCV


for model_name, (model, params) in modelos.items():
    try:
        # Buscar los mejores hiperparámetros usando GridSearchCV
        mejor_modelo = GridSearchCV(model, params, cv=5)
        mejor_modelo.fit(X_train, y_train)

        # Entrenar modelo con mejores hiperparámetros
        model.set_params(**mejor_modelo.best_params_)
        model.fit(X_train, y_train)

        # Mostrar información sobre las métricas
        print(f"\nMétricas para {model_name}:")

        if isinstance(model, RandomForestRegressor):
            # Métricas específicas para RandomForestRegressor
            feature_importances = model.feature_importances_
            print(f"Feature Importances: {feature_importances}")

        for metric_name, metric_function in metricas_regresion.items():
            # Calcular métrica
            y_test_pred = model.predict(X_test)
            metric_value = metric_function(y_test, y_test_pred)

            # Mostrar resultado
            print(f"{metric_name}: {metric_value:.4f}")

            if metric_name == 'R2 Score' and isinstance(model, LogisticRegression):
                # Calcular R2 Ajustado para la regresión logística
                n = len(X_test)
                p = X_test.shape[1]
                r2_ajustado = 1 - (1 - metric_value) * (n - 1) / (n - p - 1)
                print(f"R2 Ajustado: {r2_ajustado:.4f}")
    except Exception as e:
        print(f"Error en el modelo {model_name}: {e}")

# Obtener datos de las acciones del portafolio desde el 02-01-2023 hasta el 30-11-2023
fecha_inicio_prediccion = '2023-03-01'
fecha_fin_prediccion = '2023-11-30'

stock_precio_prediccion = pd.DataFrame()

for x in portafolio:
    stock_prediccion = yf.Ticker(x)
    precio_cierre_prediccion = stock_prediccion.history(start=fecha_inicio_prediccion, end=fecha_fin_prediccion)['Close']

    # Insertar datos en el DataFrame
    stock_precio_prediccion[x] = precio_cierre_prediccion

# Eliminar valores nulos
stock_precio_prediccion = stock_precio_prediccion.dropna()

# Calcular los retornos de las acciones del portafolio para el período de predicción
stock_retorno_prediccion = stock_precio_prediccion.pct_change().dropna()

# Calcular el rendimiento esperado y el riesgo individual para el período de predicción
esperados_stock_retorno_prediccion = stock_retorno_prediccion.mean()
stock_riesgo_individual_prediccion = stock_retorno_prediccion.std()

# Obtener la matriz de covarianza para el período de predicción
stock_retorno_cov_matriz_prediccion = stock_retorno_prediccion.cov()

# Mostrar datos para el período de predicción
print("\nRendimientos esperados para el período de predicción:")
print(esperados_stock_retorno_prediccion)

print("Riesgo individual para el período de predicción:")
print(stock_riesgo_individual_prediccion)

print("Matriz de covarianza de rendimientos de activos para el período de predicción:")
display(stock_retorno_cov_matriz_prediccion)

# Graficar series temporales para el período de predicción
graficar_series_temporales(stock_precio_prediccion, portafolio)

# Simulación de rendimientos y riesgos para el período de predicción
num_simulaciones_prediccion = 100
simulaciones_prediccion = np.random.multivariate_normal(
    esperados_stock_retorno_prediccion,
    stock_retorno_cov_matriz_prediccion,
    num_simulaciones_prediccion
)
simulacion_rendimientos_prediccion = np.dot(simulaciones_prediccion, portafolio_optimo.transpose())
simulacion_riesgos_prediccion = np.std(simulaciones_prediccion, axis=1)

# Crear DataFrame con simulaciones para el período de predicción
df_prediccion = pd.DataFrame({
    'Rendimientos': simulacion_rendimientos_prediccion.ravel(),
    'Riesgos': simulacion_riesgos_prediccion.ravel(),
})

# Calcular y mostrar métricas adicionales para el período de predicción (Sharpe Ratio, R2, MAE)
sharpe_ratio_prediccion = (retorno_esperado_optimo - 0.09) / portafolio_optimo_riesgo
r2_prediccion = np.corrcoef(
    simulacion_rendimientos_prediccion.flatten(),
    df_prediccion['Rendimientos'].values.flatten()
)[0, 1] ** 2
mae_prediccion = median_absolute_error(
    simulacion_rendimientos_prediccion.flatten(),
    df_prediccion['Rendimientos'].values.flatten()
)

print('\nMétricas adicionales para el período de predicción:')
print(f'Sharpe Ratio: {sharpe_ratio_prediccion:.4f}')
print(f'R2: {r2_prediccion:.4f}')
print(f'MAE: {mae_prediccion:.4f}')

# Visualizar la clasificación de rendimientos para el período de predicción
df_prediccion['Clasificacion'] = np.where(
    df_prediccion['Rendimientos'] > df_prediccion['Rendimientos'].median(), 1, 0
)

plt.figure(figsize=(10, 6))
plt.scatter(
    df_prediccion['Riesgos'],
    df_prediccion['Rendimientos'],
    c=df_prediccion['Clasificacion'],
    cmap='coolwarm',
    alpha=0.7
)
plt.title('Clasificación de Rendimientos para el Período de Predicción')
plt.xlabel('Riesgo')
plt.ylabel('Rendimiento')
plt.show()

# Comparar predicciones con datos reales para el período de predicción
df_prediccion['Clasificacion_real'] = np.where(
    stock_retorno_prediccion[portafolio[0]][:len(df_prediccion)] > stock_retorno_prediccion[portafolio[0]].median(), 1, 0
)

# Calcular porcentaje de certeza para el período de predicción
porcentaje_certeza_prediccion = (df_prediccion['Clasificacion'] == df_prediccion['Clasificacion_real']).mean() * 100

# Calcular grado de acierto para el período de predicción
aciertos_prediccion = np.sum(
    (df_prediccion['Clasificacion'] == df_prediccion['Clasificacion_real']).values & (df_prediccion['Clasificacion_real'] == 1).values
)
total_positivos_reales_prediccion = np.sum(df_prediccion['Clasificacion_real'] == 1)
grado_acierto_prediccion = aciertos_prediccion / total_positivos_reales_prediccion if total_positivos_reales_prediccion != 0 else 0

# Mostrar resultados para el período de predicción
print(f"\nPorcentaje de certeza entre las predicciones y los datos reales para el período de predicción: {porcentaje_certeza_prediccion:.2f}%")
print(f"Grado de acierto para el período de predicción: {grado_acierto_prediccion:.2f}")

# Visualizar la clasificación de rendimientos para el período de predicción (datos reales vs. predicciones)
plt.figure(figsize=(10, 6))
plt.scatter(
    df_prediccion['Riesgos'],
    df_prediccion['Rendimientos'],
    c=df_prediccion['Clasificacion_real'],
    cmap='coolwarm',
    alpha=0.7,
    label='Datos Reales'
)
plt.scatter(
    df_prediccion['Riesgos'],
    df_prediccion['Rendimientos'],
    c=df_prediccion['Clasificacion'],
    marker='x',
    cmap='coolwarm',
    alpha=0.7,
    label='Predicciones'
)
plt.title('Comparación de Clasificación de Rendimientos para el Período de Predicción')
plt.xlabel('Riesgo')
plt.ylabel('Rendimiento')
plt.legend()
plt.show()