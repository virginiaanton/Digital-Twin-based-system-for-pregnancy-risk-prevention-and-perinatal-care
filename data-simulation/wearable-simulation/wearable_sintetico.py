#!/usr/bin/env python
# coding: utf-8

# **Dado la falta de datos completos y consistentes que capturen los signos vitales, se ha optado por generar datos sintéticos 
# que se adapten más al diseño de nuestro sistema. 
# Para crear el primer dataset, con datos de salud maternos tomados en tiempos discretos, se ha usado la librería sdv
# de python.**

# A partir del dataset Maternal Health Risk, queremos obtener datos sintéticos, que sigan los ismos patrones que el dataset encontrado, para poder disponer de más datos con el que entrenar a nuestro modelo. 
# 
# Esta librería además, se usará para simular diferentes historias de usuarios y simular su 
# desarrollo y seguimiento durante los meses de gestación. 

# In[3]:


from ctgan import CTGAN


# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

import scipy.stats as stats
import seaborn as sns
from sklearn.cluster import KMeans

from datetime import date
from datetime import datetime
from datetime import timedelta


# Configurar Seaborn para gráficos más bonitos
sns.set(style="whitegrid")


# In[6]:


df= pd.read_csv('Maternal Health Risk Data Set.csv')


# # ANÁLISIS DEL DATASET 

# <ol>
#     <li>Limpieza del dataset: tipos de datos, identificación de valores nulos, valores atípicos. </li>
#     <li>Verificar rangos de valores.</li>
# </ol>

# In[11]:


print(f"Tipos de datos del dataset :\n\n {df.dtypes}")
print(f"\nBuscamos si hay valores nulos en las columnas:\n", df.isnull().sum())


# Observamos que la mayoría de los datos son enteros, salvo el azúcar en sangre y la temperatura. 
# También vemos que no hay valores nulos que tratar. 
# 

# ## Análisis estadístico

# In[15]:


print("\nVista del dataframe:\n")
df.head()


# In[17]:


data_dup = df.duplicated().any()
data_dup


# In[19]:


print("\nEstadísticas del dataframe\n")
print(df.describe())


# In[21]:


def verifica_rango(dataframe):
    
    rangos_limites = {
        'SystolicBP': (70,200),
        'DiastolicBP':(40,120),
        'BodyTemp': (82,106.5),
        'BS':(2.22,33.3),
        'HeartRate':(30,200),
    }
    
    results= {}
    for col, (min_val, max_val) in rangos_limites.items():
        if col in dataframe.columns:
            
            validos = dataframe[(dataframe[col] >= min_val) & (dataframe[col] <= max_val)]
            fuera_rango = len(dataframe) - len(validos)
            results[col] = {
                'val_fuera_rango':fuera_rango,
                'total_filas': len(dataframe), 
                'valores_validos': len(validos)
            }

    for col, info in results.items():
        print(f"- Valores fuera del rango de {col}:{info['val_fuera_rango']}\n")
        print(f"- Total valores de {col}:{info['total_filas']}\n")
        print(f"- Valores validos de {col}: {info['valores_validos']}\n")
    


# In[23]:


fig, axes = plt.subplots(1, 6, figsize=(20, 6))  # Crear un subplot con 6 gráficos

# Lista de variables y sus títulos
variables = [
    ('Age', 'Distribución variable Age\n'),
    ('BodyTemp', 'Distribución variable BodyTemp\n'),
    ('BS', 'Distribución variable Blood Sugar\n'),
    ('HeartRate', 'Distribución variable HeartRate\n'),
    ('DiastolicBP', 'Distribución variable DiastolicBP\n'),
    ('SystolicBP', 'Distribución variable SystolicBP\n')
]

# Graficar cada variable en su respectivo eje
for ax, (var, title) in zip(axes, variables):
    sns.kdeplot(data=df, x=var, fill=True, ax=ax)  # Graficar en el eje correspondiente
    ax.set_title(title)
    ax.set_xlabel(var)
    ax.set_ylabel('Densidad')

# Ajustar diseño
plt.tight_layout()
plt.show()


# Vamos a observar la distribución que siguen nuestros datos y comprobar si hay valores extremos, para ello usaremos diagramas *boxplots*. 

# In[28]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report

fig, axes = plt.subplots(1, 5, figsize=(20, 6))

#Diagrama columna 'Age'
sns.boxplot(y=df['Age'], ax=axes[0], orient = "v", legend = False, linewidth = 1,
           native_scale=True, width = .25, palette = "pastel", color = '#a1c9f4')
axes[0].set_title('Distribución variable Age')

#Diagrama BodyTemp
sns.boxplot(y=df['BodyTemp'], ax=axes[1], orient = "v", legend = False, linewidth = 1,
            native_scale=True, width = .25, palette = "pastel", color = '#8de5a1')
axes[1].set_title('Distribución variable BodyTemp')

# Boxplot para BS (Blood Sugar)
sns.boxplot(y=df['BS'], ax=axes[2],  orient = "v", legend = False, linewidth = 1,native_scale=True, 
            width = .25, palette = "pastel", color = '#ff9f9b')
axes[2].set_title('Distribución variable Blood Sugar')

# Boxplot para HeartRate
sns.boxplot(y=df['HeartRate'], ax=axes[3], orient = "v", legend = False, linewidth = 1,
            native_scale=True, width = .25, palette = "pastel", color = '#d0bbff')
axes[3].set_title('Distribución variable HeartRate')

# Boxplot para SystolicBP
sns.boxplot(y=df['DiastolicBP'], ax= axes[4], orient = "v", legend = False, linewidth = 1,
           native_scale=True, width = .25, palette = "pastel",color = '#fffea3')
axes[4].set_title('Distribución variable DiastolicBP')

plt.tight_layout()
plt.show()


# In[30]:


# 5. **Verificar duplicados**
# Verificar y eliminar filas duplicadas si es necesario
print(f"Número de filas duplicadas: {df.duplicated().sum()}")
df = df.drop_duplicates()  # Eliminar filas duplicadas

# 6. **Verificar rangos válidos para las variables**:
# Comprobar si los valores están dentro de los rangos fisiológicos razonables
verifica_rango(df)

# 7. **Comprobar las correlaciones**:
# Verificar las correlaciones generales antes de dividir en patológicos y sanos
corr_matrix = df[['Age','SystolicBP', 'DiastolicBP', 'BodyTemp', 'BS', 'HeartRate']].corr()
print("Matriz de correlación general:\n", corr_matrix)



# Queríamos ver si hay outliers fuera de los rangos fisiológicos incompatibles con la vida, pero hemos podido comprobar que ninguno se sale de los rangos posibles. Así no tenemos que limpiarlo, porque los outliers nos interesan para tener más casos de usuarios y entrenar a nuestros modelos. 

# In[33]:


# 8. **Visualizar correlaciones** con un heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("\nHeatmap de Correlaciones de los signos vitales\n")
plt.show()


# Las variables que más correlacionadas están son:
# <ol>
#     <li> La presión sistólica y diastólica: Una aumenta en respuesto al aumento de la otra. </li>
#     <li>La edad y la presión sistólica, la edad y la presión diastólica: cuánto más mayores nos hacemos, aumenta la presión, cobra sentido
#     ya que los vasos sanguíneos son cada vez menos flexible y el corazón bombea con menos fuerza</li>
#     <li>La edad y el azúcar en sangre: según vamos envejeciendo, disminuye la producción de insulina y aumenta la resistencia a esta, 
#     es por ello que procesamos peor el azúcar que ingerimos. </li>
# 

# Por último vamos a comprobar la distribución de las etiquetas, para saber cómo son las pacientes
# en los que nos basamos, si la mayoría enfermas, o sanas, o si hablamos de una muestra balanceada. 

# In[37]:


print(f"Frecuencia niveles de riesgo en la muestra de pacientes:  {df['RiskLevel'].value_counts()}")

sns.histplot(data=df, x='RiskLevel', discrete=True, kde=True, color='skyblue', alpha=0.7)
plt.title('Distribución de RiskLevel')
plt.xlabel('Niveles de Riesgo')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45)
plt.show()


# Observamos que la mayoría (más de la mitad) de las pacientes son sanas, y presentan bajo riesgo en su embarazo. 
# 
# 

# ---

# # CREACIÓN DATOS SINTÉTICOS SALUD MATERNA

# # Creación de datos sintéticos wearable

# La sección de generación del dataset crea datos sintéticos de signos vitales continuos para cada usuario y marca de tiempo. Se basa en los valores iniciales del archivo CSV y simula la variabilidad fisiológica utilizando distribuciones de probabilidad y patrones temporales.
# 
# Simulación de la presión arterial: Tanto la presión arterial sistólica (SystolicBP) como la diastólica (DiastolicBP) se simulan utilizando una distribución normal truncada (scipy.stats.truncnorm). Esto permite generar valores realistas que se agrupan alrededor de la media de cada usuario (obtenida del CSV), pero que se mantienen dentro de límites fisiológicos plausibles (70-190 mmHg para sistólica, 40-120 mmHg para diastólica). La desviación estándar de la distribución se ajusta para controlar la variabilidad de los datos sintéticos. Finalmente, los valores se redondean al entero más cercano.
# 
# Simulación de la glucosa en sangre: La glucosa en sangre (BS) se modela considerando una variación diurna. Se añade una onda sinusoidal a los valores base para simular las fluctuaciones diarias típicas relacionadas con la ingesta de alimentos y el ayuno. Además, se introduce ruido aleatorio mediante una distribución normal para representar la variabilidad individual y las fluctuaciones a lo largo del tiempo.
# 
# Simulación de la temperatura corporal: La temperatura corporal (BodyTemp) se simula utilizando una distribución normal centrada en la temperatura base de cada usuario (del CSV) con una pequeña desviación estándar, reflejando las fluctuaciones normales de la temperatura corporal.
# 
# Simulación de la frecuencia cardíaca: Similar a la presión arterial, la frecuencia cardíaca (HeartRate) se genera con una distribución normal truncada (truncnorm) para mantener los valores dentro de un rango fisiológico (40-120 lpm) y centrados en la frecuencia cardíaca base de cada usuario. Los valores generados también se redondean al entero más cercano.
# 
# Vectorización y formato del DataFrame: Para una mayor eficiencia, todas las operaciones de generación de datos se realizan utilizando funciones vectorizadas de NumPy. Esto permite generar los datos para todos los usuarios y marcas de tiempo simultáneamente, en lugar de iterar a través de bucles. Finalmente, los arrays resultantes se aplanan utilizando .flatten() para convertirlos en arrays 1D y poder insertarlos correctamente en las columnas correspondientes del DataFrame wearable_df.

# A continuación se presenta un desglose de las variables que suelen considerarse distribuidas normalmente o de forma gaussiana, y cómo generar datos simulados más realistas:
# <ol>
#     <li>Presión arterial sistólica (PSistólica) y presión arterial diastólica (PDiastólica): Suelen tener una distribución aproximadamente normal en poblaciones sanas, pero pueden estar sesgadas en individuos con hipertensión u otras afecciones.
# </li>
# <li>Azúcar en sangre (BS): El azúcar en sangre no suele tener una distribución normal. Tiende a aumentar después de las comidas y a disminuir durante los periodos de ayuno. Modelarlo como normal podría ser engañoso, especialmente para pacientes con diabetes.</li>
# 
# <li>Temperatura corporal (BT/BodyTemp): La temperatura corporal se distribuye de forma aproximadamente normal en torno a un punto de referencia (por ejemplo, 37 °C o 98,6 °F), pero con una desviación estándar relativamente pequeña.</li></ol>
# 
# 

# In[45]:


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import scipy.stats as stats

df = pd.read_csv("synthetic_vital_signs.csv")
n_users = len(df)

# Create timestamps for the 3-month period (Jan 1st to Mar 31st)
timestamps = pd.date_range("2024-01-01", "2024-04-01", freq="h")
n_timestamps = len(timestamps)

# Create the DataFrame directly
wearable_df = pd.DataFrame({
    "User ID": np.repeat(df.index.values, n_timestamps),
    "Timestamp": np.tile(timestamps, n_users),
    "Age": np.repeat(df.Age.values, n_timestamps),
})
wearable_df = wearable_df.set_index("Timestamp")

# Generate simulated data using vectorized operations
wearable_df["SystolicBP"] = np.round(stats.truncnorm.rvs(
    a=(70 - df.SystolicBP.values[:, None]) / 5,
    b=(190 - df.SystolicBP.values[:, None]) / 5,
    loc=df.SystolicBP.values[:, None],
    scale=5,
    size=(n_users, n_timestamps)
)).flatten()

wearable_df["DiastolicBP"] = np.round(stats.truncnorm.rvs(
    a=(40 - df.DiastolicBP.values[:, None]) / 3,
    b=(120 - df.DiastolicBP.values[:, None]) / 3,
    loc=df.DiastolicBP.values[:, None],
    scale=3,
    size=(n_users, n_timestamps)
)).flatten()

base_bs = np.repeat(df.BS.values[:, None], n_timestamps, axis=1) + np.random.normal(0, 0.5, size=(n_users, n_timestamps))
diurnal_variation = 0.1 * np.sin(2 * np.pi * np.arange(n_timestamps) / 24)
bs_values = base_bs + diurnal_variation
wearable_df["BS"] = bs_values.flatten()

body_temp_values = df.BodyTemp.values[:, None] + np.random.normal(0, 0.2, (n_users, n_timestamps))
wearable_df["BodyTemp"] = body_temp_values.flatten()

wearable_df["HeartRate"] = np.round(stats.truncnorm.rvs(
    a=(40 - df.HeartRate.values[:, None]) / 10,
    b=(120 - df.HeartRate.values[:, None]) / 10,
    loc=df.HeartRate.values[:, None],
    scale=10,
    size=(n_users, n_timestamps)
)).flatten()

wearable_df["HeartRate"]= wearable_df["HeartRate"].astype(int)

print(wearable_df)


# Este dataset recoge datos calculados para 3 meses de signos vitales, pero sin etiquetar, para que sea el propio algoritmo el que lo haga. Como tenemos demasiados datos, vamos a limitar a 12h donde recogemos datos de los pacientes (500), en vez de por hora (24 datos a lo largo del día), ya que no es tan significativo de hora a hora la variación de los datos, pues no se tratan de enfermedades tan críticas como se pueden tratar en los cuidados paliativos. 

# In[48]:


import pandas as pd
import numpy as np

df = wearable_df
hours_per_user = 12

# 1. Verificar las correlaciones originales en el dataset
correlation_matrix = df[['Age', 'SystolicBP', 'DiastolicBP', 'BodyTemp', 'BS', 'HeartRate']].corr()
print(correlation_matrix)

# 2. Aplicar las correlaciones deseadas según la matriz de correlación del primer dataset
desired_correlations = {
    'DiastolicBP_Age': 0.35,
    'DiastolicBP_SystolicBP': 0.79,
    'DiastolicBP_BS': 0.30,
    'DiastolicBP_BodyTemp': -0.20,
    'SystolicBP_Age': 0.38,
    'SystolicBP_BS': 0.35,
    'SystolicBP_BodyTemp': -0.21,
    'BS_Age': 0.38, 
    'BodyTemp_Age': -0.19, 
    'BodyTemp_SystolicBP': -0.21, 
    'BS_SystolicBP': 0.35, 
    'BodyTemp_DiastolicBP': -0.20, 
    'BS_DiastolicBP': 0.30, 
    'BS_BodyTemp': -0.043, 
    'HeartRate_BS': 0.14
    
}

# 3. Generar correlaciones deseadas usando la Cholesky Decomposition
# Crear la matriz de covarianza basada en las correlaciones proporcionadas
cov_matrix = np.array([
    [1, desired_correlations['SystolicBP_Age'], desired_correlations['DiastolicBP_Age'], desired_correlations['BodyTemp_Age'], desired_correlations['BS_Age'], 0],
    [desired_correlations['SystolicBP_Age'], 1, desired_correlations['DiastolicBP_SystolicBP'], desired_correlations['BodyTemp_SystolicBP'], desired_correlations['BS_SystolicBP'], 0],
    [desired_correlations['DiastolicBP_Age'], desired_correlations['DiastolicBP_SystolicBP'], 1, desired_correlations['BodyTemp_DiastolicBP'], desired_correlations['BS_DiastolicBP'], 0],
    [desired_correlations['BodyTemp_Age'], desired_correlations['BodyTemp_SystolicBP'], desired_correlations['BodyTemp_DiastolicBP'], 1, desired_correlations['BS_BodyTemp'], 0],
    [desired_correlations['BS_Age'], desired_correlations['BS_SystolicBP'], desired_correlations['BS_DiastolicBP'], desired_correlations['BS_BodyTemp'], 1, desired_correlations['HeartRate_BS']],
    [0, 0, 0, 0, desired_correlations['BS_Age'], 1]  # Correlación entre HeartRate y BS se ha omitido
])

# Usar la factorización Cholesky para generar variables con las correlaciones deseadas

cholesky_matrix = np.linalg.cholesky(cov_matrix)
synthetic_data = np.dot(np.random.randn(len(df), 6), cholesky_matrix.T)

# 5. Crear un DataFrame con los datos sintéticos generados
synthetic_df = pd.DataFrame(synthetic_data, columns=['Age', 'SystolicBP', 'DiastolicBP', 'BodyTemp', 'BS', 'HeartRate'])

# Add diurnal variations (e.g., for BodyTemp, BS)
time_series = np.tile(np.arange(hours_per_user), n_users)

diurnal_variation_bs = 0.1 * np.sin(2 * np.pi * time_series / 24)

diurnal_variation_body_temp = 1.8 * np.sin((3/2)* np.pi * time_series / 24) #latemperatura varía de 97.7ºF-99.5ºF

diurnal_variation_heart_rate = np.round( np.sin(2 * np.pi * time_series / 24),0)

# **Variación diurna para Presión Arterial (Diastolic BP)**
# Forma tipo campana de Gauss: máximo a las 12 p.m. (las 6 primeras horas)
# Usamos una forma de campana de Gauss centrada en las 12 p.m. (hora 6)
diurnal_variation_bp = np.exp(-0.5 * ((time_series - 6) / 2) ** 2)
diurnal_variation_systolic_bp = np.exp(-0.5 * ((time_series - 6) / 2) ** 2)

 #**Fluctuaciones patológicas independientes** (esto no se ve afectado por la amplitud base)
# Variabilidad patológica (simula fluctuaciones como en hipertensión)
pathological_fluctuations_bp = np.random.normal(0, 5, size=(n_users * hours_per_user))  # Picos aleatorios de hipertensión

# Variabilidad patológica se agrega sin truncar la onda base
diurnal_variation_bp += pathological_fluctuations_bp  # Fluctuaciones independientes
diurnal_variation_systolic_bp += pathological_fluctuations_bp 

#Redondeamos HR, SBP, DBP:
diurnal_variation_heart_rate = np.round(diurnal_variation_heart_rate).astype(int)
diurnal_variation_bp = np.round(diurnal_variation_bp).astype(int)
diurnal_variation_systolic_bp = np.round(diurnal_variation_systolic_bp).astype(int)

#Creamos dataset final con serie temporal  y correlaciones
data = {
    'Age': np.repeat(df.Age.values[:n_users], hours_per_user),
    'SystolicBP': np.repeat(df.SystolicBP.values[:n_users], hours_per_user) + diurnal_variation_systolic_bp,  # Agregar fluctuación diaria
    'DiastolicBP': np.repeat(df.DiastolicBP.values[:n_users], hours_per_user) + diurnal_variation_bp,  # Agregar fluctuación diaria
    'BodyTemp': np.repeat(df.BodyTemp.values[:n_users], hours_per_user) + diurnal_variation_body_temp,  # Add daily fluctuation
    'BS': np.repeat(df.BS.values[:n_users], hours_per_user) + diurnal_variation_bs,  # Add daily fluctuation
    'HeartRate': np.repeat(df.HeartRate.values[:n_users], hours_per_user) + diurnal_variation_heart_rate,  # Add daily fluctuation
}

timestamps = pd.date_range("2024-01-01", periods=hours_per_user, freq="h")
data['Timestamp'] = np.tile(timestamps, n_users)
data['User ID'] = np.repeat(range(n_users), hours_per_user)



# 7. Mostrar el DataFrame con las correlaciones aplicadas
#import ace_tools as tools; tools.display_dataframe_to_user(name="Synthetic Wearable Dataset with Time Series and Correlations", dataframe=synthetic_df)

# Verificar las primeras filas del DataFrame generado
pd.DataFrame(data)
final_df= pd.DataFrame(data)
final_df["HeartRate"] = final_df["HeartRate"].astype(int) #que sean datos tipo int y no float
    # Add timestamps and user IDs


# In[50]:


final_df.head()


# Ahora vamos a aseguranos que sigue las variables se encuentra entre los datos fisiologicos válidos, tal y como hemos hecho en el base

# In[53]:


# 5. **Verificar duplicados**
# Verificar y eliminar filas duplicadas si es necesario
print(f"Número de filas duplicadas: {df.duplicated().sum()}")
final_df = final_df.drop_duplicates()  # Eliminar filas duplicadas
print(len(final_df))
# 6. **Verificar rangos válidos para las variables**:
# Comprobar si los valores están dentro de los rangos fisiológicos razonables
verifica_rango(final_df)




# Como el dataset base, no hay ningún dato que no sea posible o compatible con la vida.

# ### Comprobación y validación de la calidad dataset sintético

# Hay que demostrar que nuestro dataset refleja datos del mundo real.

# <ol>
#     <li>Comparación de propiedades estadísticas</li>
#     <li>Test estadísticos</li>
#     <li>Comprobar consistencia temporal</li>
# 

# ##### Propiedades estadísticas
# 

# Distribución de cada variable sintética debe ser similar a dataset base: Kernel Density Functions e histogramas

# In[61]:


import seaborn as sns
import matplotlib.pyplot as plt

# Compara distribuciones con KDE e histogramas
plt.figure(figsize=(12, 6))

# Gráfico dataset base vs sintético para variable DiastolicBP

sns.histplot(df['DiastolicBP'], kde=True, color='blue', label='Base', stat='density')
sns.histplot(final_df['DiastolicBP'], kde=True, color='orange', label='Sintético', stat='density')
plt.legend()
plt.title('Comparación distribución- DiastolicBP')
plt.show()


# In[63]:


# Compara distribuciones con KDE e histogramas
plt.figure(figsize=(12, 6))

# Gráfico dataset base vs sintético para variable BodyTemp

sns.histplot(df['BodyTemp'], kde=True, color='blue', label='Base', stat='density')
sns.histplot(final_df['BodyTemp'], kde=True, color='orange', label='Sintético', stat='density')
plt.legend()
plt.title('Comparación distribución- BodyTemp')
plt.show()


# In[65]:


# Compara distribuciones con KDE e histogramas
plt.figure(figsize=(12, 6))

# Gráfico dataset base vs sintético para variable HeartRate

sns.histplot(df['HeartRate'], kde=True, color='blue', label='Base', stat='density')
sns.histplot(final_df['HeartRate'], kde=True, color='orange', label='Sintético', stat='density')
plt.legend()
plt.title('Comparación distribución-HeartRate')
plt.show()


# In[66]:


# Compara distribuciones con KDE e histogramas
plt.figure(figsize=(12, 6))

# Gráfico dataset base vs sintético para variable BloodSugar

sns.histplot(df['BS'], kde=True, color='blue', label='Base', stat='density')
sns.histplot(final_df['BS'], kde=True, color='orange', label='Sintético', stat='density')
plt.legend()
plt.title('Comparación distribución-Blood Sugar')
plt.show()


# In[68]:


# Compara distribuciones con KDE e histogramas
plt.figure(figsize=(12, 6))

# Gráfico dataset base vs sintético para variable SystolicBP

sns.histplot(df['SystolicBP'], kde=True, color='blue', label='Base', stat='density')
sns.histplot(final_df['SystolicBP'], kde=True, color='orange', label='Sintético', stat='density')
plt.legend()
plt.title('Comparación distribución-SystolicBP')
plt.show()


# In[69]:


# Compara distribuciones con KDE e histogramas
plt.figure(figsize=(12, 6))

# Gráfico dataset base vs sintético para variable Age

sns.histplot(df['Age'], kde=True, color='blue', label='Base', stat='density')
sns.histplot(final_df['Age'], kde=True, color='orange', label='Sintético', stat='density')
plt.legend()
plt.title('Comparación distribución-Age')
plt.show()


# In[75]:


import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp


#Renombramos columnas dataframe sintetico

common_columns = [col for col in df.columns if col in final_df.columns]

Results = {}
for column in common_columns:
    # Comprobar si la columna es numérica
    if pd.api.types.is_numeric_dtype(df[column]):
        
        # Kolmogorov-Smirnov Test
        ks_stat, p_value = ks_2samp(df[column], final_df[column])
        Results[column] = {'KS Statistic': ks_stat, 'P-value': p_value}

# Convertir los resultados a un DataFrame para visualizarlos mejor
ksResults_finaldf = pd.DataFrame(Results).T
print(ksResults_finaldf)


# ### Exportación del dataset sintético

# In[78]:


final_df.to_csv("finalSimulationWearableData.csv", index=True) 


# In[80]:


final_df['User ID'].value_counts()


# ## USO DE TIME-DGAN PARA CREAR NUESTRO DATASET
# 

# In[83]:


df_dgan= pd.read_csv('time-series-dgan-wearable-data.csv')


# In[85]:


df_dgan.head()


# ##### Arreglos del dataset

# Vamos a pasar a numeros enteros los valores de la columna Age, DiastolicBloodPressure, SystolicBloodPressure y HeartRate. 
# Además, vamos a eliminar la columna example_id ya que ha sido el algoritmo quien lo ha puesto automaticamente para parecerse al datasetbase, y esto
# no nos hace falta. 
# 
# Finalmente lo que vamos a hacer es transformar la columna de BloodSugar de mg/dl a mmol/l como viene en el dataset de Maternal Health Risk. 

# In[89]:


#1. conversion a enteros
columns_to_convert = ['Age', 'DiastolicBloodPressure', 'SystolicBloodPressure', 'HeartRate']
for col in columns_to_convert:
    df_dgan[col] = df_dgan[col].astype(int) 

#2. Elimino example_id
if 'example_id' in df_dgan.columns:
    df_dgan.drop(columns=['example_id'], inplace=True)


#3. Conversion mg/dl a mmol/L
df_dgan['BloodSugar'] = df_dgan['BloodSugar'] / 18

df_dgan.head()


# In[91]:


df_dgan.describe()


# In[93]:


# 5. **Verificar duplicados**
# Verificar y eliminar filas duplicadas si es necesario
print(f"Número de filas duplicadas: {df_dgan.duplicated().sum()}")
#df_dgan= df_dgan.drop_duplicates()  # Eliminar filas duplicadas

# 6. **Verificar rangos válidos para las variables**:
# Comprobar si los valores están dentro de los rangos fisiológicos razonables
verifica_rango(df_dgan)


# ##### Búsqueda y manejo de outliers

# In[96]:


fig, axes = plt.subplots(1, 5, figsize=(20, 6))

#Diagrama columna 'Age'
sns.boxplot(y=df_dgan['Age'], ax=axes[0], orient = "v", legend = False, linewidth = 1,
           native_scale=True, width = .25, palette = "pastel", color = '#a1c9f4')
axes[0].set_title('Distribución variable Age')

#Diagrama BodyTemp
sns.boxplot(y=df_dgan['BodyTemp'], ax=axes[1], orient = "v", legend = False, linewidth = 1,
            native_scale=True, width = .25, palette = "pastel", color = '#8de5a1')
axes[1].set_title('Distribución variable BodyTemp')

# Boxplot para BS (Blood Sugar)
sns.boxplot(y=df_dgan['BloodSugar'], ax=axes[2],  orient = "v", legend = False, linewidth = 1,native_scale=True, 
            width = .25, palette = "pastel", color = '#ff9f9b')
axes[2].set_title('Distribución variable Blood Sugar')

# Boxplot para HeartRate
sns.boxplot(y=df_dgan['HeartRate'], ax=axes[3], orient = "v", legend = False, linewidth = 1,
            native_scale=True, width = .25, palette = "pastel", color = '#d0bbff')
axes[3].set_title('Distribución variable HeartRate')

# Boxplot para SystolicBP
sns.boxplot(y=df_dgan['DiastolicBloodPressure'], ax= axes[4], orient = "v", legend = False, linewidth = 1,
           native_scale=True, width = .25, palette = "pastel",color = '#fffea3')
axes[4].set_title('Distribución variable DiastolicBP')

plt.tight_layout()
plt.show()


# Eliminamos los outliers que se salen de los valores compatibles con la vida

# In[99]:


# 1. Definir los rangos válidos para cada variable
valid_ranges = {
    'BodyTemp': (82, 106.5),  # Rango fisiológico de temperatura corporal (Fahrenheit)
    'BloodSugar': (2.2, 33.3),  # Azúcar en sangre (mmol/L)
    'SystolicBloodPressure': (70, 180),  # Presión sistólica (mmHg)
    'DiastolicBloodPressure': (40, 120)  # Presión diastólica (mmHg)
}

# 2. Aplicar filtros para eliminar valores fuera de los rangos
for column, (min_val, max_val) in valid_ranges.items():
    if column in df_dgan.columns:
        df_dgan = df_dgan[(df_dgan[column] >= min_val) & (df_dgan[column] <= max_val)]

# 3. Mostrar las primeras filas del DataFrame actualizado
print("\nDataset limpio:\n")
df_dgan.head()


# Vamos a comprobar si la correlación entre variables coincide con el dataset base: 

# In[102]:


# 7. **Comprobar las correlaciones**:

corrMatrix = df_dgan[['Age','SystolicBloodPressure', 'DiastolicBloodPressure', 'BodyTemp', 'BloodSugar', 'HeartRate']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corrMatrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("\nHeatmap de Correlaciones de los signos vitales\n")
plt.show()


# In[ ]:





# #### Comprobación y validación del dataset sintético

# ##### Comprobación de propiedades estadísticas

# In[107]:


import seaborn as sns
import matplotlib.pyplot as plt

# Compara distribuciones con KDE e histogramas
plt.figure(figsize=(12, 6))

# Gráfico dataset base vs sintético para variable DiastolicBP

sns.histplot(df['DiastolicBP'], kde=True, color='blue', label='Base', stat='density')
sns.histplot(df_dgan['DiastolicBloodPressure'], kde=True, color='orange', label='Sintético', stat='density')
plt.legend()
plt.title('Comparación distribución- DiastolicBP')
plt.show()


# In[108]:


# Compara distribuciones con KDE e histogramas
plt.figure(figsize=(12, 6))

# Gráfico dataset base vs sintético para variable BodyTemp

sns.histplot(df['BodyTemp'], kde=True, color='blue', label='Base', stat='density')
sns.histplot(df_dgan['BodyTemp'], kde=True, color='orange', label='Sintético', stat='density')
plt.legend()
plt.title('Comparación distribución- BodyTemp')
plt.show()


# In[109]:


# Compara distribuciones con KDE e histogramas
plt.figure(figsize=(12, 6))

# Gráfico dataset base vs sintético para variable HeartRate

sns.histplot(df['HeartRate'], kde=True, color='blue', label='Base', stat='density')
sns.histplot(df_dgan['HeartRate'], kde=True, color='orange', label='Sintético', stat='density')
plt.legend()
plt.title('Comparación distribución-HeartRate')
plt.show()


# In[110]:


# Compara distribuciones con KDE e histogramas
plt.figure(figsize=(12, 6))

# Gráfico dataset base vs sintético para variable BloodSugar

sns.histplot(df['BS'], kde=True, color='blue', label='Base', stat='density')
sns.histplot(df_dgan['BloodSugar'], kde=True, color='orange', label='Sintético', stat='density')
plt.legend()
plt.title('Comparación distribución-Blood Sugar')
plt.show()


# In[112]:


# Compara distribuciones con KDE e histogramas
plt.figure(figsize=(12, 6))

# Gráfico dataset base vs sintético para variable SystolicBP

sns.histplot(df['SystolicBP'], kde=True, color='blue', label='Base', stat='density')
sns.histplot(df_dgan['SystolicBloodPressure'], kde=True, color='orange', label='Sintético', stat='density')
plt.legend()
plt.title('Comparación distribución-SystolicBP')
plt.show()


# In[113]:


# Compara distribuciones con KDE e histogramas
plt.figure(figsize=(12, 6))

# Gráfico dataset base vs sintético para variable Age

sns.histplot(df['Age'], kde=True, color='blue', label='Base', stat='density')
sns.histplot(df_dgan['Age'], kde=True, color='orange', label='Sintético', stat='density')
plt.legend()
plt.title('Comparación distribución-Age')
plt.show()


# Como podemos comprobar, las distribuciones son muy distintas de forma visual. 
# Aún asi, hay varias estrategias para mejorar la calidad de los datos que podemos seguir, 
# ya que este es in porblema común a la hora de generar datos sintéticos. 

# Antes de ajustar, vamos a entender cómo y dónde se diferencian nuestras distribuciones: 
# Vamos a emplear el test Kolmogorov-Smirnov, mide las distancias entre distribuciones:
# <ul>
#     <li>ks_stat: representa la distancia máxima entre las distribuciones. Valores ceracnos a cero indican que las distribuciones son similares.</li>
#     <li>p_value: indica si las dos distribuciones son estadísticamente diferentes. Un valor pequeño del p-value, significa que rechazamos la hipótesis nula (distribuciones significativamente diferentes).</li>
# </ul>

# In[121]:


import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp


#Renombramos columnas dataframe sintetico

df_dgan.rename(columns={'SystolicBloodPressure': 'SystolicBP', 'DiastolicBloodPressure': 'DiastolicBP', 'BloodSugar':'BS'}, inplace=True)
common_columns = [col for col in df.columns if col in df_dgan.columns]

ks_results = {}
for column in common_columns:
    # Comprobar si la columna es numérica
    if pd.api.types.is_numeric_dtype(df[column]):
        
        # Kolmogorov-Smirnov Test
        ks_stat, p_value = ks_2samp(df[column], df_dgan[column])
        ks_results[column] = {'KS Statistic': ks_stat, 'P-value': p_value}

# Convertir los resultados a un DataFrame para visualizarlos mejor
ks_results_df = pd.DataFrame(ks_results).T
print(ks_results_df)


# Podemos observar que hay altas KS (gran diferencia) y p-values muy altos. 

# ---

# ## DATAFRAME UN SÓLO DATO POR USUARIO

# In[126]:


data= pd.read_csv('navigator-batch.csv')


# In[128]:


data.head()


# In[130]:


data.info()


# Vamos a modificar los datos: 
# <ul>
#     <li>Renombramos las columnas como en el base. </li>
#     <li>Convertimos las unidades de medida de Blood Sugar de mg/dl a mmol/l</li>
# </ul>
# 
# 

# In[133]:


#1. Renombramos las columnas
data.rename(columns={'SystolicBloodPressure': 'SystolicBP', 'DiastolicBloodPressure': 'DiastolicBP', 'BloodSugar':'BS'}, inplace=True)

#2. Conversión de unidades : mg/dl a mmol/l
data['BS'] = (data['BS']/18)
data['BS'] = data['BS'].round(2)
data.head()
    


# In[135]:


data.describe()


# In[137]:


# 5. **Verificar duplicados**
# Verificar y eliminar filas duplicadas si es necesario
print(f"Número de filas duplicadas: {data.duplicated().sum()}")

# 6. **Verificar rangos válidos para las variables**:
# Comprobar si los valores están dentro de los rangos fisiológicos razonables
verifica_rango(data)


# In[139]:


fig, axes = plt.subplots(1, 5, figsize=(20, 6))

#Diagrama columna 'Age'
sns.boxplot(y=data['Age'], ax=axes[0], orient = "v", legend = False, linewidth = 1,
           native_scale=True, width = .25, palette = "pastel", color = '#a1c9f4')
axes[0].set_title('Distribución variable Age')

#Diagrama BodyTemp
sns.boxplot(y=data['BodyTemp'], ax=axes[1], orient = "v", legend = False, linewidth = 1,
            native_scale=True, width = .25, palette = "pastel", color = '#8de5a1')
axes[1].set_title('Distribución variable BodyTemp')

# Boxplot para BS (Blood Sugar)
sns.boxplot(y=data['BS'], ax=axes[2],  orient = "v", legend = False, linewidth = 1,native_scale=True, 
            width = .25, palette = "pastel", color = '#ff9f9b')
axes[2].set_title('Distribución variable Blood Sugar')

# Boxplot para HeartRate
sns.boxplot(y=data['HeartRate'], ax=axes[3], orient = "v", legend = False, linewidth = 1,
            native_scale=True, width = .25, palette = "pastel", color = '#d0bbff')
axes[3].set_title('Distribución variable HeartRate')

# Boxplot para SystolicBP
sns.boxplot(y=data['DiastolicBP'], ax= axes[4], orient = "v", legend = False, linewidth = 1,
           native_scale=True, width = .25, palette = "pastel",color = '#fffea3')
axes[4].set_title('Distribución variable DiastolicBP')

plt.tight_layout()
plt.show()


# A diferencia del resto de opciones, este presente dataset tiene muchos outliers, por
# lo que vamos a eliminarlos. 

# In[142]:


def elimina_outliers(dataframe):
    rangos_fisiologicos = {
        'BodyTemp': (82, 106.5),  
        'BS': (2.2, 33.3),  
        'SystolicBP': (70, 180),  
        'DiastolicBP': (40, 120)  
    }
    for columna, (min_val, max_val) in rangos_fisiologicos.items():
        if columna in df.columns:
            dataframe = dataframe[(dataframe[columna] >= min_val) & (dataframe[columna] <= max_val)]
    return dataframe


# In[144]:


data = elimina_outliers(data)
verifica_rango(data)


# Comprobamos que correlaciones hay entre las variables y si coinciden con el dataset base:

# In[147]:


correlation_matrix = data[['Age','SystolicBP', 'DiastolicBP', 'BodyTemp', 'BS', 'HeartRate']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("\nHeatmap de Correlaciones de los signos vitales\n")
plt.show()


# In[149]:


import seaborn as sns
import matplotlib.pyplot as plt

# Compara distribuciones con KDE e histogramas
plt.figure(figsize=(12, 6))

# Gráfico dataset base vs sintético para variable DiastolicBP

sns.histplot(df['DiastolicBP'], kde=True, color='blue', label='Base', stat='density')
sns.histplot(data['DiastolicBP'], kde=True, color='orange', label='Sintético', stat='density')
plt.legend()
plt.title('Comparación distribución- DiastolicBP')
plt.show()


# In[150]:


# Compara distribuciones con KDE e histogramas
plt.figure(figsize=(12, 6))

# Gráfico dataset base vs sintético para variable SystolicBP

sns.histplot(df['SystolicBP'], kde=True, color='blue', label='Base', stat='density')
sns.histplot(data['SystolicBP'], kde=True, color='orange', label='Sintético', stat='density')
plt.legend()
plt.title('Comparación distribución-SystolicBP')
plt.show()


# In[151]:


# Compara distribuciones con KDE e histogramas
plt.figure(figsize=(12, 6))

# Gráfico dataset base vs sintético para variable BodyTemp

sns.histplot(df['BodyTemp'], kde=True, color='blue', label='Base', stat='density')
sns.histplot(data['BodyTemp'], kde=True, color='orange', label='Sintético', stat='density')
plt.legend()
plt.title('Comparación distribución- BodyTemp')
plt.show()


# In[152]:


# Compara distribuciones con KDE e histogramas
plt.figure(figsize=(12, 6))

# Gráfico dataset base vs sintético para variable HeartRate

sns.histplot(df['HeartRate'], kde=True, color='blue', label='Base', stat='density')
sns.histplot(data['HeartRate'], kde=True, color='orange', label='Sintético', stat='density')
plt.legend()
plt.title('Comparación distribución-HeartRate')
plt.show()


# In[153]:


# Compara distribuciones con KDE e histogramas
plt.figure(figsize=(12, 6))

# Gráfico dataset base vs sintético para variable BloodSugar

sns.histplot(df['BS'], kde=True, color='blue', label='Base', stat='density')
sns.histplot(data['BS'], kde=True, color='orange', label='Sintético', stat='density')
plt.legend()
plt.title('Comparación distribución-Blood Sugar')
plt.show()


# In[154]:


# Compara distribuciones con KDE e histogramas
plt.figure(figsize=(12, 6))

# Gráfico dataset base vs sintético para variable Age

sns.histplot(df['Age'], kde=True, color='blue', label='Base', stat='density')
sns.histplot(data['Age'], kde=True, color='orange', label='Sintético', stat='density')
plt.legend()
plt.title('Comparación distribución-Age')
plt.show()


# A diferencia de las otras dos opciones, cubren practicamente todo el rango, aunque los valores salgan con menos frecuencia. 
# Pero no nos preocupa mucho porque en el siguiente apartado nuestro objetivo es aumentar el número de datos registrados por usuario al día, 
# y luego después al mes. Con lo que cabe esperar que la densidad en las gráficas aumentará, pero al menos no solo alberga una zona. 

# In[157]:


import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp


#Renombramos columnas dataframe sintetico

common_columns = [col for col in df.columns if col in data.columns]

ksResults = {}
for column in common_columns:
    # Comprobar si la columna es numérica
    if pd.api.types.is_numeric_dtype(df[column]):
        
        # Kolmogorov-Smirnov Test
        ks_stat, p_value = ks_2samp(df[column], data[column])
        ksResults[column] = {'KS Statistic': ks_stat, 'P-value': p_value}

# Convertir los resultados a un DataFrame para visualizarlos mejor
ksResults_data = pd.DataFrame(ks_results).T
print(ksResults_data)


# ## ANÁLISIS DEL PROBLEMA ACTUAL: DESCARTAR O NO EL DATASET GENERADO

# <ol>
#     <li>Test Kolmogorov-Smirnov: los valores del p-value son muy bajos, lo que bindica que las distribuciones entre el dataset base y el sintético son muy diferentes.</li>
#     <li>Histograma / KDE : observando los gráficos vemos que las distribuciones son significativamente distintas, por lo que el dataset sintético no representa con precisión algunas de las características del dataset base. Sin embargo, es importante notar que el dataset sintético cubre el rango fisiológico.</li>
#     <li>Extensión del dataset sintético: el objetivo es simular varias medidas diarias para cada usuario, y esperamos que la densidad de los valores aumente y las distribuciones se acercquen más a las reales, aunque no es seguro esperar que las distribuciones finales sean idénticas.</li>
# </ol>
# 
# Otras consideraciones:
# El propóstio del dataset es para que nuestro modelo prediga los riesgos de una mujer embarazada de padecer tener un embarazo de riesgo o de presentar alguna complicación durante la gestación. Lo importante es que los datos sintéticos sean fisológicamente razonables y capture relaciones clave entre variables, no necesariamente necesitammos que sean idénticas las correlaciones, pero sí consistentes con la fisiología humana. 
# 
# Es por ello, que al cubrir más valores del rango, introduce menor sesgo, y tras la extensión se espera aumentar la densidad, acercándose más a los datos reales. 
# 

# ## INTENTO 3 WEARABLE, MENOS USUARIOS, MÁS DÍAS Y 3 CAPTURADAS AL DÍA

# In[237]:


wearabledf= pd.read_csv('user-wearable.csv')


# In[239]:


wearabledf.info()


# In[241]:


wearabledf.head()


# In[243]:


wearabledf.describe()


# In[245]:


#1. Renombramos las columnas
wearabledf.rename(columns = {'age':'Age'}, inplace=True)

#2. Mirar nº de filas duplicadas
print(f"\nNúmero de filas duplicadas: {data.duplicated().sum()}\n")

# 3. **Verificar rangos válidos para las variables**:
# Comprobar si los valores están dentro de los rangos fisiológicos razonables y eliminarlos
verifica_rango(wearabledf)
elimina_outliers(wearabledf)


# In[247]:


fig, axes = plt.subplots(1, 5, figsize=(20, 6))

#Diagrama columna 'Age'
sns.boxplot(y=wearabledf['Age'], ax=axes[0], orient = "v", legend = False, linewidth = 1,
           native_scale=True, width = .25, palette = "pastel", color = '#a1c9f4')
axes[0].set_title('Distribución variable Age')

#Diagrama BodyTemp
sns.boxplot(y=wearabledf['BodyTemp'], ax=axes[1], orient = "v", legend = False, linewidth = 1,
            native_scale=True, width = .25, palette = "pastel", color = '#8de5a1')
axes[1].set_title('Distribución variable BodyTemp')

# Boxplot para BS (Blood Sugar)
sns.boxplot(y=wearabledf['BS'], ax=axes[2],  orient = "v", legend = False, linewidth = 1,native_scale=True, 
            width = .25, palette = "pastel", color = '#ff9f9b')
axes[2].set_title('Distribución variable Blood Sugar')

# Boxplot para HeartRate
sns.boxplot(y=wearabledf['HeartRate'], ax=axes[3], orient = "v", legend = False, linewidth = 1,
            native_scale=True, width = .25, palette = "pastel", color = '#d0bbff')
axes[3].set_title('Distribución variable HeartRate')

# Boxplot para SystolicBP
sns.boxplot(y=wearabledf['DiastolicBP'], ax= axes[4], orient = "v", legend = False, linewidth = 1,
           native_scale=True, width = .25, palette = "pastel",color = '#fffea3')
axes[4].set_title('Distribución variable DiastolicBP')

plt.tight_layout()
plt.show()


# In[249]:


corrMatWearable = wearabledf[['Age','SystolicBP', 'DiastolicBP', 'BodyTemp', 'BS', 'HeartRate']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corrMatWearable, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("\nHeatmap de Correlaciones de los signos vitales\n")
plt.show()


# In[251]:


import seaborn as sns
import matplotlib.pyplot as plt

# Compara distribuciones con KDE e histogramas
plt.figure(figsize=(12, 6))

# Gráfico dataset base vs sintético para variable DiastolicBP

sns.histplot(df['DiastolicBP'], kde=True, color='blue', label='Base', stat='density')
sns.histplot(wearabledf['DiastolicBP'], kde=True, color='orange', label='Sintético', stat='density')
plt.legend()
plt.title('Comparación distribución- DiastolicBP')
plt.show()


# In[252]:


# Compara distribuciones con KDE e histogramas
plt.figure(figsize=(12, 6))

# Gráfico dataset base vs sintético para variable SystolicBP

sns.histplot(df['SystolicBP'], kde=True, color='blue', label='Base', stat='density')
sns.histplot(wearabledf['SystolicBP'], kde=True, color='orange', label='Sintético', stat='density')
plt.legend()
plt.title('Comparación distribución-SystolicBP')
plt.show()


# In[253]:


# Compara distribuciones con KDE e histogramas
plt.figure(figsize=(12, 6))

# Gráfico dataset base vs sintético para variable BodyTemp

sns.histplot(df['BodyTemp'], kde=True, color='blue', label='Base', stat='density')
sns.histplot(wearabledf['BodyTemp'], kde=True, color='orange', label='Sintético', stat='density')
plt.legend()
plt.title('Comparación distribución- BodyTemp')
plt.show()


# In[254]:


# Compara distribuciones con KDE e histogramas
plt.figure(figsize=(12, 6))

# Gráfico dataset base vs sintético para variable HeartRate

sns.histplot(df['HeartRate'], kde=True, color='blue', label='Base', stat='density')
sns.histplot(wearabledf['HeartRate'], kde=True, color='orange', label='Sintético', stat='density')
plt.legend()
plt.title('Comparación distribución-HeartRate')
plt.show()


# In[255]:


# Compara distribuciones con KDE e histogramas
plt.figure(figsize=(12, 6))

# Gráfico dataset base vs sintético para variable BloodSugar

sns.histplot(df['BS'], kde=True, color='blue', label='Base', stat='density')
sns.histplot(wearabledf['BS'], kde=True, color='orange', label='Sintético', stat='density')
plt.legend()
plt.title('Comparación distribución-Blood Sugar')
plt.show()


# In[256]:


# Compara distribuciones con KDE e histogramas
plt.figure(figsize=(12, 6))

# Gráfico dataset base vs sintético para variable Age

sns.histplot(df['Age'], kde=True, color='blue', label='Base', stat='density')
sns.histplot(wearabledf['Age'], kde=True, color='orange', label='Sintético', stat='density')
plt.legend()
plt.title('Comparación distribución-Age')
plt.show()


# In[187]:


import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp


#Renombramos columnas dataframe sintetico

common_columns = [col for col in df.columns if col in wearabledf.columns]

ksResults = {}
for column in common_columns:
    # Comprobar si la columna es numérica
    if pd.api.types.is_numeric_dtype(df[column]):
        
        # Kolmogorov-Smirnov Test
        ks_stat, p_value = ks_2samp(df[column], wearabledf[column])
        ksResults[column] = {'KS Statistic': ks_stat, 'P-value': p_value}

# Convertir los resultados a un DataFrame para visualizarlos mejor
ksResultsWearable = pd.DataFrame(ksResults).T
print(ksResultsWearable)


# ### INTENTO 3

# In[263]:


import pandas as pd
user_wearable= pd.read_csv('user_wearable2.csv')


# In[265]:


user_wearable.info()


# In[267]:


user_wearable.head()


# In[269]:


#1. Renombramos las columnas
user_wearable.rename(columns = {'age':'Age'}, inplace=True)

#2. Mirar nº de filas duplicadas
print(f"\nNúmero de filas duplicadas: {user_wearable.duplicated().sum()}\n")

# 3. **Verificar rangos válidos para las variables**:
# Comprobar si los valores están dentro de los rangos fisiológicos razonables y eliminarlos
verifica_rango(user_wearable)
elimina_outliers(user_wearable)


# In[271]:


fig, axes = plt.subplots(1, 5, figsize=(20, 6))

#Diagrama columna 'Age'
sns.boxplot(y=user_wearable['Age'], ax=axes[0], orient = "v", legend = False, linewidth = 1,
           native_scale=True, width = .25, palette = "pastel", color = '#a1c9f4')
axes[0].set_title('Distribución variable Age')

#Diagrama BodyTemp
sns.boxplot(y=user_wearable['BodyTemp'], ax=axes[1], orient = "v", legend = False, linewidth = 1,
            native_scale=True, width = .25, palette = "pastel", color = '#8de5a1')
axes[1].set_title('Distribución variable BodyTemp')

# Boxplot para BS (Blood Sugar)
sns.boxplot(y=user_wearable['BS'], ax=axes[2],  orient = "v", legend = False, linewidth = 1,native_scale=True, 
            width = .25, palette = "pastel", color = '#ff9f9b')
axes[2].set_title('Distribución variable Blood Sugar')

# Boxplot para HeartRate
sns.boxplot(y=user_wearable['HeartRate'], ax=axes[3], orient = "v", legend = False, linewidth = 1,
            native_scale=True, width = .25, palette = "pastel", color = '#d0bbff')
axes[3].set_title('Distribución variable HeartRate')

# Boxplot para SystolicBP
sns.boxplot(y=user_wearable['DiastolicBP'], ax= axes[4], orient = "v", legend = False, linewidth = 1,
           native_scale=True, width = .25, palette = "pastel",color = '#fffea3')
axes[4].set_title('Distribución variable DiastolicBP')

plt.tight_layout()
plt.show()


# In[272]:


corrMatWearable = user_wearable[['Age','SystolicBP', 'DiastolicBP', 'BodyTemp', 'BS', 'HeartRate']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corrMatWearable, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("\nHeatmap de Correlaciones de los signos vitales\n")
plt.show()


# In[273]:


import seaborn as sns
import matplotlib.pyplot as plt

# Compara distribuciones con KDE e histogramas
plt.figure(figsize=(12, 6))

# Gráfico dataset base vs sintético para variable DiastolicBP

sns.histplot(df['DiastolicBP'], kde=True, color='blue', label='Base', stat='density')
sns.histplot(user_wearable['DiastolicBP'], kde=True, color='orange', label='Sintético', stat='density')
plt.legend()
plt.title('Comparación distribución- DiastolicBP')
plt.show()


# In[276]:


# Compara distribuciones con KDE e histogramas
plt.figure(figsize=(12, 6))

# Gráfico dataset base vs sintético para variable BodyTemp

sns.histplot(df['BodyTemp'], kde=True, color='blue', label='Base', stat='density')
sns.histplot(df_dgan['BodyTemp'], kde=True, color='orange', label='Sintético', stat='density')
plt.legend()
plt.title('Comparación distribución- BodyTemp')
plt.show()


# In[277]:


# Compara distribuciones con KDE e histogramas
plt.figure(figsize=(12, 6))

# Gráfico dataset base vs sintético para variable HeartRate

sns.histplot(df['HeartRate'], kde=True, color='blue', label='Base', stat='density')
sns.histplot(user_wearable['HeartRate'], kde=True, color='orange', label='Sintético', stat='density')
plt.legend()
plt.title('Comparación distribución-HeartRate')
plt.show()


# In[278]:


# Compara distribuciones con KDE e histogramas
plt.figure(figsize=(12, 6))

# Gráfico dataset base vs sintético para variable BloodSugar

sns.histplot(df['BS'], kde=True, color='blue', label='Base', stat='density')
sns.histplot(user_wearable['BS'], kde=True, color='orange', label='Sintético', stat='density')
plt.legend()
plt.title('Comparación distribución-Blood Sugar')
plt.show()


# In[279]:


# Compara distribuciones con KDE e histogramas
plt.figure(figsize=(12, 6))

# Gráfico dataset base vs sintético para variable SystolicBP

sns.histplot(df['SystolicBP'], kde=True, color='blue', label='Base', stat='density')
sns.histplot(user_wearable['SystolicBP'], kde=True, color='orange', label='Sintético', stat='density')
plt.legend()
plt.title('Comparación distribución-SystolicBP')
plt.show()


# In[280]:


# Compara distribuciones con KDE e histogramas
plt.figure(figsize=(12, 6))

# Gráfico dataset base vs sintético para variable Age

sns.histplot(df['Age'], kde=True, color='blue', label='Base', stat='density')
sns.histplot(user_wearable['Age'], kde=True, color='orange', label='Sintético', stat='density')
plt.legend()
plt.title('Comparación distribución-Age')
plt.show()


# Podemos observar que de todas las simulaciones, la más parecida a los patrones que sigan las variables
# en Maternal Health Risk dataset es esta última. Al ser una simulación, no nos preocupa que no sea idéntica al dataste base, pero
# podemos observar que las distribuciones y la variación de los valores es parecida, al menos sabemos que siguen la misma naturaleza. 

# # SIMULACIÓN PARA PRIMER MES DE GESTACIÓN

# Usaremos los datos recogidos de un sólo día y los replicaremos para cubrir un mes. 
# Para ello, hay que replicar los datos diarios por 30 días. Manteniendo las fluctuaciones diarias para cada uno de esos días. 
# 
# Nuestro siguiente objetivo será modificar gradualmente los signos vitales para reflejar su variación durante los 9 meses de embarazo. Como primera idea se nos ha ocurrido incluir un factor progresivo, que simule la variación.

# In[284]:


from sdv.metadata import Metadata
import json
with open('metadata-1-month-augment.json') as archive:
    metadata_dict = json.load(archive)


# In[285]:


metadata = Metadata.load_from_dict(metadata_dict)
metadata.validate()


# In[286]:


import pandas as pd

# 'timestamp' debe estar en formato datetime
user_wearable['timestamp'] = pd.to_datetime(user_wearable['timestamp'], errors='coerce')

# Cambiamos los valores al formato del df base
user_wearable['timestamp'] = user_wearable['timestamp'].dt.strftime('%d/%m/%Y %H:%M:%S')
print(user_wearable['timestamp'].head())



# In[287]:


from sdv.sequential import PARSynthesizer

# Step 1: Create the synthesizer
synthesizer = PARSynthesizer(metadata)

# Step 2: Train the synthesizer
synthesizer.fit(user_wearable)

# Step 3: Generate synthetic data
synthetic_data = synthesizer.sample(num_sequences=100)


# In[298]:


synthetic_data.to_json('synthetic_data')


# In[300]:


# Ajustar los valores de las variables a los rangos fisiológicos:

synthetic_data['Age'] = synthetic_data['Age'].clip(lower=15, upper=60)
synthetic_data['SystolicBP'] = synthetic_data['SystolicBP'].clip(lower=70, upper=180)
synthetic_data['DiastolicBP'] = synthetic_data['DiastolicBP'].clip(lower=49, upper=100)
synthetic_data['BS'] = synthetic_data['BS'].clip(lower=6.0, upper=19.0)
synthetic_data['BodyTemp'] = synthetic_data['BodyTemp'].clip(lower=98.0, upper=103.0)
synthetic_data['HeartRate'] = synthetic_data['HeartRate'].clip(lower=7, upper=120)

synthetic_data.head()


# In[302]:


synthetic_data.describe()


# Vamos a comprobar si todos los datos recolectados para los usuarios se encuentran en el mismo periodo, 
# Vamos a ver qué usuario corresponde la primera medida y a qué usuario corresponde la última medida guardada. 

# In[305]:


# i. Obtener los valores mínimo y máximo de la columna timestamp
min_timestamp = synthetic_data["timestamp"].min()
max_timestamp = synthetic_data["timestamp"].max()

# ii. Identificar el usuario asociado al mínimo y máximo timestamp
min_user = synthetic_data.loc[synthetic_data["timestamp"] == min_timestamp, "user_id"].values
max_user = synthetic_data.loc[synthetic_data["timestamp"] == max_timestamp, "user_id"].values

# iii. Observamos los resultados
print(f"Min Timestamp: {min_timestamp} corresponde al usuario con ID: {min_user}")
print(f"Max Timestamp: {max_timestamp} corresponde al usuario con ID: {max_user}")


# In[307]:


# Calcular el min y max timestamp para cada usuario
user_timestamp_range = synthetic_data.groupby("user_id")["timestamp"].agg(["min", "max"]).reset_index()

# Mostrar el rango de timestamps para cada usuario
print(user_timestamp_range)


# Como se puede observar, los datos no parten de la misma fecha. Nos da igual que a lo largo del día unos datos
# queden registrados a una hora y otros a otra, pero sí nos interesa tener el mismo número de medidas para cada usuario y
# que empiecen y acaben en el mismo periodo de fechas, para simplificar el algoritmo.
# 
# Vamos a crear usuarios que se encuentren en el mismo rango y vamos a suponer que 
# todas las pacientes han tenido la primera consulta en las mismas fechas, y haremos el seguimiento a la vez 
# para esta muestra. Por ello, vamos a filtrar todos los usuarios que tengamos por ejemplo desde el 1 de Mayo de 2024 hasta
# el 31 de Julio de 2024, y el seguimiento será alrededor de 9-10 meses. 

# In[310]:


#Definimos periodo
def periodOfSimulation(start, end, df):
    filtered_data = df.query("timestamp >= @start and timestamp <= @end ")
    filtered_users = filtered_data["user_id"].unique()
    num_users = filtered_data["user_id"].nunique()
    return filtered_users, num_users
    


# In[312]:


inicio = "01/06/2024 00:00:00"
fin = "31/07/2024 23:59:59"
filtered_users, num_users = periodOfSimulation(inicio, fin, synthetic_data)
print(f"{num_users} usuarios que registraron datos desde {inicio}-{fin}:\n",filtered_users)


# El dataset filtrado quedaría:

# In[315]:


# Filtra por el periodo de interés
synthetic_data= synthetic_data[(synthetic_data['timestamp'] >= inicio) & (synthetic_data['timestamp'] < fin)]


# In[317]:


min_timestamp = synthetic_data["timestamp"].min()
max_timestamp = synthetic_data["timestamp"].max()
print(min_timestamp, max_timestamp)


# In[319]:


synthetic_data.head()


# In[321]:


#comprobamos si  los usuarios que hemos encontrado antes están el dataset filtrado
# Obtener los usuarios únicos en el dataset filtrado
users_in_df = set(synthetic_data['user_id'].unique())

# Convertir la lista de usuarios de interés en un conjunto
interested_users_set= set(filtered_users)

# Encontrar los usuarios que están en la lista de interés pero no en el dataset
missing_users = interested_users_set - users_in_df

# Imprimir el resultado
if missing_users:
    print(f"Usuarios que no están en el dataset: {missing_users}")
else:
    print("Todos los usuarios de la lista de interés están presentes en el dataset.")


# Ahora vamos a comprobar que todos los usuarios tienen el mismo número de medidas al día. Dado que es 
# el interfaz que, en caso de no recibir 3 medidas al día de forma voluntaria, se lo pide, de forma que 
# no perdamos la continuidad e información durante los 9 meses. 

# In[324]:


def verifica_medidas_porDia(df, user_col='user_id', timestamp_col='timestamp'):
    """
    Verifica si todos los usuarios tienen el mismo número de medidas al día. 
    Args:
        - df: dataframe 
        - user_col: columna con ids usuarios. 
        - timestamp_col : columna con timestamps. 
    Returns: 
        None: imprime resultados. 
    """
     #creamos fecha columna sin las horas
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce', dayfirst=True)

    df['date'] = df[timestamp_col].dt.date
    medidas_porDia = df.groupby([user_col, 'date']).size().reset_index(name='num_medidas')

    #verificar si todos los usuarios tienen mismo num de medidas al día
    medidas_unicas = medidas_porDia['num_medidas'].unique()
    
    if len(medidas_unicas)==1:
        print(f"Todos los usuarios tienen el mismo numero de medidas al día: {medidas_unicas[0]}")
    else:
            print(f"Los usuarios tienen diferentes números de medidas al día:{ medidas_porDia.groupby('num_medidas').size()}")
            plt.hist(medidas_porDia['num_medidas'], bins=range(1, medidas_porDia['num_medidas'].max() + 2), edgecolor='black')
            plt.title("Distribución del número de medidas al día")
            plt.xlabel("Número de medidas al día")
            plt.ylabel("Frecuencia")
            plt.xticks(range(1, medidas_porDia['num_medidas'].max() + 1))
            plt.show()


# In[326]:


synCopy = synthetic_data.copy()


# In[328]:


verifica_medidas_porDia(synCopy)


# Para que tengan el mismo número de medidas, rellenamos los datos. Debemos tener cuidado de respetar las distribucioes individuales
# y las correlaciones entre columnas. Para ello, podemos usar SDV (Synthetic Data Vault), tiene Modelos Temporales Condicionados (TimeSeriesTransformer), en concreto, vamos a emplear el modelo PAR (Probabilistic AutoRegressive) de dicha librería, para generar datos respetando las tendencias y correlaciones entre signos vitales. 
# 

# In[331]:


synthetic_data.head()


# In[333]:


import pandas as pd

def summarize_missing_values(df):
    """
    Devuelve un resumen del número y porcentaje de valores NaN en cada columna de un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame a analizar.

    Returns:
        pd.DataFrame: Resumen con columnas 'missing_count' y 'missing_percentage',
                      ordenado por la cantidad de valores faltantes en orden descendente.
    """
    missing_summary = pd.DataFrame({
        'missing_count': df.isnull().sum(),
        'missing_percentage': df.isnull().mean() * 100
    }).sort_values(by='missing_count', ascending=False)
    
    return missing_summary


# summary = summarize_missing_values(df)
# print(summary)


# In[335]:


summary_synthetic_data = summarize_missing_values(synthetic_data)
print(summary_synthetic_data)


# Podemos observar que ha surgido un problema añadido, y es que no sólo faltan medidas por día en el usuario, es que no aparecen estas filas con
# valores NaN para poder rellenar: vamos a reindexar el dataframe para que existan huecos que podamos rellenar. 

# In[389]:


synthetic_data['day'] =  pd.to_datetime(synthetic_data['timestamp'], errors='coerce', dayfirst=True)
synthetic_data.head()


# In[395]:


import pandas as pd
import numpy as np

def add_measures_with_nans(df):
    """
    Añade un mínimo de 3 medidas al día con valores NaN salvo en 'Age', 'day', 'timestamp' y 'user_id',
    para que puedan ser interpolados después.
    """
    # Asegúrate de que la columna 'timestamp' sea de tipo datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')

    # Crea una columna 'day' que contiene solo la fecha (sin hora)
    df['day'] = df['timestamp'].dt.date

    # Crear un nuevo DataFrame para almacenar los resultados
    expanded_df = pd.DataFrame()

    # Iterar sobre cada usuario
    for user_id, user_data in df.groupby('user_id'):
        # Iterar sobre cada día del usuario
        for day, day_data in user_data.groupby('day'):
            # Añadir las filas originales del día
            expanded_df = pd.concat([expanded_df, day_data])

            # Si ya hay al menos 3 medidas, no es necesario añadir
            if len(day_data) >= 3:
                continue

            # Añadir medidas adicionales para alcanzar un mínimo de 3 por día
            additional_measures = 3 - len(day_data)
            for i in range(additional_measures):
                # Generar timestamps espaciados (8 horas) dentro del mismo día
                new_timestamp = day_data['timestamp'].iloc[0] + pd.Timedelta(hours=8 * (i + 1))
                if new_timestamp.date() != day:  # Asegurarse de que sigue siendo el mismo día
                    new_timestamp = pd.Timestamp(day) + pd.Timedelta(hours=8 * (i + 1))
                
                # Crear una nueva fila con NaN excepto en 'Age', 'day', 'timestamp' y 'user_id'
                new_row = {col: np.nan for col in df.columns}
                new_row['user_id'] = user_id
                new_row['day'] = day
                new_row['timestamp'] = new_timestamp
                
                # Si 'Age' está en las columnas, conservar el valor original
                if 'Age' in df.columns:
                    new_row['Age'] = day_data['Age'].iloc[0]
                
                # Añadir la nueva fila al DataFrame expandido
                expanded_df = pd.concat([expanded_df, pd.DataFrame([new_row])])

    # Ordenar el DataFrame por usuario y timestamp
    expanded_df = expanded_df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
    return expanded_df


# In[397]:


# Aplicar la función por usuario y día
# Ordenar el dataframe para que sea más sencillo la ampliación
synt_expanded = synthetic_data.sort_values(by=['user_id', 'timestamp'])
#print(synt_expanded)
synt_expanded = add_measures_with_nans(synthetic_data)
print(synt_expanded)


# summary_expanded_data = summarize_missing_values(synt_expanded)
# print(summary_expanded_data)

# Ahora podemos insertar datos en las celdas 'NaN'

# ## INSERCIÓN DE DATOS EN EL DATASET

# Volvemos a cargar nuestro dataframe sintético para un mes y hacemos copia del sintético
# con datos NaN añadidos para no perder los originales. Vamos a estudiar varias formas de interpolación y escoger la más 
# adecuada. 

# In[399]:


import json
import pandas as pd

# Ruta al archivo JSON
file_path = "synthetic_data"

# Abrir y cargar el fichero JSON
with open(file_path, "r") as file:
    data = json.load(file)

# Convertir a DataFrame
synthetic_data= pd.DataFrame(data)

# Mostrar el DataFrame
print(synthetic_data)


# In[401]:


expDataNaN= synt_expanded.copy()
print(expDataNaN)


# ### Opción 1: Función para rellenar con referencia al día anterior/posterior

# Vamos a usar el relleno hacia delante y el relleno hacia atrás para obtener valores aproximados
# de días anteriores y posteriores, y luego aplicamos interpolación temporal para suavizar los valores

# In[403]:


def insercionConDatosPrevios(df):
    """
    Rellena los valores NaN basándose en el día anterior o Posterior para filas insertadas. 
    ARGS:
        - df: dataframe con columnas NaN. 
    RETURNS:
        - pd.Dataframe: dataframe con valores insertados. 
    """

#Nos aseguramos que el índice es el timestamp
    if 'timestamp' in df.columns:
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors = 'coerce')
            df.set_index('timestamp', inplace = True)
        
    #Identifico columnas numericas
    col_numericas = df.select_dtypes(include=['float64', 'int64']).columns 
    
    df_ffill = df[col_numericas].ffill() #valores día anterior
    df_bfill = df[col_numericas].bfill() #valores día posterior
   
    #promedio entre estos valores
    df_promedio = (df_ffill + df_bfill)/2
    
    #Insertamos valores en dataframe original 
    df.update(df_promedio)
    
    df[col_numericas] = df[col_numericas].interpolate(method='time')

    #indices originales intactos
    return df_promedio


# In[405]:


syn_datosprevios = insercionConDatosPrevios(expDataNaN)
print(syn_datosprevios)


# ### Opción 2: Interpolación con K.Nearest Neighbors (KNN)

# Este algoritmo inserta valores basándose en los máss cercanos, y no según el día anterior o posterior 
# como en el anterior

# In[407]:


from fancyimpute import KNN

def insercionConKNN(df, k):
    """
    Interpola valores faltantes en un Dataframe usando K-Nearest Neighbors. 

    ARGS:
        - df: Dataframe a insertar valores. 
        - k: Número de vecinos a considerar para KNN. 
    RETURNS:
        - pd.Dataframe: Dataframe con valores interpolados. 
    """

    #Transformo dataframe a matriz numerica

    matrixDf = df.select_dtypes(include=['float', 'int64'])
    imputed_array = KNN(k=k).fit_transform(matrixDf)

    #Convertir la matriz interpolada de vuelta a df original

    interpolated_df = df.copy()
    interpolated_df[matrixDf.columns] = imputed_array
    print(interpolated_df)

    return interpolated_df
    
    


# In[409]:


syn_knn = insercionConKNN(synt_expanded, 6)
syn_knn.head()


# ### Opción 3: Con Redes Neuronales (ANN)

# Permite capturar patrones más complejos

# In[411]:


import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer

def insercionConANN(df, target_columns):
    """
    Interpola valores faltantes usando MLP, una red neuronal. 
    ARGS:
        - df: dataframe con valores faltantes
        - target_columns (lista): lista de columnas para hacer interpoalción. 
    RETURNS:
        - pd.Dataframe: Dataframe con valores interpolados. 
    """

    df_interpolated = df.copy()
    
    # Seleccionar solo columnas numéricas
    numeric_predictors = df.select_dtypes(include=['float64', 'int64']).drop(columns=target_columns, errors='ignore')

    for target_col in target_columns: 

        if target_col not in df.columns or df[target_col].dtype not in ['float64', 'int64']:
            raise ValueError(f"La columna objetivo '{target_col}' no es numérica o no existe en el DataFrame.")
        
        #definimos predictores y objetivo
        predictors = numeric_predictors.copy()
        target = df[target_col]
        # Imputar valores faltantes en los predictores (solo para entrenamiento)
    
    imputer = SimpleImputer(strategy='mean')

    #dividir datos completos e incompletos
    train_x= predictors[target.notna()]
    train_y= target[target.notna()]
    test_x = predictors[target.isna()]

    #configurar y entrenar la red 
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    mlp.fit(train_x, train_y)

    #predicción de valores
    df_interpolated.loc[target.isna(), target_col] = mlp.predict(test_x)

    return df_interpolated
    
    


# In[413]:


target_columns = ['DiastolicBP', 'SystolicBP', 'BS', 'BodyTemp', 'HeartRate']
synt_ANN = insercionConANN(synt_expanded, target_columns)
print(synt_ANN)


# Vemos que este método falla porque hay columnas con muchos NaN, por tanto no es capaz de predicirlo. 

# Vamos a comparar los dos métodos anteriores cuál nos sirve más para nuestro dataframe!

# ### Comparación de Interpolación con Fill-forward and Fill-Backwards con Interpolaciión con KNN

# In[415]:


from sklearn.metrics import mean_squared_error, mean_absolute_error

def comparaInserciones(df1, df2, method1Name = 'fbfill', method2Name='KNN', columns_to_compare=None):
    """
    Compara dos Dataframes interpolados frente a un Dataframe original en términos  de calidad de los datos obtenida. 

    ARGS:
        - original: dataframe original con valores NaN.
        - df1: dataframe con valores insertados usando Fill-Forward
        - df2: segundo dataframe con valores insertados usando KNN. 
        - columns_to_compare: lista de columnas a incluir en la comparación, si es None, compara todas las columnas. 

    RETURNS:
        - pd.Dataframe: métricas de RMSE y MAE para ambos métodos.
        
    """
    if columns_to_compare is None: #columnas a comparar
        columns_to_compare = list(set(df1.columns) & set(df2.columns))
    results = []

    for col in columns_to_compare:
        #excluir columnas no numéricas

        if df1[col].dtype not  in ['float64', 'int64']:
            continue
        #RMSE y MAE para primer df
        rmse = np.sqrt(mean_squared_error(df1[col], df2[col]))
        mae = mean_absolute_error(df1[col], df2[col])

        
    #Añado resultado a results
        results.append({'Column': col, 'Method1': method1Name,  'Method2': method2Name,'RMSE': rmse, 'MAE': mae})

    return pd.DataFrame(results)
    

    
    


# In[417]:


compResults = comparaInserciones(syn_datosprevios, syn_knn)
print(compResults)


# Como se puede comprobar, en todas las variables (menos en la edad, ya que ahí no se hizo ningún
# cambio) el error del método de KNN es menor, de esta forma usaremos el dataframe interpolado con KNN. 

# ### COMPROBACIÓN DISTRIBUCIÓN DE DATOS

# Vamos a comprobar que todos los datos se distribuyen en el mes de junio a julio como queremos, y en caso
# de encontrar más datos, guardar el dataset sólo con los datos y usuarios de esos meses.

# In[419]:


def count_data_per_month(df, date_col):
    """
    Cuenta el número de datos por mes en un DataFrame. 
    Args:
        df: Dataframe
        date_col: columna con fechas formato datetime
    Returns: 
        pd.Dataframe con el conteo de datos por mes
    """
    df['month&year'] = pd.to_datetime(df[date_col])
    df['month&year'] = df['month&year'].dt.to_period('M')
    
    # Contar datos por mes
    conteo_mensual = df.groupby('month&year').size().reset_index(name='conteo')
    
    return conteo_mensual


# In[421]:


conteo_mensual = count_data_per_month(syn_knn, 'day')
print(conteo_mensual)


# In[423]:


# Filtrar los meses de junio y julio de 2024
syn_knn_filtrado = syn_knn[(syn_knn['timestamp'] >= '2024-06') & (syn_knn['timestamp'] <= '2024-07')]
print(syn_knn_filtrado)


# In[425]:


syn_knn_filtrado.to_csv('wearable-primerMes-knn-syn', index=False)  # index=False evita guardar el índice del DataFrame


# In[ ]:





# In[ ]:




