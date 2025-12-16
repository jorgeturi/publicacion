import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model  # Cargar modelo
import joblib #cargar escaladores
import pandas as pd # cargar datos 
import os
import shutil #para copiar archivo
import pickle
import sys #para terminar 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping


def cargar_modelo(ruta_al_modelo="default.txt"):
    """
    Carga un modelo desde un archivo y registra el resultado en el log.

    Args:
    ruta_al_modelo (str): La ruta al archivo del modelo.

    Returns:
    model: El modelo cargado si la carga fue exitosa, None sino.
    """
    try:
        # Intentar cargar el modelo
        model = load_model(ruta_al_modelo)
        logging.info(f'Modelo cargado exitosamente desde {ruta_al_modelo}.')
        return model
    except Exception as e:
        logging.error(f'Error al cargar el modelo desde {ruta_al_modelo}: {e}')
        return None
    

def cargar_escaladores(ruta_a_escaladores="rutafake.txt"):
    """
    Carga escaladores desde un archivo y registra el resultado en el log.

    Args:
    ruta_a_escaladores (str): La ruta a escaladores del modelo.

    Returns:
    escaladores: escaladores cargados si la carga fue exitosa, None sino.
    """
    try:
        # Intentar cargar el modelo
        scalers = joblib.load(ruta_a_escaladores)

        logging.info(f'Escaladores cargados exitosamente desde {ruta_a_escaladores}.')
        return scalers
    except Exception as e:
        logging.error(f'Error al cargar escaladores desde {ruta_a_escaladores}: {e}')
        return None




def crear_ventana(dataset, ventana_entrada, ventana_salida):
    logging.info("Creando ventanas.")
    print("el dataset es", dataset.shape)
    # Agregar las columnas t-1, t-2, t-3, t-4 al DataFrame
    #for i in range(0, mediciones_previas):  # Empieza desde 1 hasta ventana_entrada (incluye 8 si ventana_entrada = 8)
    #    dataset[f'activa_t-{i}'] = dataset['activa'].shift(i)

    # Eliminar las filas con valores nulos causados por el desplazamiento
    dataset = dataset.dropna()


    # Extraer las características necesarias (t-4, t-3, t-2, t-1 y t)
    columnas_necesarias = ['activa'] + ['pico'] + ['dia_sen'] + ['dia_cos'] + ['mes_sen'] + ['mes_cos'] #+ ['dia_habil'] 

    #columnas_necesarias = [f'activa_t-{i}' for i in range(0,mediciones_previas)] + ['pico'] + ['dia_sen'] + ['dia_cos'] + ['mes_sen'] + ['mes_cos'] +['l1'] +['l2'] + ['l3']
    #columnas_necesarias = [f'activa_t-{i}' for i in range(0,mediciones_previas)] + ['pico'] +['l1'] +['l2'] + ['l3']

    # Incluir las columnas necesarias en el DataFrame
    features = dataset[columnas_necesarias].values

    # Calcular el número total de ventanas que se pueden crear
    total_muestras = len(dataset) - ventana_entrada - ventana_salida + 1

    # Ventanas de entrada (características de t-4, t-3, t-2, t-1 y t)
    X = np.array([features[i:i + ventana_entrada] for i in range(total_muestras)])

    # Ventanas de salida (valor futuro de la potencia activa)
    y = np.array([dataset['activa'].values[i+ventana_entrada :i+ventana_entrada + ventana_salida] for i in range(total_muestras)])
    #y = np.array([dataset['activa'].values[i + ventana_entrada : i + ventana_entrada + ventana_salida] for i in range(total_muestras)])
    #y = y.reshape(-1, ventana_entrada-1, ventana_salida)
    #y = np.array([dataset['activa'].values[i + ventana_entrada-1] for i in range(total_muestras)])
    #y = y.reshape(-1,1)
    logging.info("Ventanas creadas")

    return X, y


def crear_ventana2(dataset, ventana_entrada, ventana_salida, mediciones_previas):
    logging.info("Creando ventanas.")

    # Agregar las columnas t-1, t-2, t-3, t-4 al DataFrame
    for i in range(1, mediciones_previas + 1):  # Empieza desde 1 hasta ventana_entrada (incluye 8 si ventana_entrada = 8)
        dataset[f'activa_t-{i}'] = dataset['activa'].shift(i)

    # Eliminar las filas con valores nulos causados por el desplazamiento
    dataset = dataset.dropna()

    # Extraer las características necesarias (t-4, t-3, t-2, t-1 y t)
    columnas_necesarias = [f'activa_t-{i}' for i in range(1,mediciones_previas+1)]

    # Incluir las columnas necesarias en el DataFrame
    features = dataset[columnas_necesarias].values

    # Calcular el número total de ventanas que se pueden crear
    total_muestras = len(dataset) - ventana_entrada - ventana_salida + 1

    # Ventanas de entrada (características de t-4, t-3, t-2, t-1 y t)
    X = np.array([features[i:i + ventana_entrada] for i in range(total_muestras)])

    # Ventanas de salida (valor futuro de la potencia activa)
    #y = np.array([dataset['activa'].values[i :i + ventana_entrada-1] for i in range(total_muestras)])
    #y = np.array([dataset['activa'].values[i + ventana_entrada : i + ventana_entrada + ventana_salida] for i in range(total_muestras)])
    #y = y.reshape(-1, ventana_entrada-1, ventana_salida)
    y = np.array([dataset['activa'].values[i + ventana_entrada-1] for i in range(total_muestras)])
    y = y.reshape(-1,1)
    logging.info("Ventanas creadas")

    return X, y


def crear_ventana3(dataset, ventana_entrada, ventana_salida):
    logging.info("Creando ventanas.")

    # Agregar las columnas t-1, t-2, t-3, t-4 para 'activa'
    for i in range(0, ventana_entrada + 1):
        dataset.loc[:, f'activa_t-{i}'] = dataset['activa'].shift(i)
    
    # Agregar las columnas t-1, t-2, t-3, t-4 para las otras variables
    for var in ['dia_sen', 'dia_cos', 'mes_sen', 'mes_cos']:
        for i in range(0, ventana_entrada + 1):
            dataset.loc[:, f'{var}_t-{i}'] = dataset[var].shift(i)

    # Eliminar las filas con valores nulos causados por el desplazamiento
    dataset = dataset.dropna()

    # Extraer las características necesarias (t-4, t-3, t-2, t-1 y t para cada variable)
    columnas_necesarias = [f'activa_t-{i}' for i in range(ventana_entrada)]
    for var in ['dia_sen', 'dia_cos', 'mes_sen', 'mes_cos']:
        columnas_necesarias += [f'{var}_t-{i}' for i in range(ventana_entrada)]
    
    # Incluir las columnas necesarias en el DataFrame
    features = dataset[columnas_necesarias].values

    # Calcular el número total de ventanas que se pueden crear
    total_muestras = len(dataset) - ventana_entrada - ventana_salida + 1

    # Ventanas de entrada (características de t-4, t-3, t-2, t-1 y t para cada variable)
    X = np.array([features[i:i + ventana_entrada] for i in range(total_muestras)])

    # Ventanas de salida (valor futuro de la potencia activa)
    y = np.array([dataset['activa'].values[i + 1:i + 1 + ventana_entrada] for i in range(total_muestras)])

    logging.info("Ventanas creadas")

    return X, y



def crear_ventana4(dataset, ventana_entrada, ventana_salida, mediciones_previas):
    logging.info("Creando ventanas.")
    print("el dataset es", dataset.shape)
    # Agregar las columnas t-1, t-2, t-3, t-4 al DataFrame
    for i in range(0, mediciones_previas):  # Empieza desde 1 hasta ventana_entrada (incluye 8 si ventana_entrada = 8)
        dataset[f'activa_t-{i}'] = dataset['activa'].shift(i)

    # Eliminar las filas con valores nulos causados por el desplazamiento
    dataset = dataset.dropna()


    # Extraer las características necesarias (t-4, t-3, t-2, t-1 y t)
    columnas_necesarias = [f'activa_t-{i}' for i in range(0,mediciones_previas)] + ['pico'] + ['dia_sen'] + ['dia_cos'] + ['mes_sen'] + ['mes_cos'] #+ ['dia_habil'] 

    #columnas_necesarias = [f'activa_t-{i}' for i in range(0,mediciones_previas)] + ['pico'] + ['dia_sen'] + ['dia_cos'] + ['mes_sen'] + ['mes_cos'] +['l1'] +['l2'] + ['l3']
    #columnas_necesarias = [f'activa_t-{i}' for i in range(0,mediciones_previas)] + ['pico'] +['l1'] +['l2'] + ['l3']

    # Incluir las columnas necesarias en el DataFrame
    features = dataset[columnas_necesarias].values

    # Calcular el número total de ventanas que se pueden crear
    total_muestras = len(dataset) - ventana_entrada - ventana_salida + 1

    # Ventanas de entrada (características de t-4, t-3, t-2, t-1 y t)
    X = np.array([features[i:i + ventana_entrada] for i in range(total_muestras)])

    # Ventanas de salida (valor futuro de la potencia activa)
    y = np.array([dataset['activa'].values[i+ventana_entrada :i+ventana_entrada + ventana_salida] for i in range(total_muestras)])
    #y = np.array([dataset['activa'].values[i + ventana_entrada : i + ventana_entrada + ventana_salida] for i in range(total_muestras)])
    #y = y.reshape(-1, ventana_entrada-1, ventana_salida)
    #y = np.array([dataset['activa'].values[i + ventana_entrada-1] for i in range(total_muestras)])
    #y = y.reshape(-1,1)
    logging.info("Ventanas creadas")

    return X, y



def crear_ventana5(dataset, ventana_entrada, ventana_salida, mediciones_previas):
    logging.info("Creando ventanas.")

    # Agregar las columnas t-1, t-2, t-3, t-4 al DataFrame
    for i in range(1, mediciones_previas + 1):  # Empieza desde 1 hasta ventana_entrada (incluye 8 si ventana_entrada = 8)
        dataset[f'activa_t-{i}'] = dataset['activa'].shift(i)

    # Eliminar las filas con valores nulos causados por el desplazamiento
    dataset = dataset.dropna()

    # Extraer las características necesarias (t-4, t-3, t-2, t-1 y t)
    columnas_necesarias = [f'activa_t-{i}' for i in range(1,mediciones_previas+1)] + ['pico']

    # Incluir las columnas necesarias en el DataFrame
    features = dataset[columnas_necesarias].values

    # Calcular el número total de ventanas que se pueden crear
    total_muestras = len(dataset) - ventana_entrada - ventana_salida + 1

    # Ventanas de entrada (características de t-4, t-3, t-2, t-1 y t)
    X = np.array([features[i:i + ventana_entrada] for i in range(total_muestras)])

    # Ventanas de salida (valor futuro de la potencia activa)
    y = np.array([dataset['activa'].values[i :i + ventana_entrada] for i in range(total_muestras)])
    #y = np.array([dataset['activa'].values[i + ventana_entrada : i + ventana_entrada + ventana_salida] for i in range(total_muestras)])
    y = y.reshape(-1, ventana_entrada, ventana_salida)
    #y = np.array([dataset['activa'].values[i + ventana_entrada-1] for i in range(total_muestras)])
    #y = y.reshape(-1,1)
    logging.info("Ventanas creadas")

    return X, y


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def cargar_datos_especificos(archivo_potencias='potencias.csv', dias_semanales=None, horas=None):
    print("esto es cargar datos de prediccion potencia")
    """
    Carga los datos desde el archivo CSV de potencias, filtra según días de la semana y horas, y registra el resultado en el log.

    Args:
    archivo_potencias (str): La ruta al archivo de potencias (csv).
    dias_semanales (list): Lista de días de la semana a cargar (0=domingo, 1=lunes, ..., 6=sábado). 
                           Si es None, no filtra por días de la semana.
    horas (list): Lista de horas (0-23) a cargar. Si es None, no filtra por horas.
    
    Returns:
    final_df: dataframe con los datos si la carga fue exitosa, None sino.
    """
    try:
        # Leer encabezados para verificar que los archivos se pueden abrir
        encabezados_potencias = pd.read_csv(archivo_potencias, nrows=0).columns
        if 'timestamp;;;;' in encabezados_potencias: ###algo que se genero al modificar
            encabezados_potencias = encabezados_potencias.str.replace('timestamp;;;;', 'timestamp')

        fila_inicio = 1
        numero_filas = 120000  # Número de filas a cargar
        
        # Leer el archivo CSV de potencias
        potencias = pd.read_csv(archivo_potencias, skiprows=fila_inicio, nrows=numero_filas, header=None, names=encabezados_potencias)

        
        
        # Convertir la columna 'timestamp' a datetime
        potencias['timestamp'] = pd.to_datetime(potencias['timestamp'], errors='coerce')

        # Verificar si alguna fecha no fue convertida correctamente
        if potencias['timestamp'].isnull().any():
            raise ValueError("Algunas fechas no pudieron ser convertidas correctamente en el archivo de potencias.")

        # Codificación del tiempo del día
        #potencias['tiempo_del_dia'] = potencias['timestamp'].dt.hour + potencias['timestamp'].dt.minute / 60.0
        #potencias['dia_sen'] = np.sin(2 * np.pi * potencias['tiempo_del_dia'] / 24)
        #potencias['dia_cos'] = np.cos(2 * np.pi * potencias['tiempo_del_dia'] / 24)

        # Codificación del día del año
        #potencias['dia_del_año'] = potencias['timestamp'].dt.dayofyear
        #potencias['mes_sen'] = np.sin(2 * np.pi * potencias['dia_del_año'] / 365)
        #potencias['mes_cos'] = np.cos(2 * np.pi * potencias['dia_del_año'] / 365)


        # Normalización de la codificación cíclica a rango [0, 1]
        #potencias['dia_sen'] = (potencias['dia_sen'] + 1) / 2
        #potencias['dia_cos'] = (potencias['dia_cos'] + 1) / 2
        #potencias['mes_sen'] = (potencias['mes_sen'] + 1) / 2
        #potencias['mes_cos'] = (potencias['mes_cos'] + 1) / 2

        # Filtrar por días de la semana
        if dias_semanales is not None:
            potencias = potencias[potencias['timestamp'].dt.dayofweek.isin(dias_semanales)]

        # Filtrar por horas
        if horas is not None:
            potencias = potencias[potencias['timestamp'].dt.hour.isin(horas)]
        potencias_2 = potencias.groupby(potencias['timestamp'].dt.floor('h')).first()
        #potencias_2 = potencias.reset_index(drop=False)  # Asegura que no se duplica 'timestam
        #potencias_2 = potencias
        print("WNRWWWWWWWWWWWWWWWWWWWW ACA")


        
        potencias_2['tiempo_del_dia'] = potencias_2['timestamp'].dt.hour + potencias_2['timestamp'].dt.minute / 60.0
        potencias_2['dia_sen'] = np.sin(2 * np.pi * potencias_2['tiempo_del_dia'] / 24)
        potencias_2['dia_cos'] = np.cos(2 * np.pi * potencias_2['tiempo_del_dia'] / 24)

        # Codificación del día del año
        #potencias_2['dia_del_año'] = potencias_2['timestamp'].dt.dayofyear
        #potencias_2['mes_sen'] = np.sin(2 * np.pi * potencias_2['dia_del_año'] / 365)
        #potencias_2['mes_cos'] = np.cos(2 * np.pi * potencias_2['dia_del_año'] / 365)
        ##HABIA UN DESFASE POR EL AÑO BISIESTO
        dias_en_el_anio = potencias_2['timestamp'].dt.is_leap_year.map(lambda x: 366 if x else 365)
        potencias_2['dia_del_año'] = potencias_2['timestamp'].dt.dayofyear
        potencias_2['mes_sen'] = np.sin(2 * np.pi * potencias_2['dia_del_año'] / dias_en_el_anio)
        potencias_2['mes_cos'] = np.cos(2 * np.pi * potencias_2['dia_del_año'] / dias_en_el_anio)


        #print(potencias_2)
        #print(potencias)

        # Calcular la diferencia con el valor anterior en la columna 'activa'
        potencias_2['diferencia_activa'] = potencias_2['activa'].diff()
        potencias_2['pico'] = np.where(np.abs(potencias_2['diferencia_activa']) > 5, 1, 0)
        #potencias = potencias_2
        # Agregar la columna "n° de medición"
        #potencias['numero_de_medicion'] = range(1, len(potencias) + 1)

       
        # Reorganizacion las columnas en el formato deseado
        final_df = potencias_2[['activa', 'dia_sen', 'dia_cos', 'mes_sen', 'mes_cos', 'pico', 'timestamp']]
        #final_df = potencias[['activa']]

        # Graficar la codificación cíclica para verificar la distribución en el círculo
        """sin_hora = potencias_2['dia_sen'].iloc[:24]  # Extraer los primeros 24 valores
        cos_hora = potencias_2['dia_cos'].iloc[:24]  # Extraer los primeros 24 valores

        plt.figure(figsize=(6, 6))
        plt.scatter(sin_hora, cos_hora)
        plt.title("Codificación cíclica de hora del día (sin vs cos) - primeros 24 valores")
        plt.xlabel("sin(hora)")
        plt.ylabel("cos(hora)")
        plt.axis("equal")
        plt.grid(True)
        plt.show()"""



        logging.info(f"Datos cargados y filtrados. Total de registros: {len(final_df)}")

        if np.any(np.isnan(final_df)):
            print("Hay valores NaN en los datos.")
        if np.any(np.isinf(final_df)):
            print("Hay valores infinitos en los datos.")
        
        return final_df
    
    except Exception as e:
        logging.error(f'Error al cargar los datos: {e}')
        return None


def agregar_dia_habil(df):
    """
    Agrega una columna 'dia_habil' con valor 1 para lunes a viernes y 0 para sábado y domingo.
    """
    # Añadir la columna 'dia_habil' (1 para lunes a viernes, 0 para sábado y domingo)
    df['dia_habil'] = df['Time'].dt.dayofweek.apply(lambda x: 1 if x < 5 else 0)
    return df

def asignar_dia_habil(df, dia, mes, año):
    """
    Asigna un 0 en la columna 'dia_habil' para una fecha específica.
    
    Args:
    - df: DataFrame con la columna 'timestamp' y 'dia_habil'.
    - dia: Día del mes (1-31).
    - mes: Mes (1-12).
    - año: Año (por ejemplo, 2023).
    """
    # Crear un objeto de tipo datetime con el día, mes y año proporcionado
    fecha_objetivo = pd.Timestamp(year=año, month=mes, day=dia)
    
    # Asignar 0 a 'dia_habil' en las filas que correspondan a esa fecha
    df.loc[df['Time'].dt.date == fecha_objetivo.date(), 'dia_habil'] = 0
    
    return df


def cargar_datos_especificos2(archivo_potencias='potencias.csv', archivo_corrientes = 'corrientes.csv', dias_semanales=None, horas=None):
    """
    Carga los datos desde el archivo CSV de potencias, filtra según días de la semana y horas, y registra el resultado en el log.

    Args:
    archivo_potencias (str): La ruta al archivo de potencias (csv).
    dias_semanales (list): Lista de días de la semana a cargar (0=domingo, 1=lunes, ..., 6=sábado). 
                           Si es None, no filtra por días de la semana.
    horas (list): Lista de horas (0-23) a cargar. Si es None, no filtra por horas.
    
    Returns:
    final_df: dataframe con los datos si la carga fue exitosa, None sino.
    """
    try:
        # Leer encabezados para verificar que los archivos se pueden abrir
        encabezados_corrientes = pd.read_csv(archivo_corrientes, nrows=0).columns
        encabezados_potencias = pd.read_csv(archivo_potencias, nrows=0).columns
        
        fila_inicio = 1
        numero_filas = 120000  # Número de filas que deseas cargar
        
        # Leer los archivos CSV
        print("Cargando datos de corrientes y potencias...")
        corrientes = pd.read_csv(archivo_corrientes, skiprows=fila_inicio, nrows=numero_filas, header=None, names=encabezados_corrientes)
        potencias = pd.read_csv(archivo_potencias, skiprows=fila_inicio, nrows=numero_filas, header=None, names=encabezados_potencias)
        
        print("Archivos cargados correctamente.")

        # Convertir la columna 'timestamp' a datetime
        corrientes['timestamp'] = pd.to_datetime(corrientes['timestamp'])
        potencias['timestamp'] = pd.to_datetime(potencias['timestamp'])


        potencias = pd.merge(corrientes, potencias, on=['timestamp'])

        # Verificar si alguna fecha no fue convertida correctamente
        if potencias['timestamp'].isnull().any():
            raise ValueError("Algunas fechas no pudieron ser convertidas correctamente en el archivo de potencias.")

        # Codificación del tiempo del día
        potencias['tiempo_del_dia'] = potencias['timestamp'].dt.hour + potencias['timestamp'].dt.minute / 60.0
        potencias['dia_sen'] = np.sin(2 * np.pi * potencias['tiempo_del_dia'] / 24)
        potencias['dia_cos'] = np.cos(2 * np.pi * potencias['tiempo_del_dia'] / 24)

        # Codificación del día del año
        potencias['dia_del_año'] = potencias['timestamp'].dt.dayofyear
        potencias['mes_sen'] = np.sin(2 * np.pi * potencias['dia_del_año'] / 365)
        potencias['mes_cos'] = np.cos(2 * np.pi * potencias['dia_del_año'] / 365)


        # Normalización de la codificación cíclica a rango [0, 1]
        potencias['dia_sen'] = (potencias['dia_sen'] + 1) / 2
        potencias['dia_cos'] = (potencias['dia_cos'] + 1) / 2
        potencias['mes_sen'] = (potencias['mes_sen'] + 1) / 2
        potencias['mes_cos'] = (potencias['mes_cos'] + 1) / 2

        # Filtrar por días de la semana
        if dias_semanales is not None:
            potencias = potencias[potencias['timestamp'].dt.dayofweek.isin(dias_semanales)]

        # Filtrar por horas
        if horas is not None:
            potencias = potencias[potencias['timestamp'].dt.hour.isin(horas)]
        potencias_2 = potencias.groupby(potencias['timestamp'].dt.floor('h')).first()
        #potencias_2 = potencias.reset_index(drop=False)  # Asegura que no se duplica 'timestam
        potencias_2 = potencias
        
        #print(potencias_2)
        #print(potencias)

        # Calcular la diferencia con el valor anterior en la columna 'activa'
        potencias_2['diferencia_activa'] = potencias_2['activa'].diff()
        potencias_2['pico'] = np.where(np.abs(potencias_2['diferencia_activa']) > 10, 1, 0)
        potencias = potencias_2
        # Agregar la columna "n° de medición"
        #potencias['numero_de_medicion'] = range(1, len(potencias) + 1)

       
        # Reorganizacion las columnas en el formato deseado
        final_df = potencias[['activa', 'dia_sen', 'dia_cos', 'mes_sen', 'mes_cos', 'pico','l1','l2','l3']]
        #final_df = potencias[['activa']]

        logging.info(f"Datos cargados y filtrados. Total de registros: {len(final_df)}")

        if np.any(np.isnan(final_df)):
            print("Hay valores NaN en los datos.")
        if np.any(np.isinf(final_df)):
            print("Hay valores infinitos en los datos.")
        
        return final_df
    
    except Exception as e:
        logging.error(f'Error al cargar los datos: {e}')
        return None






import numpy as np
import pandas as pd
import os

def calcular_resultados(ytest, prediccionesTest, carpeta):
    # Crear listas para almacenar los resultados
    resultados = []
    errores_totales = []  # Para almacenar todos los errores

    # Inicializar listas para métricas por columna
    r2_por_columna = []
    mae_por_columna = []
    rmse_por_columna = []  # Nueva lista para RMSE por columna
    desviacion_estandar_por_columna = []

    for valor in range(len(ytest)):
        y_real = ytest[valor]
        prediccion = prediccionesTest[valor]
        
        # Calcular errores
        errores = [prediccion[i] - y_real[i] for i in range(len(y_real))]
        errores_totales.extend(errores)  # Guardamos todos los errores para análisis global

        # Calcular error promedio y desviación estándar
        error_promedio = np.mean(np.abs(errores))
        desviacion_estandar = np.std(errores)
        
        # Error relativo porcentual
        error_relativo_porcentual = [(abs(errores[i]) / abs(y_real[i])) * 100 if y_real[i] != 0 else 0 for i in range(len(y_real))]
        
        # Almacenar resultados
        for i in range(len(y_real)):
            resultados.append({
                'valor': valor,
                'prediccion': prediccion[i],
                'valor_real': y_real[i],
                'error': errores[i],
                'error_promedio': error_promedio,
                'desviacion_estandar': desviacion_estandar,
                'error_relativo_porcentual': error_relativo_porcentual[i]
            })

    # Convertir a DataFrame
    df_resultados = pd.DataFrame(resultados)

    # Calcular métricas globales
    error_promedio_global = np.mean(errores_totales)
    desviacion_estandar_global = np.std(errores_totales)
    error_relativo_porcentual_promedio = np.mean(df_resultados['error_relativo_porcentual'])

    # Calcular RMSE global
    rmse_global = np.sqrt(mean_squared_error(ytest.flatten(), prediccionesTest.flatten()))  # RMSE global

    # Calcular las métricas de errores menores a 1, 2, 3, 4, 5
    errores_menores_a_1 = sum(abs(error) <= 1 for error in errores_totales)
    errores_menores_a_2 = sum(abs(error) <= 2 for error in errores_totales)
    errores_menores_a_3 = sum(abs(error) <= 3 for error in errores_totales)
    errores_menores_a_4 = sum(abs(error) <= 4 for error in errores_totales)
    errores_menores_a_5 = sum(abs(error) <= 5 for error in errores_totales)
    
    total_datos = len(errores_totales)

    # Calcular el porcentaje de errores menores a 1, 2, 3, 4, 5
    porcentaje_menores_a_1 = (errores_menores_a_1 / total_datos) * 100
    porcentaje_menores_a_2 = (errores_menores_a_2 / total_datos) * 100
    porcentaje_menores_a_3 = (errores_menores_a_3 / total_datos) * 100
    porcentaje_menores_a_4 = (errores_menores_a_4 / total_datos) * 100
    porcentaje_menores_a_5 = (errores_menores_a_5 / total_datos) * 100

    # Error máximo
    error_maximo = max(abs(error) for error in errores_totales)

    # Calcular MAPE por columna
    from sklearn.metrics import mean_absolute_percentage_error
    mape_por_columna = [mean_absolute_percentage_error(ytest[:, i], prediccionesTest[:, i]) for i in range(ytest.shape[1])]

    # Calcular R² por columna
    r2_por_columna = [r2_score(ytest[:, i], prediccionesTest[:, i]) for i in range(ytest.shape[1])]

    # Calcular MAE por columna
    mae_por_columna = [mean_absolute_error(ytest[:, i], prediccionesTest[:, i]) for i in range(ytest.shape[1])]
    mae_por_columna = [f"{mae:.2f}" for mae in mae_por_columna]

    # Calcular RMSE por columna
    rmse_por_columna = [np.sqrt(mean_squared_error(ytest[:, i], prediccionesTest[:, i])) for i in range(ytest.shape[1])]
    rmse_por_columna = [f"{rmse:.2f}" for rmse in rmse_por_columna]

    r2_por_columna = [f"{r2:.2f}" for r2 in r2_por_columna]
    
    # Promediar los MAPE de todas las columnas para obtener un único valor
    mape_promedio = np.mean(mape_por_columna)
    mape_por_columna = [f"{mape * 100:.2f}%" for mape in mape_por_columna]

    # Calcular desviación estándar por columna
    desviacion_estandar_por_columna = [np.std(prediccionesTest[:, i] - ytest[:, i]) for i in range(ytest.shape[1])]
    desviacion_estandar_por_columna = [f"{desviacion:.2f}" for desviacion in desviacion_estandar_por_columna]

    # Guardar resultados globales en un archivo de texto
    resultados_txt_path = os.path.join(carpeta, 'resultados.txt')
    with open(resultados_txt_path, 'a') as f:
        f.write("\n")
        f.write(f"Error promedio global: {error_promedio_global:.2f}\n")
        f.write(f"Desviación estándar global: {desviacion_estandar_global:.2f}\n")
        f.write(f"Error relativo porcentual promedio global: {error_relativo_porcentual_promedio:.2f}%\n")
        f.write(f"RMSE global: {rmse_global:.2f}\n")  # RMSE global
        f.write("\n")
        f.write(f"MAPE por columna: {mape_por_columna}\n")
        f.write(f"MAPE promedio: {mape_promedio*100:.2f}%\n")
        f.write(f"R² por columna: {r2_por_columna}\n")
        f.write(f"MAE por columna: {mae_por_columna}\n")
        f.write(f"RMSE por columna: {rmse_por_columna}\n")  # RMSE por columna
        f.write(f"Desviación estándar por columna: {desviacion_estandar_por_columna}\n")
        f.write(f"Cantidad de datos: {total_datos}\n")
        f.write("\n")
        f.write(f"Cantidad de errores menores o iguales a 1: {errores_menores_a_1} ({porcentaje_menores_a_1:.2f}%)\n")
        f.write(f"Cantidad de errores menores o iguales a 2: {errores_menores_a_2} ({porcentaje_menores_a_2:.2f}%)\n")
        f.write(f"Cantidad de errores menores o iguales a 3: {errores_menores_a_3} ({porcentaje_menores_a_3:.2f}%)\n")
        f.write(f"Cantidad de errores menores o iguales a 4: {errores_menores_a_4} ({porcentaje_menores_a_4:.2f}%)\n")
        f.write(f"Cantidad de errores menores o iguales a 5: {errores_menores_a_5} ({porcentaje_menores_a_5:.2f}%)\n")
        f.write(f"El error más grande cometido: {error_maximo:.2f}\n")

    # Guardar a un archivo CSV los resultados individuales
    resultados_csv_path = os.path.join(carpeta, 'resultados_predicciones.csv')
    df_resultados.to_csv(resultados_csv_path, index=False)




def crear_carpeta_y_guardar(nombre_modelo):
    carpeta = f"modelos/{nombre_modelo}"
    
    # Verificar si la carpeta ya existe
    if os.path.exists(carpeta):
        respuesta = input(f"La carpeta '{carpeta}' ya existe. ¿Deseas sobrescribirla? (s/n): ").strip().lower()
        if respuesta != 's':
            print("Operación cancelada. No se sobrescribió la carpeta.")
            sys.exit(-1)
            return None, None
    
    # Crear la carpeta con el nombre del modelo
    os.makedirs(carpeta, exist_ok=True)

    # Copiar los scripts a la carpeta
    ruta_script = 'tools/red_principal.py'  # Ruta del script
    destino_script = os.path.join(carpeta, 'red_principal.py')
    shutil.copy(ruta_script, destino_script)

    ruta_script = 'principal.py'  # Ruta del script
    destino_script = os.path.join(carpeta, 'principal.py')
    shutil.copy(ruta_script, destino_script)

    # Crear un archivo de resultados
    resultados_path = os.path.join(carpeta, 'resultados.txt')

    # Si el archivo de resultados ya existe, preguntar si se desea sobrescribir
    if os.path.exists(resultados_path):
        respuesta = input(f"El archivo '{resultados_path}' ya existe. ¿Deseas sobrescribirlo? (s/n): ").strip().lower()
        if respuesta != 's':
            print("Operación cancelada. No se sobrescribió el archivo de resultados.")
            return None, None
    
    # Abrir el archivo de resultados para escribir
    with open(resultados_path, 'w') as f:
        f.write(f"Resultados de la predicción para el {nombre_modelo}:\n")

    print(f"Carpeta '{carpeta}' y archivo de resultados creado exitosamente.")
    return carpeta, resultados_path


def guardar_modelo_y_resultados(carpeta, modelo, scalers):
    # Guardar el modelo
    modelo_path = os.path.join(carpeta, 'modelo')
    modelo.save(modelo_path)

    # Guardar los escaladores
    scalers_path = os.path.join(carpeta, 'scalers.pkl')
    with open(scalers_path, 'wb') as f:
        pickle.dump(scalers, f)


    return modelo_path, scalers_path








#boundaries = [5, 10, 80, 120, 180, 250]  # Los límites de los intervalos (épocas en este caso)
#values = [0.001, 0.0001, 0.00001, 0.000001, 0.00005, 0.00001, 0.000001]  # Learning rates correspondientes a los intervalos
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop, Adadelta, Nadam
from tensorflow.keras import backend as K


from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, InputLayer

def entrenar_modelo(Xtrain, ytrain, Xval, yval, path_guardado='modelo_entrenado.h5'):
    # Asegurar reproducibilidad
    np.random.seed(47)
    tf.random.set_seed(47)
    initializer = GlorotUniform(seed=47)

    # Define los intervalos y los valores de learning rate
    boundaries = [5, 10, 20, 50, 100, 250,350,450]  # Los límites de los intervalos (épocas en este caso)
    values = [0.01, 0.004, 0.002, 0.001, 0.00005, 0.00001, 0.000004,0.000001,0.0000001]  # Learning rates correspondientes a los intervalos

    # Crea el scheduler de learning rate
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=boundaries,
        values=values
    )
    # Crear el modelo LSTM
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, dilation_rate=1,input_shape=(Xtrain.shape[1], Xtrain.shape[2]), activation='relu', padding='causal'))
    model.add(Conv1D(filters=32, kernel_size=2, dilation_rate=1, activation='relu', padding='causal'))

    model.add(LSTM(100, return_sequences=False, kernel_initializer=initializer ) )
    #model.add(BatchNormalization())
    #model.add(Dropout(0.2)) 
    #model.add(LSTM(100, return_sequences=False, kernel_initializer=initializer ) )
    #model.add(Dense(300, activation="linear"))
    #model.add(BatchNormalization())

    #model.add(Dropout(0.4))

    #model.add(Dense(100, activation="linear"))
    #model.add(Dropout(0.1))
    #model.add(BatchNormalization())
    #model.add(LSTM(128, return_sequences=True, kernel_initializer=initializer ) )
    #model.add(BatchNormalization())
    #model.add(LSTM(32, return_sequences=True, kernel_initializer=initializer ) )
    #model.add(BatchNormalization())
  
    #model.add(Dropout(0.1))
    #model.add(BatchNormalization())
    #model.add(LSTM(50, return_sequences=False, kernel_initializer=initializer ) )
    #model.add(Dropout(0.1))
    #model.add(BatchNormalization())

    model.add(Dense(ytrain.shape[1], activation="linear"))

    # Compilar el modelo con el optimizador personalizado
    #optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    #optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)

    model.compile(optimizer='adam', loss="mse")

    # EarlyStopping para evitar sobreajuste
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # ModelCheckpoint para guardar el modelo durante el entrenamiento
    #checkpoint = ModelCheckpoint(path_guardado, monitor='val_loss', save_best_only=True, verbose=1)

    try:
        # Entrenar el modelo con datos de validación, EarlyStopping y ModelCheckpoint
        model.fit(Xtrain, ytrain, epochs=100, verbose=1, batch_size=4,
                  validation_data=(Xval, yval), callbacks=[early_stopping])
    except Exception as e:
        print(f"Se produjo un error: {e}")
        print("Guardando el modelo hasta el último punto alcanzado...")
        model.save(path_guardado)  # Guarda el modelo si ocurre otro error

    return model


from tensorflow.keras.layers import Flatten
def entrenar_con_rnn(Xtrain, ytrain, Xval, yval, path_guardado='modelo_entrenado_rnn.h5'):
    # Asegurar reproducibilidad
    np.random.seed(47)
    tf.random.set_seed(47)
    initializer = GlorotUniform(seed=47)

    # Define los intervalos y los valores de learning rate
    boundaries = [5, 10, 20, 50, 100, 250]
    values = [0.05, 0.02, 0.01, 0.001, 0.0005, 0.0001, 0.00001]

    # Crea el scheduler de learning rate
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=boundaries,
        values=values
    )

    # Crear el modelo Feedforward
    model = Sequential()
    
    model.add(Dense(500, activation='relu', kernel_initializer=initializer, input_shape=(Xtrain.shape[1], Xtrain.shape[2])))  # (Tiempos, Features)
    model.add(Flatten())  # Convierte (4,6) → (24,)
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(450, activation='relu', kernel_initializer=initializer))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
   
    model.add(Dense(ytrain.shape[1], activation="linear"))  # Salida de 6 características

    # Compilar el modelo con el optimizador personalizado
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mae')

    # EarlyStopping para evitar sobreajuste
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    try:
        # Entrenar el modelo con datos de validación y EarlyStopping
        model.fit(Xtrain, ytrain, epochs=300, verbose=1, batch_size=16,
                  validation_data=(Xval, yval), callbacks=[early_stopping])
    except Exception as e:
        print(f"Se produjo un error: {e}")
        print("Guardando el modelo hasta el último punto alcanzado...")
        model.save(path_guardado)  # Guarda el modelo si ocurre error
    return model



def cargar_muestra_especifica(archivo_potencias='potencias.csv', dias_semanales=None, horas=None,
                             timestamp_inicio=None, pasos=48):
    """
    Carga los datos desde un archivo CSV y filtra por días de semana y horas específicas.
    Realiza el preprocesamiento, como la codificación cíclica de hora y mes, y la detección de picos.
    Devuelve una secuencia de tamaño fijo para usar en LSTM.

    Parámetros:
    archivo_potencias (str): Ruta al archivo CSV de potencias.
    dias_semanales (list): Lista de días de la semana a incluir (0=lunes, ..., 6=domingo).
    horas (list): Lista de horas del día a incluir (0-23).
    timestamp_inicio (str o pd.Timestamp): Momento inicial desde el cual cortar la secuencia.
    pasos (int): Cantidad de pasos (filas) a incluir en la secuencia resultante.

    Retorna:
    np.ndarray: Array de shape (1, pasos, features) o None si ocurre un error.
    """
    try:
        encabezados_potencias = pd.read_csv(archivo_potencias, nrows=0).columns
        if 'timestamp;;;;' in encabezados_potencias:
            encabezados_potencias = encabezados_potencias.str.replace('timestamp;;;;', 'timestamp')

        fila_inicio = 1  # Ajustar si el archivo tiene encabezados adicionales
        numero_filas = 120000
        potencias = pd.read_csv(archivo_potencias, skiprows=fila_inicio, nrows=numero_filas,
                                 header=None, names=encabezados_potencias)

        potencias['timestamp'] = pd.to_datetime(potencias['timestamp'], errors='coerce')
        if potencias['timestamp'].isnull().any():
            raise ValueError("Fechas no válidas encontradas en el archivo.")

        if dias_semanales is not None:
            potencias = potencias[potencias['timestamp'].dt.dayofweek.isin(dias_semanales)]
        if horas is not None:
            potencias = potencias[potencias['timestamp'].dt.hour.isin(horas)]

        potencias = potencias.groupby(potencias['timestamp'].dt.floor('h')).first()
        potencias['tiempo_del_dia'] = potencias['timestamp'].dt.hour + potencias['timestamp'].dt.minute / 60.0
        potencias['dia_sen'] = np.sin(2 * np.pi * potencias['tiempo_del_dia'] / 24)
        potencias['dia_cos'] = np.cos(2 * np.pi * potencias['tiempo_del_dia'] / 24)

        potencias['dia_del_año'] = potencias['timestamp'].dt.dayofyear
        dias_en_el_anio = potencias['timestamp'].dt.is_leap_year.map(lambda x: 366 if x else 365)
        potencias['mes_sen'] = np.sin(2 * np.pi * potencias['dia_del_año'] / dias_en_el_anio)
        potencias['mes_cos'] = np.cos(2 * np.pi * potencias['dia_del_año'] / dias_en_el_anio)

        potencias['diferencia_activa'] = potencias['activa'].diff()
        potencias['pico'] = np.where(np.abs(potencias['diferencia_activa']) > 5, 1, 0)

        potencias = potencias.dropna().reset_index(drop=True)

        final_df = potencias[['activa', 'pico', 'dia_sen', 'dia_cos', 'mes_sen', 'mes_cos', 'timestamp']]

        # Si se proporciona timestamp_inicio, cortamos la secuencia
        if timestamp_inicio is not None:
            timestamp_inicio = pd.to_datetime(timestamp_inicio)
            inicio_idx = final_df[final_df['timestamp'] >= timestamp_inicio].index.min()
            if pd.isna(inicio_idx) or inicio_idx + pasos > len(final_df):
                raise ValueError("No hay suficientes datos desde el timestamp inicial.")
        else:
            inicio_idx = 0

        df_slice = final_df.iloc[inicio_idx:inicio_idx + pasos]
        if len(df_slice) < pasos:
            raise ValueError("La secuencia seleccionada no tiene suficientes pasos.")

        features = df_slice.drop(columns=['timestamp']).values
        X = np.expand_dims(features, axis=0)  # (1, pasos, features)
        return X

    except Exception as e:
        print(f"Error al cargar datos específicos: {e}")
        return None
