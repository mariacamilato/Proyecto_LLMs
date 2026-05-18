import pandas as pd

def eficiencia_promedio_alta_radiacion(df_solar):
    filtro = df_solar[df_solar['radiacion_w_m2'] > 800]
    if filtro.empty:
        return 0.0
    return float(filtro['eficiencia_porcentaje'].mean())

