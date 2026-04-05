import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

def generar_caso_de_uso_calcular_ganancia_informacion():
    """
    Genera casos de uso aleatorios para la función calcular_ganancia_informacion(X, y).
    
    Returns:
        input  (dict): diccionario con claves 'X' e 'y'
        output (numpy.ndarray): ganancia de información por feature
    """
    # Dimensiones aleatorias
    n_samples  = np.random.randint(80, 300)
    n_features = np.random.randint(3, 12)

    # Matriz de features con distribución aleatoria (normal o uniforme)
    dist = np.random.choice(["normal", "uniform"])
    if dist == "normal":
        X = np.random.randn(n_samples, n_features)
    else:
        X = np.random.uniform(-3, 3, size=(n_samples, n_features))

    # Vector objetivo binario con proporción de clases aleatoria
    p_positivo = np.random.uniform(0.2, 0.8)
    y = np.random.binomial(1, p_positivo, size=n_samples)

    # ── Lógica que debe replicar calcular_ganancia_informacion ──────────────
    medianas = np.median(X, axis=0)
    X_bin    = (X > medianas).astype(int)

    ganancia = mutual_info_classif(X_bin, y, discrete_features=True,
                                   random_state=42)
    # ────────────────────────────────────────────────────────────────────────

    input_  = {"X": X, "y": y}
    output  = ganancia          # numpy.ndarray shape (n_features,)

    return input_, output


# ── Ejemplo de uso ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    inp, out = generar_caso_de_uso_calcular_ganancia_informacion()

    print("Forma de X:   ", inp["X"].shape)
    print("Forma de y:   ", inp["y"].shape)
    print("Clases en y:  ", np.unique(inp["y"], return_counts=True))
    print("Ganancia info:", np.round(out, 6))
    print("Shape salida: ", out.shape)
