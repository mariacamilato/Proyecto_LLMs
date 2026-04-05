import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def generar_caso_de_uso_varianza_acumulada_hasta_umbral():
    """
    Genera casos de uso aleatorios para la función
    varianza_acumulada_hasta_umbral(X, umbral=0.85).

    Returns:
        input  (dict): diccionario con claves 'X' y 'umbral'
        output (int): número mínimo de componentes PCA necesarios
                      para alcanzar o superar el umbral de varianza acumulada
    """
    # Dimensiones aleatorias
    n_samples  = np.random.randint(100, 400)
    n_features = np.random.randint(5, 25)

    # Umbral aleatorio entre 0.70 y 0.99
    umbral = round(np.random.uniform(0.70, 0.99), 2)

    # Generar X con estructura de correlación realista:
    # se construye como combinación lineal de pocos factores latentes
    # para que PCA encuentre componentes con varianza decreciente natural
    n_factores = np.random.randint(2, min(n_features, 8) + 1)
    factores   = np.random.randn(n_samples, n_factores)
    pesos      = np.random.randn(n_factores, n_features)
    ruido      = np.random.normal(0, np.random.uniform(0.1, 0.5),
                                  size=(n_samples, n_features))
    X = factores @ pesos + ruido

    # ── Lógica que debe replicar varianza_acumulada_hasta_umbral ────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_scaled)

    varianza_acumulada = np.cumsum(pca.explained_variance_ratio_)

    # Número mínimo de componentes para alcanzar o superar el umbral
    n_componentes = int(np.argmax(varianza_acumulada >= umbral) + 1)
    # ────────────────────────────────────────────────────────────────────────

    input_  = {"X": X, "umbral": umbral}
    output  = n_componentes     # int

    return input_, output


# ── Ejemplo de uso ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    inp, out = generar_caso_de_uso_varianza_acumulada_hasta_umbral()

    print("Forma de X:             ", inp["X"].shape)
    print("Umbral:                 ", inp["umbral"])
    print("Componentes necesarios: ", out)
    print("Tipo de salida:         ", type(out))

    # Verificación visual: mostrar varianza acumulada de los primeros
    # componentes para confirmar que 'out' es correcto
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(inp["X"])
    pca = PCA()
    pca.fit(X_scaled)
    var_acum = np.cumsum(pca.explained_variance_ratio_)
    print(f"\nVarianza acumulada en los primeros {out} componentes: "
          f"{round(var_acum[out - 1], 6)}")
    print(f"Varianza acumulada en {out - 1} componentes:          "
          f"{round(var_acum[out - 2], 6) if out > 1 else 0.0}")
    print(f"Umbral exigido:                                    "
          f"{inp['umbral']}")
