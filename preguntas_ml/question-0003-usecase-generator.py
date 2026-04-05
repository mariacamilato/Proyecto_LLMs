import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def generar_caso_de_uso_calcular_compacidad_clusters():
    """
    Genera casos de uso aleatorios para la función
    calcular_compacidad_clusters(X, n_clusters, random_state=42).

    Returns:
        input  (dict): diccionario con claves 'X', 'n_clusters' y 'random_state'
        output (float): distancia euclidiana media de cada muestra
                        a su centroide asignado
    """
    # Dimensiones aleatorias
    n_clusters  = np.random.randint(2, 7)          # entre 2 y 6 clusters
    n_features  = np.random.randint(2, 10)          # entre 2 y 9 features
    # Al menos 10 muestras por cluster para que KMeans sea estable
    n_samples   = np.random.randint(n_clusters * 10, n_clusters * 40)

    # Generar X con estructura de clusters reales:
    # cada cluster tiene su propio centro y dispersión aleatoria
    centros     = np.random.uniform(-5, 5, size=(n_clusters, n_features))
    dispersion  = np.random.uniform(0.3, 1.5)

    muestras_por_cluster = np.array_split(
        np.arange(n_samples), n_clusters
    )
    X_lista = []
    for i, idx in enumerate(muestras_por_cluster):
        n_i   = len(idx)
        bloque = centros[i] + np.random.randn(n_i, n_features) * dispersion
        X_lista.append(bloque)

    X = np.vstack(X_lista)
    # Mezclar filas para que no queden ordenadas por cluster
    perm = np.random.permutation(n_samples)
    X    = X[perm]

    # random_state aleatorio para el caso de uso (el estudiante debe
    # respetar el que reciba como argumento)
    random_state = np.random.randint(0, 100)

    # ── Lógica que debe replicar calcular_compacidad_clusters ───────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans   = KMeans(n_clusters=n_clusters, n_init=10,
                      random_state=random_state)
    kmeans.fit(X_scaled)

    etiquetas  = kmeans.labels_                        # (n_samples,)
    centroides = kmeans.cluster_centers_               # (n_clusters, n_features)

    # Distancia euclidiana de cada muestra a su centroide asignado
    distancias = np.linalg.norm(
        X_scaled - centroides[etiquetas], axis=1
    )
    compacidad = float(np.mean(distancias))
    # ────────────────────────────────────────────────────────────────────────

    input_  = {"X": X, "n_clusters": n_clusters, "random_state": random_state}
    output  = compacidad    # float

    return input_, output


# ── Ejemplo de uso ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    inp, out = generar_caso_de_uso_calcular_compacidad_clusters()

    print("Forma de X:      ", inp["X"].shape)
    print("n_clusters:      ", inp["n_clusters"])
    print("random_state:    ", inp["random_state"])
    print("Compacidad:      ", round(out, 6))
    print("Tipo de salida:  ", type(out))
