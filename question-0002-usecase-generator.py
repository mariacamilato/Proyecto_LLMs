import numpy as np
from sklearn.decomposition import TruncatedSVD

def generar_caso_de_uso_calcular_error_reconstruccion_svd():
    """
    Genera casos de uso aleatorios para la función
    calcular_error_reconstruccion_svd(X, n_componentes).

    Returns:
        input  (dict): diccionario con claves 'X' y 'n_componentes'
        output (numpy.ndarray): error cuadrático medio por fila,
                                shape (n_usuarios,)
    """
    # Dimensiones aleatorias
    n_usuarios = np.random.randint(50, 200)
    n_items    = np.random.randint(20, 80)

    # n_componentes debe ser < min(n_usuarios, n_items)
    max_comp    = min(n_usuarios, n_items) - 1
    n_componentes = np.random.randint(2, min(max_comp, 15) + 1)

    # Matriz X con estructura latente realista:
    # se genera como producto de factores latentes + ruido
    n_factores_reales = max(n_componentes, np.random.randint(3, 8))
    factores_usuario  = np.random.randn(n_usuarios, n_factores_reales)
    factores_item     = np.random.randn(n_factores_reales, n_items)
    ruido             = np.random.normal(0, 0.3, size=(n_usuarios, n_items))
    X = factores_usuario @ factores_item + ruido

    # ── Lógica que debe replicar calcular_error_reconstruccion_svd ──────────
    svd         = TruncatedSVD(n_components=n_componentes, random_state=42)
    X_reducido  = svd.fit_transform(X)          # (n_usuarios, n_componentes)
    X_reconstruido = svd.inverse_transform(X_reducido)  # (n_usuarios, n_items)

    # Error cuadrático medio por fila
    error_por_fila = np.mean((X - X_reconstruido) ** 2, axis=1)
    # ────────────────────────────────────────────────────────────────────────

    input_  = {"X": X, "n_componentes": n_componentes}
    output  = error_por_fila    # numpy.ndarray shape (n_usuarios,)

    return input_, output


# ── Ejemplo de uso ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    inp, out = generar_caso_de_uso_calcular_error_reconstruccion_svd()

    print("Forma de X:          ", inp["X"].shape)
    print("n_componentes:       ", inp["n_componentes"])
    print("Shape salida:        ", out.shape)
    print("Error mín / máx:     ", round(out.min(), 6), "/", round(out.max(), 6))
    print("Error medio global:  ", round(out.mean(), 6))
    print("Primeros 5 errores:  ", np.round(out[:5], 6))