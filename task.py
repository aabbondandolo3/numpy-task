import numpy as np

# Controlla il file readme.md per i dettagli su ciascun sub-task

def prodotto_scalare(v1: list, v2: list) -> float:
    """Sub-task 1: Prodotto Scalare."""

    a = np.array(v1)
    b = np.array(v2)


    if a.shape != b.shape:
        raise ValueError("I vettori devono avere la stessa dimensione")


    return float(np.dot(a, b))

def rango_matrice(m: list) -> int:
    """Sub-task 2: Calcola il rango di una matrice."""

    matrice = np.array(m)

    if matrice.ndim != 2:
        raise ValueError("L'input deve essere una matrice (lista di liste)")


    return int(np.linalg.matrix_rank(matrice))



def risolvi_sistema_lineare(A: list, b: list) -> np.ndarray:
    """Sub-task 3: Risolvere un Sistema Lineare."""
    matrice = np.array(A)
    termini_noti = np.array(b)


    if matrice.ndim != 2:
        raise ValueError("A deve essere una lista di liste (matrice)")


    if termini_noti.ndim != 1:
        raise ValueError("b deve essere una lista (vettore)")


    if matrice.shape[0] != matrice.shape[1]:
            raise ValueError("La matrice A deve essere quadrata")


    if matrice.shape[0] != termini_noti.shape[0]:
            raise ValueError("Dimensioni incompatibili tra A e b")


    soluzione = np.linalg.solve(matrice, termini_noti)

    return soluzione




def correlazione_matrici(m1: list, m2: list) -> np.ndarray:
    """Sub-task 4: Correlazione tra Matrici 2x2."""
    a = np.array(m1)
    b = np.array(m2)

    if a.shape != (2, 2) or b.shape != (2, 2):
        raise ValueError("Entrambe le matrici devono essere 2x2")

    a_flat = a.flatten()
    b_flat = b.flatten()

    corr_matrix = np.corrcoef(a_flat, b_flat)

    return corr_matrix





def operazioni_elemento_per_elemento(v1: list) -> tuple:
    """Sub-task 5: Restituisce (seno, coseno, arcoseno, arcocoseno) elemento per elemento calcolati sul primo array."""
    v = np.array(v1)

    seno = np.sin(v)
    coseno = np.cos(v)
    arcoseno = np.arcsin(v)
    arcocoseno = np.arccos(v)

    return (seno, coseno, arcoseno, arcocoseno)


def main():
    print("Sub-task 1:", prodotto_scalare([1, 2, 3], [4, 5, 6]))
    print("Sub-task 1:", rango_matrice([[1, 2], [3, 4]]))
    print("Sub-task 3:", risolvi_sistema_lineare([[2, 1], [1, 3]], [5, 7]))
    print("Sub-task 4:", correlazione_matrici([[1, 2], [3, 4]], [[2, 4], [6, 8]]))
    print("Sub-task 5:", operazioni_elemento_per_elemento([0, 0.5, 1, -0.5]))





if __name__ == "__main__":
    main()
