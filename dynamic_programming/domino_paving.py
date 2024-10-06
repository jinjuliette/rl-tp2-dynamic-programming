# Exercice 3 : pavage d'un rectangle avec des dominos
# ---------------------------------------------------
# On considère un rectangle de dimensions 3xN, et des dominos de
# dimensions 2x1. On souhaite calculer le nombre de façons de paver le
# rectangle avec des dominos.

# Ecrire une fonction qui calcule le nombre de façons de paver le
# rectangle de dimensions 3xN avec des dominos.
# Indice: trouver une relation de récurrence entre le nombre de façons
# de paver un rectangle de dimensions 3xN et le nombre de façons de
# paver un rectangle de dimensions 3x(N-1), 3x(N-2) et 3x(N-3).


def domino_paving(n: int) -> int:
    """
    Calcule le nombre de façons de paver un rectangle de dimensions 3xN
    avec des dominos.
    """
    # BEGIN SOLUTION
    if (3 * n) % 2 != 0:
        return 0

    d_n4, d_n2, d_n6 = 1, 3, 1
    for i in range(n // 2 - 1):
        d_n4, d_n2, d_n6 = d_n2, 3 * (d_n4 + d_n2) - d_n6, d_n4
    return d_n2
    # END SOLUTION
