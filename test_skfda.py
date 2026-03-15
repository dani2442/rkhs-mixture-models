import numpy as np
import skfda
from skfda.representation.basis import BSplineBasis

def test_fda():
    X = np.random.randn(10, 5) # 10 samples, 5 base coeffs
    basis = BSplineBasis(n_basis=5)
    fd = skfda.FDataBasis(basis, X)
    fd_grid = fd.to_grid(np.linspace(0, 1, 100))

    try:
        kmeans = skfda.ml.clustering.KMeans(n_clusters=2)
        kmeans.fit(fd_grid)
        print("KMeans works with FDataGrid!")
    except Exception as e:
        print("KMeans failed with FDataGrid:", e)

if __name__ == "__main__":
    test_fda()
