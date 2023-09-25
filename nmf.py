import numpy as np
from sklearn.decomposition import NMF

# a matrisini oluşturun
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# NMF modelini oluşturun
model = NMF(n_components=2, init='random', random_state=0)

# a matrisini w ve h matrislerine ayırın
w = model.fit_transform(a)
h = model.components_

print("w matrisi:")
print(w)
print("h matrisi:")
print(h)