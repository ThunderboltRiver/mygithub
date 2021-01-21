from itertools import chain,combinations
from mygithub.persistence import *
import sympy as sp
import numpy as np

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
alpha = 1
points = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
f = d.Filtration()
for i, subset in enumerate(powerset(points)):
    s = np.array(list(subset))
    print(i, s)
    if len(s) == 0:
    	continue
    dim = points.shape[1]
    s_indexes = []
    for point in s:
        point_index = np.argmax(np.sum(points == point, axis = 1))
        s_indexes.append(point_index)
    if len(s) == 1:
        f.append(d.Simplex(s_indexes, 0))
        continue
    A = s[:len(s) - 1] - s[-1]
    x = sp.symbols('x0:dim')
    b = (np.linalg.norm(s[:len(s) - 1], axis = 1)**2 - np.linalg.norm(s[-1])**2) / 2.0
    system = sp.Matrix(A), sp.Matrix(b)
    general_center = list(sp.linsolve(system, x))[0]
    if len(general_center):
        center = np.array([])
        for condition in general_center:
            for var in x:
                condition = condition.subs(var, 0)
            center = np.append(center, condition)
        print(f'{s[0]}-----{center}')
        radius = np.linalg.norm(s[0] - center)
        if radius <= alpha:
            f.append(d.Simplex(s_indexes, radius))
f.sort()
for s in f:
    print(s)
