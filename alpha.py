from itertools import chain,combinations
import sympy as sp
import numpy as np
import dionysus as d

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

class Circumscribed_circle:
    def __init__(self, Points):
        self.Points = Points
        self.center = np.array([])
        self.radius = None
        
        if len(Points)==1:
            self.center = Points
            self.radius = 0.0
            
        else:
            A = Points[:len(Points) - 1] - Points[-1]
            dim = Points.shape[1]
            x = sp.symbols([f'x{i}' for i in range(dim)])
            b = (np.linalg.norm(Points[:len(Points) - 1], axis = 1)**2 - np.linalg.norm(Points[-1])**2) / 2.0
            system = sp.Matrix(A), sp.Matrix(b)
            general_center = sp.Matrix(list(sp.linsolve(system, x))[0])
            print(general_center)
            
            if len(general_center):
                grad = (sp.Matrix(Points[0]) - general_center).T * general_center.jacobian(x)
                print(f'{grad =}')
                norm_grad = grad.norm()
                
                if norm_grad:
                    CenterSolution = sp.solve(grad, x)
                    print(f'{CenterSolution= }')
                    for condition in general_center:
                        for key, value in CenterSolution.items():
                            condition = condition.subs(key, value)
                        self.center = np.append(self.center, condition)
                    
                else:
                    self.center = np.array(general_center).flatten()
                    
                print(f'{self.center=}')
                Point = Points[0]
                print(f'{Point=}, info:{Point.dtype}')
                print(f'{Point - self.center=}')
                self.radius = float(np.linalg.norm(Point - self.center.astype(np.float64)))


def Delaunay_Complex(Points):
    f = d.Filtration()
    dim = Points.shape[1]
    for SubSet in powerset(Points):
        if len(SubSet):
            print(len(SubSet))
            SubPoints = np.array(SubSet)
            circle = Circumscribed_circle(SubPoints)
            if len(circle.center):
                SubIndexes = np.array([])
                for childpoint in SubPoints:
                    for i, parepoint in enumerate(Points):
                        if np.sum(childpoint == parepoint) == dim:
                            SubIndexes = np.append(SubIndexes, i)
                SubIndexes = SubIndexes.astype(np.int64)
                print(f'{circle.radius=}')
                Simplex = d.Simplex(SubIndexes, circle.radius)
                f.append(Simplex)
    
    return f

points = np.array([[1.0,0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
f = Delaunay_Complex(points)
for s in f:
    print(s)


