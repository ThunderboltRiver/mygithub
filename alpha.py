from itertools import chain,combinations
import sympy as sp
import numpy as np
import dionysus as d
import os

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

class Circumscribed_circle:
    def __init__(self, Points):
        self.Points = Points
        self.center = np.array([])
        self.radius = 0.0
        
        if len(Points)==1:
            self.center = Points
            
        else:
            CircumMatrix = Points[:len(Points) - 1] - Points[-1]
            dim = Points.shape[1]
            x = sp.symbols([f'x{i}' for i in range(dim)])
            CircumVector = (np.linalg.norm(Points[:len(Points) - 1], axis = 1)**2 - np.linalg.norm(Points[-1])**2) / 2.0
            system = sp.Matrix(CircumMatrix), sp.Matrix(CircumVector)
            GeneralCenter = sp.linsolve(system, x)
            if len(GeneralCenter):
                GeneralCenter = sp.Matrix(list(GeneralCenter)[0])
                Point = Points[0]
                grad_GeneralCenter = (sp.Matrix(Point) - GeneralCenter).T * GeneralCenter.jacobian(x)
                
                if grad_GeneralCenter.norm():
                    CenterSolution = sp.solve(grad_GeneralCenter, x)
                    for condition in GeneralCenter:
                        for key, value in CenterSolution.items():
                            condition = condition.subs(key, value)
                        self.center = np.append(self.center, float(condition))
                    
                else:
                    self.center = np.array(GeneralCenter, dtype = float).flatten()
                    
                
                self.radius = float(np.linalg.norm(Point - self.center))


def fill_alpha(Points, Maxdim = 2, Maxradius = float('inf')):
    f = d.Filtration()
    Pdim = Points.shape[1]
    for SubSet in powerset(Points):
        if 0 < len(SubSet) <= Maxdim + 1:
            SubPoints = np.array(SubSet)
            circle = Circumscribed_circle(SubPoints)
            if len(circle.center) and circle.radius <= Maxradius:
                SubIndexes = []
                for childpoint in SubPoints:
                    for i, parepoint in enumerate(Points):
                        if np.sum(childpoint == parepoint) == Pdim:
                            SubIndexes.append(i)
                            break
                Simplex = d.Simplex(SubIndexes, circle.radius)
                f.append(Simplex)
        
        elif len(SubSet) > Maxdim + 1:
            f.sort()
            print('--------create_alpha_filtration---------')
            return f
            
def test():
    points = np.array([[1.0,0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    f = fill_alpha(points)
    for s in f:
        print(s)
        
    p = d.homology_persistence(f)
    dgms = d.init_diagrams(p, f)
    d.plot.plot_bars(dgms[0], show = True)
    



