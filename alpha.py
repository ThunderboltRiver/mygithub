from itertools import chain,combinations
import sympy as sp
import numpy as np
import dionysus as d
import os


PointsSet_cash = []
filtration_cash = []


def get_key_from_value(d, val):
    keys = [k for k, v in d.items() if v == val]
    if keys:
        return keys[0]
    return None

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    
def condi_powerset(iterable, func):
    for s in powerset(iterable):
        if func(s):
            yield s

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

def ndarray_to_set(ndarray):
    return set(tuple(p) for p in ndarray)

                
def have_intersection(Points1, Points2):
    set1 = set(tuple(p) for p in Points1)
    set2 = set(tuple(p) for p in Points2)
    return not(set1.isdisjoint(set2))
    
def ndarrays_difference(ndarray1, ndarray2):
    set1, set2 = ndarray_to_set(ndarray1), ndarray_to_set(ndarray2)
    return np.array([row for row in set1 - set2])
    

def fill_alpha(Points, Maxdim = 2, Maxradius = float('inf')):
    f = d.Filtration([])
    print(f)
    Pdim = Points.shape[1]
    PointsSet = ndarray_to_set(Points)
    PreviousPointsSet = set()
    
    if PointsSet_cash and (PointsSet_cash[-1] <= PointsSet):
        Previous_PointsSet = PointsSet_cash[-1]
        f = filtration_cash[-1]
        print('use filtration_cash')
        
    NewPointsSet = PointsSet - PreviousPointsSet
    
    for SubPointsSet in condi_powerset(Points, lambda SubPointsSet:have_intersection(SubPointsSet, NewPointsSet)):
        SubPoints = np.array(SubPointsSet)
        print(SubPoints)
        if 0 < len(SubPoints) <= Maxdim + 1:
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
                
                del childpoint, i, parepoint, SubIndexes, Simplex
            print('check')
            print(f)
            
        elif len(SubPoints) > Maxdim + 1:
            f.sort()
            print('--------create_alpha_filtration---------')
            PointsSet_cash.append(PointsSet)
            filtration_cash.append(f)
            return f
            
                    
def test():
    points = np.array([[1.0,0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    f = fill_alpha(points)
    for s in f:
        print(s)



