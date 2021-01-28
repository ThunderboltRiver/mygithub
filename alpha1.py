from itertools import chain,combinations
import sympy as sp
import numpy as np
import dionysus as d
import math
import os


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
        s = np.array(s)
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
                    
                
                self.radius = np.linalg.norm(Point - self.center)
                
def dist_vector(Points, point):
    return np.linalg.norm(Points - point, axis = 1)
    

def get_index(Points, point):
    '''
    if point in Points
    '''
    norms = dist_vector(Points, point)
    for i, dist in enumerate(norms):
        if math.isclose(dist, 0):
            return i
    
def complex_condition(Points, center, radius, Maxradius):
    if len(center):
        vector = list(dist_vector(Points, center))
        vector.append(Maxradius)
        return math.isclose(np.min(vector), radius)
    else:
        return False
        
                
def fill_2D_alpha(Points, Maxradius):
    Simplexes = []
    rips = d.fill_rips(Points, 2, 2 * Maxradius)
    for simplex in rips:
        if simplex.dimension() == 1:
            simplex.data /= 2.0
        Simplexes.append(simplex)
        if simplex.dimension() == 2:
            SubPoints = Points[list(simplex)]
            circle = Circumscribed_circle(SubPoints)
            if complex_condition(Points, circle.center, circle.radius, Maxradius):
                simplex.data = circle.radius
                Simplexes.append(simplex)
                
    print('--------create_alpha_filtration---------')
    f = d.Filtration(Simplexes)
    f.sort()
    return f

def test():
    points1 = np.array([[1.0,0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],[1.0, 1.0],[-1.0, -1.0]])
    l = [3, 2, 1, 0]
    points2 = np.array([[1.0,0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -2.0]])
    f = fill_2D_alpha(points1, 1.0)
    for s in f:
        print(f'f:{s}')
        
    print(d.is_simplicial(f))
    g = fill_2D_alpha(points2, 1.0)
    for s in g:
        print(f'f:{s}')
    print(d.is_simplicial(g))
        


