from functools import lru_cache
import sympy as sp
import numpy as np
import dionysus as d
import math


def hashlize(numpy_ndarray):
    return frozenset(tuple(a) for a in numpy_ndarray)
                
@lru_cache(maxsize = None)
class Circumscribed_circle:
    def __init__(self, hashlized_Points):
        self.center = np.array([])
        self.radius = 0.0
        
        Points = np.array([p for p in hashlized_Points], dtype = float)
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
    
    
def complex_condition(Points, center, radius, Maxradius):
    if len(center):
        vector = list(dist_vector(Points, center))
        vector.append(Maxradius)
        return math.isclose(np.min(vector), radius)
    else:
        return False
        
                
def fill_2D_alpha(Points, Maxradius):
    '''
    This function compute 2-skelton of the alpha complex built on the given Points, which difinition based on the way of spheres each centers are given Points.
    The argument Maxradius is the maximum value of the filtration parameter but I recommend to avoid to use huge radius as argument even if you would like to compute on huge radius,becouse of the long computing time.
    '''
    Simplexes = []
    rips = d.fill_rips(Points, 2, 2.0 * Maxradius)
    for simplex in rips:
        Index = list(simplex)
        if simplex.dimension() <= 1:
            new_simplex = d.Simplex(Index, simplex.data / 2.0)
            Simplexes.append(new_simplex)
        elif simplex.dimension() == 2:
            SubPoints = hashlize(Points[Index])
            circle = Circumscribed_circle(SubPoints)
            if complex_condition(Points, circle.center, circle.radius, Maxradius):
                new_simplex = d.Simplex(Index, circle.radius)
                Simplexes.append(new_simplex)
                
    print('--------create_alpha_filtration---------')
    f = d.Filtration(Simplexes)
    f.sort()
    return f

def test():
    '''
    this is test of creating alpha filtration
    '''
    points1 = np.random.random((50, 2))
    f = fill_2D_alpha(points1, 0.8)
    print(d.is_simplicial(f))
    for s in f:
        print(s)
    pf = d.homology_persistence(f)
    dgms = d.init_diagrams(pf,f)
    try:
        d.plot.plot_diagram(dgms[1], show = True)
        
    except ValueError:
        pass
        


