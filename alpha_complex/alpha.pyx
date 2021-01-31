from functools import lru_cache
import sympy as sp
import numpy as np
import dionysus as d
import math
import shelve
import zipfile

def hashlize(numpy_ndarray):
    return frozenset(tuple(a) for a in numpy_ndarray)
                

def Circumscribed_circle(hashlized_Points):
    center = np.array([])
    radius = 0.0
    center_condition = []
    Points = np.array([p for p in hashlized_Points], dtype = float)
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
                center_condition.append(float(condition))
            
        else:
            center_condition = list(GeneralCenter)
            
        center = np.array(center_condition, dtype = float).flatten()
        radius = np.linalg.norm(Point - center)
        
    return center, radius
                    

def dist_vector(Points, point):
    return np.linalg.norm(Points - point, axis = 1)
    
    
def complex_condition(Points, center, radius, Maxradius):
    if type(center) != type(None) and len(center):
        vector = list(dist_vector(Points, center))
        vector.append(Maxradius)
        return math.isclose(np.min(vector), radius)
        
    else:
        return False

                
def fill_2D_alpha(Points, Maxradius):

    '''
    This function compute 2-skelton of the alpha complex built on the given Points, which difinition based on the spheres method that each centers are given Points.
    The argument Maxradius is the maximum value of the filtration parameter.
    '''
    Simplexes = []
    with zipfile.ZipFile('CenterandRadius.zip') as zip:
        name = zip.namelist()[0]
        with shelve.open(name) as CR:
            rips = d.fill_rips(Points, 2, 2.0 * Maxradius)
            for simplex in rips:
                Index = list(simplex)
                if simplex.dimension() <= 1:
                    new_simplex = d.Simplex(Index, simplex.data / 2.0)
                    Simplexes.append(new_simplex)
                elif simplex.dimension() == 2:
                    SubPoints = hashlize(Points[Index])
                    try:
                        center, radius = CR[f'{hash(SubPoints)}']
                        
                    except KeyError:
                        center, radius = Circumscribed_circle(SubPoints)
                        CR[f'{hash(SubPoints)}'] = (center, radius)
                        
                    if complex_condition(Points, center, radius, Maxradius):
                        new_simplex = d.Simplex(Index, radius)
                        Simplexes.append(new_simplex)
                            
    print('--------create_alpha_filtration---------')
    f = d.Filtration(Simplexes)
    f.sort()
    return f, Simplexes

def test():
    '''
    this is test of creating alpha filtration
    '''
    kihus = np.array([[i, j] for i in range(20) for j in range(20)], dtype = float)
    index = np.random.choice(np.arange(len(kihus)), size = 50, replace = False)
    random_kihu = kihus[index]
    f, Simplexes = fill_2D_alpha(random_kihu, float('inf'))
    pf = d.homology_persistence(f)
    dgms = d.init_diagrams(pf,f)
    try:
        d.plot.plot_diagram(dgms[1], show = True)
        
    except ValueError:
        pass
        


