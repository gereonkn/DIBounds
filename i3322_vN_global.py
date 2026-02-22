import numpy as np
from math import sqrt, log2, log, pi, cos, sin
import ncpol2sdpa as ncp
from sympy.physics.quantum.dagger import Dagger
import mosek
import chaospy
import gc

print(np.linspace(4.5,5.0035,7))


def lists_gb_inner(grid):
    gamma = np.zeros(len(grid)+1)
    beta = np.zeros(len(grid)+1)
    beta[0] = -1
    gamma[0] = grid[0]
    for i in range(len(grid)):
        if i == 0:
            constante = (1 + grid[i] / (grid[i+1] - grid[i])) * np.log(grid[i+1] / grid[i]) - 1
            gamma[i+1] = grid[i] * constante
            beta[i+1] = -constante
        elif i == len(grid) - 1:
            constante = 1 - grid[i-1] / (grid[i] - grid[i-1]) * np.log(grid[i] / grid[i-1])
            gamma[i+1] = grid[i] * constante
            beta[i+1] = -constante
        else:
            constante = (1 + grid[i] / (grid[i+1] - grid[i])) * np.log(grid[i+1] / grid[i]) - grid[i-1] / (grid[i] - grid[i-1]) * np.log(grid[i] / grid[i-1])
            gamma[i+1] = constante * grid[i]
            beta[i+1] = -constante

    return gamma, beta




def test_integration_formula(gamma,beta):
    return sum(gamma)




def grid_function(c, epsilon, mu, lamb):
    grid = [mu]
    f = mu
    i = 0
    while f < lamb:
        f = grid[i] + np.sqrt(grid[i] * epsilon / c)
        grid.append(f)
        i += 1
    grid[-1] = lamb  # Ensure the last element is exactly lamb
    return grid

grid = grid_function(1,0.5,0.1,1)
gamma, beta = lists_gb_inner(grid)

print(len(gamma))
print(test_integration_formula(gamma,beta))










def get_subs():
  
    subs = {}
    subs.update(ncp.projective_measurement_constraints(A, B))
    for a in ncp.flatten([A,B]):
        for c in ncp.flatten([C]):
            subs.update({a*c: c*a})
    for c in ncp.flatten(C):
        subs.update({c*c: c})
    return subs

def get_extra_monomials():

    monos = []
   
    

    monos = []
    Aflat = ncp.flatten(A)
    Bflat = ncp.flatten(B)
    Cflat = ncp.flatten(C)
    for a in Aflat:
        for b in Bflat:
            for c1 in Cflat:
                monos += [a*b*c1]
    return monos

    




def score_constraints_i3322(score):
    def A_op(i):
     
        return 2 * A[i][0] - 1

    def B_op(i):

        return 2 * B[i][0] - 1

    E_expr = (
          A_op(0) * B_op(2)   # 〈A1B3〉
        + A_op(1) * B_op(2)   # 〈A2B3〉
        + A_op(2) * B_op(0)   # 〈A3B1〉
        + A_op(2) * B_op(1)   # 〈A3B2〉
        + A_op(0)           # 〈A1〉
        - A_op(1)           # -〈A2〉
        + B_op(0)           # 〈B1〉
        - B_op(1)           # -〈B2〉
        - A_op(0) * B_op(0)   # -〈A1B1〉
        + A_op(0) * B_op(1)   # 〈A1B2〉
        + A_op(1) * B_op(0)   # 〈A2B1〉
        - A_op(1) * B_op(1)   # -〈A2B2〉
    )
    return [E_expr - score]








def compute_vN(SDP):

    ent_value = 0

    obj = 0
    i = 0
    F = [A[0][0],1-A[0][0]]
    for time_step in range(len(gamma)):
        for a in range(len(F)):
            obj += -beta[time_step]*F[a]*C[i] - gamma[time_step]*C[i]
            i = i+1
    SDP.set_objective(obj)
    SDP.solve('mosek')
    
    return SDP.primal

LEVEL = 2
VERBOSE = 1

A_config = [2, 2, 2]
B_config = [2, 2, 2]


A = [Ai for Ai in ncp.generate_measurements(A_config, 'A')]
B = [Bj for Bj in ncp.generate_measurements(B_config, 'B')]
print(ncp.flatten([A,B]))
C = ncp.generate_operators('C', 2*len(gamma), hermitian=True)

substitutions = get_subs()
moment_ineqs = []    
moment_eqs = []       
op_eqs = []           
op_ineqs = [C[i] for i in range(2*len(gamma))]          

extra_monos = get_extra_monomials()

score_cons = score_constraints_i3322(5.035)

obj = 0.0

ops = ncp.flatten([A, B, C])
sdp = ncp.SdpRelaxation(ops, verbose=VERBOSE - 1, normalized=True)
sdp.get_relaxation(level=LEVEL,
                    equalities=op_eqs[:],
                    inequalities=op_ineqs[:],
                    momentequalities=moment_eqs[:],
                    momentinequalities=moment_ineqs[:] + score_cons,
                    objective=obj,
                    substitutions=substitutions,
                    extramonomials=extra_monos)

print(C)


entropy = []
for score in np.linspace(4.5,5.0035,7):
    score_cons =  score_constraints_i3322(score)
    sdp.process_constraints(equalities=op_eqs[:],
                                inequalities=op_ineqs[:],
                                momentequalities=moment_eqs[:],
                                momentinequalities=moment_ineqs[:]+score_cons)
    ent = compute_vN(sdp)
    print(1/np.log(2)*(1+ent),ent,1/np.log(2))
    entropy = entropy + [1/np.log(2)*(1+ent)]


#np.savetxt('i3322_vN.txt',np.transpose(np.array([np.linspace(4,5.0035,20),entropy])))