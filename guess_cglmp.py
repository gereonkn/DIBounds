import numpy as np
from math import sqrt, log2, log, pi, cos, sin
import ncpol2sdpa as ncp
from sympy.physics.quantum.dagger import Dagger
import mosek
import chaospy
import gc


def get_subs():
  
    subs = {}
    subs.update(ncp.projective_measurement_constraints(A, B))
    for a in ncp.flatten([A,B]):
        for c in ncp.flatten([C]):
            subs.update({a*c: c*a})
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
                for c2 in Cflat:
                    monos += [a*b*c1*c2]
    return monos

    


def compute_cglmp(SDP):
    def mod3z(a):
        # Works directly on 0,1,2 (Python's natural modulo 3)
        return a % 3

    def A_val(i, a):
        # Return the probability for party A, measurement i and outcome a.
        # For outcome index 2 (i.e. the third outcome), use the normalization constraint.
        if a == 2:
            return 1 - A[i][0] - A[i][1]
        else:
            return A[i][a]

    def B_val(i, a):
        if a == 2:
            return 1 - B[i][0] - B[i][1]
        else:
            return B[i][a]

    cglmp_expr = 0.0
    for a in [0, 1, 2]:
        # Positive terms:
        cglmp_expr += A_val(0, a) * B_val(0, a)               # P(A0 = B0)
        cglmp_expr += A_val(1, mod3z(a+1)) * B_val(0, a)       # P(B0 = A1+1) with shift modulo 3
        cglmp_expr += A_val(1, a) * B_val(1, a)                 # P(A1 = B1)
        cglmp_expr += A_val(0, a) * B_val(1, a)                 # P(A0 = B1)
        
        # Negative terms:
        cglmp_expr -= A_val(0, a) * B_val(0, mod3z(a - 1))       # P(A0 = B0-1)
        cglmp_expr -= A_val(1, a) * B_val(0, a)                 # P(A1 = B0)
        cglmp_expr -= A_val(1, a) * B_val(1, mod3z(a - 1))       # P(A1 = B1-1)
        cglmp_expr -= A_val(0, mod3z(a - 1)) * B_val(1, a)       # P(A0 = B1-1)
  
    chsh_expr = (A[0][0]*B[0][0] + (1-A[0][0])*(1-B[0][0]) + \
        A[0][0]*B[1][0] + (1-A[0][0])*(1-B[1][0]) + \
        A[1][0]*B[0][0] + (1-A[1][0])*(1-B[0][0]) + \
        A[1][0]*(1-B[1][0]) + (1-A[1][0])*B[1][0])/4.0
    cglmp_expr = - cglmp_expr
    SDP.set_objective(cglmp_expr)
    SDP.solve('mosek')
        
    return SDP.primal

def score_constraints_cglmp(score):

    def mod3z(a):
        # Works directly on 0,1,2 (Python's natural modulo 3)
        return a % 3

    def A_val(i, a):
        # Return the probability for party A, measurement i and outcome a.
        # For outcome index 2 (i.e. the third outcome), use the normalization constraint.
        if a == 2:
            return 1 - A[i][0] - A[i][1]
        else:
            return A[i][a]

    def B_val(i, a):
        if a == 2:
            return 1 - B[i][0] - B[i][1]
        else:
            return B[i][a]

    cglmp_expr = 0.0
    for a in [0, 1, 2]:
        # Positive terms:
        cglmp_expr += A_val(0, a) * B_val(0, a)               # P(A0 = B0)
        cglmp_expr += A_val(1, mod3z(a+1)) * B_val(0, a)       # P(B0 = A1+1) with shift modulo 3
        cglmp_expr += A_val(1, a) * B_val(1, a)                 # P(A1 = B1)
        cglmp_expr += A_val(0, a) * B_val(1, a)                 # P(A0 = B1)
        
        # Negative terms:
        cglmp_expr -= A_val(0, a) * B_val(0, mod3z(a - 1))       # P(A0 = B0-1)
        cglmp_expr -= A_val(1, a) * B_val(0, a)                 # P(A1 = B0)
        cglmp_expr -= A_val(1, a) * B_val(1, mod3z(a - 1))       # P(A1 = B1-1)
        cglmp_expr -= A_val(0, mod3z(a - 1)) * B_val(1, a)       # P(A0 = B1-1)

    #cglmp_expr = - cglmp_expr

    return [cglmp_expr - score]










def score_constraints(score):
    """
    Returns CHSH score constraint
    """
    chsh_expr = (A[0][0]*B[0][0] + (1-A[0][0])*(1-B[0][0]) + \
        A[0][0]*B[1][0] + (1-A[0][0])*(1-B[1][0]) + \
        A[1][0]*B[0][0] + (1-A[1][0])*(1-B[0][0]) + \
        A[1][0]*(1-B[1][0]) + (1-A[1][0])*B[1][0])/4

    return [chsh_expr - score]


def compute_guess(SDP):

    obj = 0


    F = [A[0][0],A[0][1],1-A[0][0]-A[0][1]]
    G = [C[0],C[1],1-C[0]-C[1]]
   
    for a in range(len(F)):
        obj -= F[a]*G[a]

    SDP.set_objective(obj)
    SDP.solve('mosek')
        
    

    return SDP.primal

LEVEL = 3
VERBOSE = 1

A_config = [3, 3]
B_config = [3, 3]


A = [Ai for Ai in ncp.generate_measurements(A_config, 'A')]
B = [Bj for Bj in ncp.generate_measurements(B_config, 'B')]
print(ncp.flatten([A,B]))
C = ncp.generate_operators('C', 2, hermitian=True)

substitutions = get_subs()
moment_ineqs = []    
moment_eqs = []       
op_eqs = []           
op_ineqs = [C[0],C[1],1-C[0],1-C[1],1-C[0]-C[1]]          

extra_monos = get_extra_monomials()

score_cons = score_constraints_cglmp(2.92)

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
for score in np.linspace(2,2.914854,40):
    score_cons =  score_constraints_cglmp(score)
    sdp.process_constraints(equalities=op_eqs[:],
                                inequalities=op_ineqs[:],
                                momentequalities=moment_eqs[:],
                                momentinequalities=moment_ineqs[:]+score_cons)
    ent = compute_guess(sdp)
    print(-log2(-ent))
    entropy = entropy + [ent]
    print(compute_cglmp(sdp),score)

np.savetxt('min_entropy_bounds_cglmp.txt',entropy)


"""
for score in np.linspace(0.5,0.85,10):
    score_cons =  score_constraints(score)
    sdp.process_constraints(equalities=op_eqs[:],
                                inequalities=op_ineqs[:],
                                momentequalities=moment_eqs[:],
                                momentinequalities=moment_ineqs[:]+score_cons)
    ent = compute_guess(sdp)
    print(-log(-ent))
"""