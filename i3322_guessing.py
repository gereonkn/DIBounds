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
    
    return monos

    


def compute_i3322(SDP):

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

    E_expr = -E_expr
    # Set the objective for the SDP.
    SDP.set_objective(E_expr)
    
    # Solve the SDP using the 'mosek' solver.
    SDP.solve('mosek')
    
    return SDP.primal






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






def compute_guess(SDP):

    obj = 0


    F = [A[0][0],1-A[0][0]]
    G = [C[0],1-C[0]]
   
    for a in range(len(F)):
        obj -= F[a]*G[a]

    SDP.set_objective(obj)
    SDP.solve('mosek')
        
    

    return SDP.primal

LEVEL = 4
VERBOSE = 1

A_config = [2, 2, 2]
B_config = [2, 2, 2]


A = [Ai for Ai in ncp.generate_measurements(A_config, 'A')]
B = [Bj for Bj in ncp.generate_measurements(B_config, 'B')]
print(ncp.flatten([A,B]))
C = ncp.generate_operators('C', 1, hermitian=True)

substitutions = get_subs()
moment_ineqs = []    
moment_eqs = []       
op_eqs = []           
op_ineqs = [C[0],1-C[0]]          

extra_monos = get_extra_monomials()

score_cons = score_constraints_i3322(5.0035)

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
for score in np.linspace(4.7805,5.0035,10):
    score_cons =  score_constraints_i3322(score)
    sdp.process_constraints(equalities=op_eqs[:],
                               inequalities=op_ineqs[:],
                                momentequalities=moment_eqs[:],
                                momentinequalities=moment_ineqs[:]+score_cons)
    ent = compute_guess(sdp)
    print(-log2(-ent))
    entropy = entropy + [-log2(-ent)]
    #print(compute_i3322(sdp),score)

np.savetxt('i3322_guessing.txt',np.transpose(np.array([np.linspace(4.7805,5.0035,10),entropy])))