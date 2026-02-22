"""
Script to compute converging lower bounds on the rates of local randomness
extracted from two devices achieving a minimal expected CHSH score. More specifically,
computes a sequence of lower bounds on the problem

    inf H(A|X=0,E)

where the infimum is over all quantum devices achieving a CHSH score of w. See
the accompanying paper for more details (this script can be used to generate data for Figure 1).
"""
import numpy as np
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



grid =grid_function(1,0.08,0.1,1)
gamma, beta = lists_gb_inner(grid)

print(len(gamma))
print(test_integration_formula(gamma,beta))




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
        + A_op(0)             # 〈A1〉
        - A_op(1)             # -〈A2〉
        + B_op(0)             # 〈B1〉
        - B_op(1)             # -〈B2〉
        - A_op(0) * B_op(0)   # -〈A1B1〉
        + A_op(0) * B_op(1)   # 〈A1B2〉
        + A_op(1) * B_op(0)   # 〈A2B1〉
        - A_op(1) * B_op(1)   # -〈A2B2〉
    )
    return [E_expr - score]

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
    """
    Returns additional monomials to add to sdp relaxation.
    """

    monos = []

    # Add ABZ
    Aflat = ncp.flatten(A)
    Bflat = ncp.flatten(B)
    for a in Aflat:
        for b in Bflat:
            for z in ncp.flatten([C]):
                monos += [a*b*z]

    # Add monos appearing in objective function
    for z in ncp.flatten([C]):
        monos += [A[0][0]*z]

    return monos[:]



def compute_entropy(SDP):
    
    ent_value = 0

    obj = 0
    i = 0
    F = [A[0][0],1-A[0][0]]
    for time_step in range(len(gamma)):
        obj = 0
        i = 0
        for a in range(len(F)):
            obj += -beta[time_step]*F[a]*C[i] - gamma[time_step]*C[i]
            i = i+1
        SDP.set_objective(obj)
        SDP.solve('mosek')
        ent_value += SDP.dual

    return ent_value



import numpy as np
from math import sqrt, log2, log, pi, cos, sin
import ncpol2sdpa as ncp
from sympy.physics.quantum.dagger import Dagger
import mosek
import chaospy
import time

LEVEL = 2                        # NPA relaxation level
KEEP_M = 0                        # Optimizing mth objective function?
VERBOSE = 1                        # If > 1 then ncpol2sdpa will also be verbose
WMAX = 0.5 + sqrt(2)/4            # Maximum CHSH score

# Description of Alice and Bobs devices (each input has 2 outputs)
A_config = [2,2,2]
B_config = [2,2,2]

# Operators in the problem Alice, Bob and Eve
A = [Ai for Ai in ncp.generate_measurements(A_config, 'A')]
B = [Bj for Bj in ncp.generate_measurements(B_config, 'B')]
C = ncp.generate_operators('C', 2, hermitian=True)


substitutions = {}            # substitutions to be made (e.g. projections)
moment_ineqs = []            # Moment inequalities (e.g. Tr[rho CHSH] >= c)
moment_eqs = []                # Moment equalities (not needed here)
op_eqs = []                    # Operator equalities (not needed here)
op_ineqs = []                # Operator inequalities (e.g. Id - A00 >= 0 -- we don't include for speed)
extra_monos = []            # Extra monomials to add to the relaxation beyond the level.


# Get the relevant substitutions
substitutions = get_subs()

# Define the moment inequality related to chsh score
test_score = 5.035
score_cons = score_constraints_i3322(test_score)

# Get any extra monomials we wanted to add to the problem
extra_monos = get_extra_monomials()

# Define the objective function (changed later)
obj = 0.0

# Finally defining the sdp relaxation in ncpol2sdpa
ops = ncp.flatten([A,B,C])
sdp = ncp.SdpRelaxation(ops, verbose = VERBOSE-1, normalized=True, parallel=0)
sdp.get_relaxation(level = LEVEL,
    equalities = op_eqs[:],
    inequalities = op_ineqs[:],
    momentequalities = moment_eqs[:],
    momentinequalities = moment_ineqs[:] + score_cons[:],
    objective = obj,
    substitutions = substitutions,
    extramonomials = extra_monos)

# # Test
# ent = compute_entropy(sdp)
# print("Analytical bound:", Hvn(test_score))
# print("SDP bound:" , ent)
# exit()


"""
Now let's collect some data
"""
# We'll loop over the different CHSH scores and compute lower bounds on the rate of the protocol
entropy = []
times = []

loop_scores = np.linspace(4.5, 5.0, 20)  # loop over these scores

for score in loop_scores:
    t1 = time.time()
    
    # Modify the CHSH score
    sdp.process_constraints(
        equalities = op_eqs[:],
        inequalities = op_ineqs[:],
        momentequalities = moment_eqs[:],
        momentinequalities = moment_ineqs[:] + score_constraints_i3322(score)
    )
    
    # Get the resulting entropy bound
    ent = compute_entropy(sdp)
    ent_adjusted = 1 / np.log(2) * (1 + ent)  # same formula you print
    entropy.append(ent_adjusted)
    print("PSD block sizes:", sdp.block_struct)
    print("Total scalar variables in PSD blocks:", sum(n*n for n in sdp.block_struct if n > 0))

    t2 = time.time()
    elapsed = t2 - t1
    times.append(elapsed)
    
    print(score, ent_adjusted)
    print(elapsed)

# Create and save two CSV files
results_entropy = np.column_stack((loop_scores, entropy))
results_time = np.column_stack((loop_scores, times))

np.savetxt("entropy_results_ks_60nodes.txt", results_entropy, delimiter=",", header="score,entropy", comments="")
np.savetxt("time_results_ks_60nodes.txt", results_time, delimiter=",", header="score,time", comments="")
