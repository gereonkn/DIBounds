import numpy as np
from math import sqrt, log2, log, pi, cos, sin
import ncpol2sdpa as ncp
from sympy.physics.quantum.dagger import Dagger
import mosek
import chaospy


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

print(lists_gb_inner(grid_function(1,0.1,0.1,1)))


def score_constraints(score):
    """
    Returns CHSH score constraint
    """
    chsh_expr = (A[0][0]*B[0][0] + (1-A[0][0])*(1-B[0][0]) + \
        A[0][0]*B[1][0] + (1-A[0][0])*(1-B[1][0]) + \
        A[1][0]*B[0][0] + (1-A[1][0])*(1-B[0][0]) + \
        A[1][0]*(1-B[1][0]) + (1-A[1][0])*B[1][0])/4.0

    return [chsh_expr - score]

def get_subs():
    """
    Returns any substitution rules to use with ncpol2sdpa. E.g. projections and
    commutation relations.
    """
    subs = {}
    # Get Alice and Bob's projective measurement constraints
    subs.update(ncp.projective_measurement_constraints(A,B))
    # Finally we note that Alice and Bob's operators should All commute with Eve's ops
    for a in ncp.flatten([A,B]):
        for p in P:
            subs.update({p*a : a*p})
    for p in P:
        subs.update({p*p : p})
    return subs


grid = grid_function(1,0.1,0.1,1)
gamma, beta = lists_gb_inner(grid)

print(test_integration_formula(gamma,beta))

"""
til here just preprocessing
"""




def get_extra_monomials():
    """
    Returns additional monomials to add to sdp relaxation.
    """

    monos = []
    Aflat = ncp.flatten(A)
    Bflat = ncp.flatten(B)
    for a in Aflat:
        for b in Bflat:
            for p in P:
                monos += [a*b*p]


    return monos[:]






def compute_entropy(SDP,gamma,beta):
    """
    Computes lower bound on H(A|X=0,E) using the fast (but less tight) method

        SDP -- sdp relaxation object
    """
    ent = 0.0        # lower bound on H(A|X=0,E)

    # We can also decide whether to perform the final optimization in the sequence
    # or bound it trivially. Best to keep it unless running into numerical problems



    F = [A[0][0], 1-A[0][0]]
    for time_step in range(len(gamma)):
        obj = 0
        for a in range(len(F)):
            obj += -beta[time_step]*F[a]*P[a] - gamma[time_step]*P[a]

        SDP.set_objective(obj)
        SDP.solve('mosek')

        if SDP.status == 'optimal':

            ent += SDP.primal


    return ent


def h(p):
    return -p*log2(p) - (1-p)*log2(1-p)
def Hvn(w):
    return 1 - h(1/2 + sqrt((8*w - 4)**2 / 4 - 1)/2)






LEVEL = 2                        # NPA relaxation level
VERBOSE = 1                        # If > 1 then ncpol2sdpa will also be verbose
WMAX = 0.5 + sqrt(2)/4            # Maximum CHSH score

# Description of Alice and Bobs devices (each input has 2 outputs)
A_config = [2,2]
B_config = [2,2]

# Operators in the problem Alice, Bob and Eve
A = [Ai for Ai in ncp.generate_measurements(A_config, 'A')]
B = [Bj for Bj in ncp.generate_measurements(B_config, 'B')]
P = ncp.generate_operators('P', 2, hermitian=True)

substitutions = {}            # substitutions to be made (e.g. projections)
moment_ineqs = []            # Moment inequalities (e.g. Tr[rho CHSH] >= c)
moment_eqs = []                # Moment equalities (not needed here)
op_eqs = []                    # Operator equalities (not needed here)
op_ineqs = []                # Operator inequalities (e.g. Id - A00 >= 0 -- we don't include for speed)
extra_monos = []            # Extra monomials to add to the relaxation beyond the level.


# Get the relevant substitutions
substitutions = get_subs()

# Define the moment inequality related to chsh score
test_score = 0.85
score_cons = score_constraints(test_score)

extra_monos = get_extra_monomials()

# Define the objective function (changed later)
obj = 0.0

# Finally defining the sdp relaxation in ncpol2sdpa
ops = ncp.flatten([A,B,P])
sdp = ncp.SdpRelaxation(ops, verbose = VERBOSE-1, normalized=True)
sdp.get_relaxation(level = LEVEL,
    equalities = op_eqs[:],
    inequalities = op_ineqs[:],
    momentequalities = moment_eqs[:],
    momentinequalities = moment_ineqs[:] + score_cons[:],
    objective = obj,
    substitutions = substitutions,
    extramonomials = extra_monos)

scores = np.linspace(0.75, WMAX-1e-4, 20).tolist()
entropy = []
for score in scores:
    # Modify the CHSH score
    sdp.process_constraints(equalities = op_eqs[:],
    inequalities = op_ineqs[:],
    momentequalities = moment_eqs[:],
    momentinequalities = moment_ineqs[:] + score_constraints(score))
    # Get the resulting entropy bound
    ent = compute_entropy(sdp,gamma,beta)
    entropy += [ent]
    print(score, Hvn(score), 1/log(2)*(1+ent))
