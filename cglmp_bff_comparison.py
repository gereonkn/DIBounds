

def objective(ti):

    obj = 0.0
    F = [A[0][0],A[0][1],1-A[0][0]-A[0][1]]
    for a in range(len(F)):
        obj += F[a] * (Z[a] + Dagger(Z[a]) + (1-ti)*Dagger(Z[a])*Z[a]) + ti*Z[a]*Dagger(Z[a])

    return obj

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

def get_subs():

    subs = {}
    # Get Alice and Bob's projective measurement constraints
    subs.update(ncp.projective_measurement_constraints(A,B))

    # Finally we note that Alice and Bob's operators should All commute with Eve's ops
    for a in ncp.flatten([A,B]):
        for z in Z:
            subs.update({z*a : a*z, Dagger(z)*a : a*Dagger(z)})

    return subs

def get_extra_monomials():

    monos = []

    # Add ABZ
    ZZ = Z + [Dagger(z) for z in Z]
    Aflat = ncp.flatten(A)
    Bflat = ncp.flatten(B)
    for a in Aflat:
        for b in Bflat:
            for z in ZZ:
                monos += [a*b*z]

    # Add monos appearing in objective function
    for z in Z:
        monos += [A[0][0]*Dagger(z)*z]

    return monos[:]


def generate_quadrature(m):

    t, w = chaospy.quad_gauss_radau(m, chaospy.Uniform(0, 1), 1)
    t = t[0]
    return t, w

def compute_entropy(SDP):

    ck = 0.0        # kth coefficient
    ent = 0.0        # lower bound on H(A|X=0,E)


    if KEEP_M:
        num_opt = len(T)
    else:
        num_opt = len(T) - 1

    for k in range(num_opt):
        ck = W[k]/(T[k] * log(2))

        # Get the k-th objective function
        new_objective = objective(T[k])

        SDP.set_objective(new_objective)
        SDP.solve('mosek')

        if SDP.status == 'optimal':
            # 1 contributes to the constant term
            ent += ck * (1 + SDP.dual)
        else:
            # If we didn't solve the SDP well enough then just bound the entropy
            # trivially
            ent = 0
            if VERBOSE:
                print('Bad solve: ', k, SDP.status)
            break

    return ent

# Some functions to help compute the known optimal bounds for the CHSH rates
def hmin(w):
    return -log2(1/2 + sqrt(2 - (8*w - 4)**2 / 4)/2)
def h(p):
    return -p*log2(p) - (1-p)*log2(1-p)
def Hvn(w):
    return 1 - h(1/2 + sqrt((8*w - 4)**2 / 4 - 1)/2)


import numpy as np
from math import sqrt, log2, log, pi, cos, sin
import ncpol2sdpa as ncp
from sympy.physics.quantum.dagger import Dagger
import mosek
import chaospy
import time

LEVEL = 2                        # NPA relaxation level
M = 4                            # Number of nodes / 2 in gaussian quadrature
T, W = generate_quadrature(M)    # Nodes, weights of quadrature
KEEP_M = 0                        # Optimizing mth objective function?
VERBOSE = 1                        # If > 1 then ncpol2sdpa will also be verbose
WMAX = 0.5 + sqrt(2)/4            # Maximum CHSH score

# Description of Alice and Bobs devices (each input has 2 outputs)
A_config = [3, 3]
B_config = [3, 3]

# Operators in the problem Alice, Bob and Eve
A = [Ai for Ai in ncp.generate_measurements(A_config, 'A')]
B = [Bj for Bj in ncp.generate_measurements(B_config, 'B')]
Z = ncp.generate_operators('Z', 3, hermitian=0)


substitutions = {}            # substitutions to be made (e.g. projections)
moment_ineqs = []            # Moment inequalities (e.g. Tr[rho CHSH] >= c)
moment_eqs = []                # Moment equalities (not needed here)
op_eqs = []                    # Operator equalities (not needed here)
op_ineqs = []                # Operator inequalities (e.g. Id - A00 >= 0 -- we don't include for speed)
extra_monos = []            # Extra monomials to add to the relaxation beyond the level.


# Get the relevant substitutions
substitutions = get_subs()

# Define the moment inequality related to chsh score
test_score = 2.914854
score_cons = score_constraints_cglmp(test_score)

# Get any extra monomials we wanted to add to the problem
extra_monos = get_extra_monomials()

# Define the objective function (changed later)
obj = 0.0

# Finally defining the sdp relaxation in ncpol2sdpa
ops = ncp.flatten([A,B,Z])
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
scores = np.linspace(2,2.914854,10)

for score in scores:
    t1 = time.time()
    
    # Modify the CHSH score
    sdp.process_constraints(
        equalities = op_eqs[:],
        inequalities = op_ineqs[:],
        momentequalities = moment_eqs[:],
        momentinequalities = moment_ineqs[:] + score_constraints_cglmp(score)
    )
    print("PSD block sizes:", sdp.block_struct)
    print("Total scalar variables in PSD blocks:", sum(n*n for n in sdp.block_struct if n > 0))

    # Get the resulting entropy bound
    ent = compute_entropy(sdp)
    entropy.append(ent)
    
    t2 = time.time()
    elapsed = t2 - t1
    times.append(elapsed)
    
    print(score, ent, f"(time: {elapsed:.4f} s)")


