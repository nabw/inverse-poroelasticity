from firedrake import *
from firedrake.formmanipulation import split_form, ExtractSubBlock
from AndersonAcceleration import AndersonAcceleration
from Forms import *
parameters["form_compiler"]["quadrature_degree"] = 6
import numpy as np
import sys
import os
from sys import argv

def parprint(*args):
    if COMM_WORLD.rank == 0:
        print("[=]", *args, flush=True)
# setting path
current = os.path.dirname(os.path.realpath(__file__))
fv_dir = os.path.dirname(current) + '/fibers-vec'
sys.path.append(fv_dir)
exec(open(fv_dir + '/fibers-vec.py').read())
f0, s0, n0 = f, s, n
#mesh = Mesh('prolate_4mm.msh'), already defined in 'exec'
mesh.coordinates.dat.data[:] *= 1e-3
dx = dx(mesh)
ds = ds(mesh)
ds_endo = ds(20)
ds_epi = ds(10)
ds_base = ds(50)
incompressibility=False
phi_idx = 2 if incompressibility else 1

# Time discretization
dt = 1e-2
ramp_time = 1e-1

# Simulation parameters
do_backward = True
do_forward = True
saveEvery = 1
printEvery=1
accelEvery=1
accelEveryForw=1
ref_tol = 1e-4
aa = int(argv[1]) 
order_back = aa
order_forw = aa
delay = 2 # Accel starts at 1=1e-2. Not anymore (check)
formulation=PRIMAL
assert formulation!=MIXEDU, "Be nice to yourself"

# Physical parameters
p_source = 3e3
p_endo = 1.5e3
theta = 3e-5
phi0_art = 0.1

# Functional setting
functions = Functions(mesh, formulation, bubble=True, fibers=(f,s,n), incompressibility=incompressibility)
functions.setCardiac(ds_epi, ds_endo, ds_base, p_endo)
Vu = functions.Vu
Vphi = functions.Vphi
#Vp = functions.Vp
V = functions.V
VuP1 = VectorFunctionSpace(mesh, 'CG', 1)

dofs = V.dim()
parprint("DoFs:", dofs)
# Functions
sol = functions.sol
sol0 = functions.sol0
if incompressibility:
    u, p, phi = functions.u, functions.p, functions.phi
    u0, p0, phi0 = functions.u0, functions.p0, functions.phi0
else:
    u, phi = functions.u, functions.phi
    u0, phi0 = functions.u0, functions.phi0
phi_n = functions.phi_n
phi0_n = functions.phi0_n


# Initial condition and bcs
sol.subfunctions[phi_idx].interpolate(Constant(phi0_art))
sol0.subfunctions[phi_idx].interpolate(Constant(phi0_art))
phi_n.assign(Constant(phi0_art))
phi0_n.assign(Constant(phi0_art))
bcs = []
if formulation==MIXEDU:
    bcs.append(DirichletBC(V.sub(phi_idx+1), zero_vec, "on_boundary"))
time = Constant(0)


anderson = AndersonAcceleration(sol0.sub(phi_idx), order_back, delay, restart=False)
FF = getBackwardProblem(time, dt, ramp_time, theta, p_source, functions, formulation, False, cardiac=True, imex=False, incompressibility=incompressibility)
FFres = getBackwardProblem(time, dt, ramp_time, theta, p_source, functions, formulation, True, cardiac=True, imex=False, incompressibility=incompressibility)
JJ = derivative(FF, sol0)
parameters={"snes_error_if_not_converged": True, 
            #"mat_type": "nest",
            #"snes_type": "newtontr",
            "snes_linesearch_type": "none",
            "snes_atol": 1e-10, # If not, stuck at 1e-8
            "snes_rtol": 1e-6, 
            "snes_stol": 0.0, 
            "snes_monitor": None,
            "ksp_converged_reason": None,
            "snes_converged_reason": None,
            #"ksp_converged_reason": None,
            #"snes_lag_jacobian": 10,
            #"ksp_type": "preonly",
            #"ksp_gmres_restart": 200,
            #"ksp_atol": 0.0,
            #"ksp_rtol": 1e-4,
            #"ksp_norm_type": "unpreconditioned", 
            #"pc_type": "lu",
            #"pc_fieldsplit_type": "additive",
            #"pc_fieldsplit_0_fields": "0,1",
            #"pc_fieldsplit_1_fields": "2",
            #"fieldsplit_0_ksp_type": "preonly", 
            #"fieldsplit_1_ksp_type": "preonly", 
            #"fieldsplit_0_ksp_converged_reason": None,
            #"fieldsplit_1_ksp_monitor": None,
            #"fieldsplit_0_pc_type": "lu",
            #"fieldsplit_0_pc_factor_mat_solver_type": "mumps",
            #"fieldsplit_0_mat_mumps_cntl_5": 1000, # null pivots
            #"fieldsplit_0_mat_mumps_icntl_14": 100,
            #"fieldsplit_0_pc_hypre_type": "euclid", 
            #"fieldsplit_0_pc_hypre_euclid_level": 3, 
            #"fieldsplit_0_pc_reuse_ordering": True, 
            #"fieldsplit_1_ksp_converged_reason": None,
            #"fieldsplit_1_pc_type": "hypre",
            #"pc_factor_mat_solver_type": "mumps"
            }

#if formulation in (MIXEDP,MIXEDU):
    #parameters["pc_fieldsplit_1_fields"] = "2,3"

#J = derivative(FF, sol0, TrialFunction(V)) # TrialFunctions implicit
#splitter = ExtractSubBlock()
#Jdiag = splitter.split(J, ((0,0), (0,1),(1,0),(1,1), (2,2),(2,3),(3,2),(3,3)))
problem = NonlinearVariationalProblem(FF, sol0, J=JJ)
solver = NonlinearVariationalSolver(problem, solver_parameters=parameters)
time.assign(ramp_time)
res_vec = assemble(FFres)
err0 = res_vec.vector().inner(res_vec.vector())
err0 = sqrt(err0)
if err0<1e-10: err0 = 1
time.assign(0.0)
i = 1
if do_backward:
    outfile = File("output/lv-one-phase-backward.pvd")
    outfile.write(*sol0.subfunctions, t=0)
    converged = False
    t = dt
    time.assign(t)
    while not converged:
        solver.solve()
        if i%accelEvery==0: 
            anderson.get_next_vector(sol0.sub(phi_idx))
        assemble(FFres, tensor=res_vec)
        err = sqrt(res_vec.vector().inner(res_vec.vector()))/err0
        if i % printEvery == 0: parprint(f"Backward: It {i:4}, time {t:4.4f}, err={err:4.2e}")
        if err < ref_tol:
            t = t + dt # Increase ramp only of converged
            time.assign(t)
            anderson.reset()
            if t > ramp_time: 
                converged=True
                break
        phi0_n.assign(sol0.subfunctions[phi_idx])
        if i % saveEvery == 0:outfile.write(*sol0.subfunctions, t=t)
        i = i + 1

# Move mesh to reference configuration
u1 = interpolate(u0, VuP1)
mesh.coordinates.vector().axpy(1.0, u1)

# Init forward solution
sol.assign(sol0)
sol.sub(0).project(Constant((0,0,0)))

J_temp = 1 / det(grad(u0) + Identity(3))
varphi0 = J_temp * functions.phi
phi0 = Function(Vphi)
phi0.interpolate(varphi0) # varphi = J * phi
phi_n.assign(phi0) 
J_fun = Function(Vp, name="J")
J_fun.interpolate(det(grad(u) + Identity(3)))
anderson = AndersonAcceleration(sol.sub(phi_idx), order_forw, delay, restart=False)
FF = getForwardProblem(time, dt, ramp_time, theta, p_source, functions, formulation, stationaryResidual=False, cardiac=True, incompressibility=incompressibility)
FFres = getForwardProblem(time, dt, ramp_time, theta, p_source, functions, formulation, stationaryResidual=True, cardiac=True, incompressibility=incompressibility)

problem = NonlinearVariationalProblem(FF, sol)
solver = NonlinearVariationalSolver(problem, solver_parameters=parameters)
time.assign(time_ramp)
res_vec = assemble(FFres)
err0 = res_vec.vector().inner(res_vec.vector())
err0 = sqrt(err0)
if err0<1e-10: err0 = 1
time.assign(0.0)

i = 1
if do_forward:
    outfile = File("output/lv-one-phase-forward.pvd")
    outfile.write(*sol.subfunctions, J_fun, t=0)
    converged = False
    t = dt
    time.assign(t)
    while not converged:
        solver.solve()
        if i%accelEveryForw==0: 
            anderson.get_next_vector(sol.sub(phi_idx))
        assemble(FFres, tensor=res_vec)
        err = sqrt(res_vec.vector().inner(res_vec.vector()))/err0
        if i % printEvery == 0: parprint(f"Forward: It {i:4}, time {t:4.4f}, err={err:4.2e}")
        if err < ref_tol:
            t = t + dt # Increase ramp only of converged
            time.assign(t)
            anderson.reset()
            if t > ramp_time: 
                converged=True
                break
        phi_n.assign(functions.phi)
        if i % saveEvery == 0: 
            J_fun.interpolate(det(grad(u) + Identity(3)))
            outfile.write(*sol.subfunctions, J_fun, t=t)
        i = i + 1
parprint("done")
