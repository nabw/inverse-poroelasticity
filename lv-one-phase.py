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
mesh.coordinates.dat.data[:] *= 1e-3  # mm to m
dx = dx(mesh)
ds = ds(mesh)
ds_endo = ds(20)
ds_epi = ds(10)
ds_base = ds(50)
incompressibility=False
phi_idx = 2 if incompressibility else 1

# Time discretization
dt = 0.1 # no ramp, physiological values are nice
ramp_time = 1.0
time = Constant(0.0)

# Simulation parameters
do_backward = True
do_forward = True
saveEvery = 1
printEvery =1
accelEvery =1
accelEveryForw=1
ref_tol = 1e-6
aa = int(argv[1])
order_back = aa
order_forw = aa
delay = 2 # Delay at each time step
formulation=PRIMAL

# Physical parameters
p_source = 5e3
p_endo = 1.5e3 # 12 mmHg, maximum phys value
theta = 3e-5
phi0_art = 0.1

# Functional setting
u_deg = 1
phi_deg = 1
functions = Functions(mesh, formulation, u_deg, phi_deg, splitting=True, fibers=(f,s,n), incompressibility=incompressibility)
functions.setCardiac(ds_epi, ds_endo, ds_base, p_endo)
Vu = functions.Vu
Vphi = functions.Vphi
Vp = functions.Vp
#V = functions.V
VuP1 = VectorFunctionSpace(mesh, 'CG', 1)

# Functions
#sol = functions.sol
#sol0 = functions.sol0
if incompressibility:
    u, p, phi = functions.u, functions.p, functions.phi
    u0, p0, phi0 = functions.u0, functions.p0, functions.phi0
else:
    u, phi = functions.u, functions.phi
    u0, phi0 = functions.u0, functions.phi0
phi_n = functions.phi_n
phi0_n = functions.phi0_n
phi_fun = functions.phi_fun
phi0_fun = functions.phi0_fun


# Initial condition and bcs
phi_fun.interpolate(Constant(phi0_art))
phi0_fun.interpolate(Constant(phi0_art))
phi_n.assign(Constant(phi0_art))
phi0_n.assign(Constant(phi0_art))
bcs = []
if formulation==MIXEDU:
    bcs.append(DirichletBC(V.sub(phi_idx+1), zero_vec, "on_boundary"))

FF, FPM = getBackwardProblem(time, dt, ramp_time, theta, p_source, functions, formulation, False, cardiac=True, imex=False, split=True, incompressibility=incompressibility)
FFres = getBackwardProblem(time, dt, ramp_time, theta, p_source, functions, formulation, True, cardiac=True, imex=False, incompressibility=incompressibility)
parameters_inner_mech={"snes_error_if_not_converged": True, 
            "snes_type": "newtonls",
            "snes_max_it": 1000,
            "snes_ksp_ew": None, 
            "snes_ksp_ew_rtol0": 1e-2, 
            "snes_ksp_ew_rtolmax": 1e-2, 
            #"snes_linesearch_type": "none",
            "snes_atol": 1e-12,
            "snes_rtol": 1e-6, 
            "ksp_atol": 1e-14, 
            "ksp_rtol": 1e-2, 
            "snes_stol": 0.0, 
            #"snes_monitor": None,
            #"ksp_monitor": None,
            #"snes_converged_reason": None,
            #"ksp_converged_reason": None,
            "ksp_type": "gmres",
            "ksp_gmres_modifiedgramschmidt": None,
            "ksp_norm_type": "unpreconditioned",
            #"ksp_max_it": 5,
            "ksp_gmres_restart": 500,
            "pc_type": "hypre",
            "pc_hypre_boomeramg_interp_type": "ext+i", 
            "pc_hypre_boomeramg_coarsen_type": "HMIS"
            }

parameters_inner={
            #"snes_error_if_not_converged": True, 
            #"mat_type": "nest",
            "snes_type": "newtonls",
            #"snes_linesearch_type": "none",
            #"snes_atol": 1e-8, # If not, stuck at 1e-9
            "snes_rtol": 1e-6, 
            #"snes_max_it": 200, 
            #"snes_stol": 0.0, 
            #"snes_ksp_ew": None, 
            #"snes_ksp_ew_rtol0": 1e-1, 
            #"snes_ksp_ew_rtolmax": 1e-1, 
            #"snes_divergence_tolerance": 1e16, 
            "ksp_divtol": 1e16, 
            "ksp_atol": 1e-14, 
            "ksp_rtol": 1e-6, 
            "ksp_max_it": 1000, 
            #"snes_monitor": None,
            #"snes_converged_reason": None,
            #"ksp_monitor": None,
            #"ksp_converged_reason": None,
            "ksp_type": "gmres",
            "ksp_gmres_restart": 1000,
            "ksp_norm_type": "unpreconditioned",
            "pc_type": "hypre",
            #"pc_hypre_type": "euclid",
            #"pc_factor_mat_solver_type": "mumps",
            #"mat_mumps_icntl_24": 1, # null pivots
            }

# Build RM nnsp for mechanics
x = mesh.coordinates
e0 = Constant((1,0,0))
e1 = Constant((0,1,0))
e2 = Constant((0,0,1))
def genVec(vv): interpolate(vv, Vu)
exps = [e0, e1, e2, cross(e0, x), cross(e1, x), cross(e2,x)]
vecs = [Function(Vu) for vv in exps]
for i,e in enumerate(exps):
    vecs[i].interpolate(e)
nn = VectorSpaceBasis(vecs)
nn.orthonormalize()


problem = NonlinearVariationalProblem(FF, functions.u0)
solver_mech = NonlinearVariationalSolver(problem, solver_parameters=parameters_inner_mech, near_nullspace=nn)
if formulation == PRIMAL:
    var = functions.pphi0 if incompressibility else functions.phi0
    problem = NonlinearVariationalProblem(FPM, var)
    solver_pm = NonlinearVariationalSolver(problem, solver_parameters=parameters_inner)
elif formulation == MIXEDP:
    var = functions.pphimu0 if incompressibility else functions.phimu0
    problem = NonlinearVariationalProblem(FPM, var)
    solver_pm = NonlinearVariationalSolver(problem, solver_parameters=parameters_inner)
anderson = AndersonAcceleration(phi0_fun, order_back, delay, restart=False)

time.assign(ramp_time) # Before PM to get good normalization factors
res_vec = assemble(FFres)
err0 = res_vec.vector().inner(res_vec.vector())
err0 = sqrt(err0)
if err0<1e-10: err0 = 1
time.assign(0.0)
i = 0
parprint(f"Backward: It {i:4}, time 0.0000, err=1.0")
i = 1
if do_backward:
    outfile = File("output/lv-one-phase-backward.pvd")
    outfile.write(functions.u0, phi0_fun, t=0)
    converged = False
    t = dt
    time.assign(t)
    while not converged:
        solver_mech.solve()
        solver_pm.solve()
        if i%accelEvery==0: 
            anderson.get_next_vector(phi0_fun)
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
        phi0_n.assign(phi0_fun)
        if i % saveEvery == 0: outfile.write(functions.u0, phi0_fun)
        i = i + 1

# Move mesh to reference configuration
u1 = interpolate(u0, VuP1)
mesh.coordinates.vector().axpy(1.0, u1)

# Init forward solution
phi_fun.assign(phi0_fun)
u.interpolate(Constant((0,0,0)))

J_temp = 1 / det(grad(u0) + Identity(3))
varphi0 = J_temp * functions.phi
phi0 = Function(Vphi)
phi0.interpolate(varphi0) # varphi = J * phi
phi_n.assign(phi0) 
phi_fun.assign(phi0)
J_fun = Function(Vp, name="J")
J_fun.interpolate(det(grad(u) + Identity(3)))
anderson = AndersonAcceleration(phi_fun, order_forw, delay, restart=False)
FF, FPM = getForwardProblem(time, dt, ramp_time, theta, p_source, functions, formulation, stationaryResidual=False, cardiac=True, imex=False, split=True, incompressibility=incompressibility)
FFres = getForwardProblem(time, dt, ramp_time, theta, p_source, functions, formulation, stationaryResidual=True, cardiac=True, imex=False, split=True, incompressibility=incompressibility)

problem = NonlinearVariationalProblem(FF, functions.u)
solver_mech = NonlinearVariationalSolver(problem, solver_parameters=parameters_inner_mech)
if formulation == PRIMAL:
    var = functions.pphi if incompressibility else functions.phi
    problem = NonlinearVariationalProblem(FPM, var)
    solver_pm = NonlinearVariationalSolver(problem, solver_parameters=parameters_inner)
elif formulation == MIXEDP:
    var = functions.pphimu if incompressibility else functions.phimu
    problem = NonlinearVariationalProblem(FPM, var)
    solver_pm = NonlinearVariationalSolver(problem, solver_parameters=parameters_inner)

time.assign(ramp_time)
res_vec = assemble(FFres)
err0 = res_vec.vector().inner(res_vec.vector())
err0 = sqrt(err0)
if err0<1e-10: err0 = 1
time.assign(0.0)

i = 0
parprint(f"Forward: It {i:4}, time 0.0000, err=1.0")
i = 1
phi_out = Function(Vphi, name="phi_current")
phi_out.interpolate(phi_fun / J_fun)
if do_forward:
    outfile = File("output/lv-one-phase-forward.pvd")
    outfile.write(functions.u, phi_fun, phi_out, J_fun, t=0)
    converged = False
    t = dt
    time.assign(t)
    while not converged:
        solver_mech.solve()
        solver_pm.solve()
        if i%accelEveryForw==0: 
            anderson.get_next_vector(phi_fun)
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
        phi_n.assign(phi_fun)
        if i % saveEvery == 0: 
            J_fun.interpolate(det(grad(u) + Identity(3)))
            phi_out.interpolate(phi_fun / J_fun)
            outfile.write(functions.u, phi_fun, phi_out, J_fun, t=t)
        i = i + 1
parprint("done")
