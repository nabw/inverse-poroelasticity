from firedrake import *
from AndersonAcceleration import AndersonAcceleration
from Forms import *
parameters["form_compiler"]["quadrature_degree"] = 6
from sys import argv

# Input parameters
form = int(argv[1])
aa = int(argv[2])

if form == 0:
    formulation=PRIMAL
elif form == 1:
    formulation=MIXEDP
elif form == 2:
    formulation=MIXEDU
else:
    assert False, "First parameter in {0,1,2}"

# Script config
do_backward = True
do_forward = True
monitor=False # active PETSc monitors
saveEvery=1
printEvery=1
accelEveryForw=1
ref_tol_back = 1e-5
ref_tol_forw = 1e-5
order_back= aa
order_forw = aa
delay = 0 # Accel starts at 1=1e-2

# Mesh
Lx = 1e-2
Ly = 1e-2

Nx = 12
Ny = Nz = int(Nx/2)
mesh = BoxMesh(Nx, Ny, Nz, 5*Lx, Ly, Ly)
zero_vec = Constant((0,0,0))
dx = dx(mesh)
ds = ds(mesh)
dim=3

# Discretization
dt = 1e-2
ramp_time = 1e-1
u_deg = 2
phi_deg = 1

# Loading parameters
theta = 1e-4
p_source = 1e4

# Functional setting
functions = Functions(mesh, formulation, u_deg, phi_deg)
Vu = functions.Vu
Vphi = functions.Vphi
Vp = functions.Vp
V = functions.V
VuP1 = VectorFunctionSpace(mesh, 'CG', 1)

dofs = V.dim()
parprint("DoFs:", dofs)

# Functions
sol = functions.sol
sol0 = functions.sol0
u, p, phi = functions.u, functions.p, functions.phi
u0, p0, phi0 = functions.u0, functions.p0, functions.phi0
phi_n = functions.phi_n
phi0_n = functions.phi0_n
sol.subfunctions[0].rename('u')
sol.subfunctions[1].rename('p')
sol.subfunctions[2].rename('phi')
sol0.subfunctions[0].rename('u0')
sol0.subfunctions[1].rename('p0')
sol0.subfunctions[2].rename('phi0')


# Initial condition and bcs
sol.subfunctions[2].interpolate(Constant(phi0_art))
sol0.subfunctions[2].interpolate(Constant(phi0_art))
phi_n.assign(Constant(phi0_art))
phi0_n.assign(Constant(phi0_art))

bcs=[]
for i in range(dim):
    bcs.append(DirichletBC(V.sub(0).sub(i), Constant(0), 2*i+1))
if formulation==MIXEDU:
    bcs.append(DirichletBC(V.sub(3), zero_vec, "on_boundary"))
time = Constant(0)

parameters={ "snes_error_if_not_converged": True, "snes_linesearch_type": "none", "snes_atol": 1e-12, "snes_rtol": 1e-4, "snes_stol": 0.0, 'snes_divergence_tolerance': 1e10}
if monitor: parameters["snes_monitor"] = None
# Initialize and iterate
i = 0
anderson = AndersonAcceleration(sol0.sub(2), order_back, delay, restart=False)
FF = getBackwardProblem(time, dt, ramp_time, theta, p_source, functions, formulation, stationaryResidual=False)
FFres = getBackwardProblem(time, dt, ramp_time, theta, p_source, functions, formulation, stationaryResidual=True)
time.assign(ramp_time)
res_vec = assemble(FFres, bcs=bcs)
err0 = sqrt(res_vec.vector().inner(res_vec.vector()))
time.assign(0.0)
if do_backward:
    outfile = File("output/pm-one-phase-3d-backward.pvd")
    outfile.write(*sol0.subfunctions, t=0)
    converged = False
    t = dt
    while not converged:
        time.assign(t)
        solve(FF==0, sol0, bcs=bcs, solver_parameters=parameters)
        if t >= ramp_time: 
            anderson.get_next_vector(sol0.sub(2))
        assemble(FFres, bcs=bcs, tensor=res_vec)
        err = sqrt(res_vec.vector().inner(res_vec.vector()))/err0
        if i % printEvery == 0: print(f"Backward: It {i:4}, time {t:4.3f}, err={err:4.2e}")
        if err < ref_tol_back and t>= ramp_time:
            converged=True
            break
        phi0_n.assign(sol0.subfunctions[2])
        if i % saveEvery == 0:outfile.write(*sol0.subfunctions, t=t)
        i, t = i + 1, t+dt

# Move mesh to reference configuration
u1 = interpolate(u0, VuP1)
mesh.coordinates.vector().axpy(1.0, u1)

# Init forward solution
sol.assign(sol0)
sol.subfunctions[0].interpolate(zero_vec)

# phi -> varphi
J_temp = 1 / det(grad(u0) + Identity(dim))
varphi0 = J_temp * phi
phi0 = Function(Vphi)
phi0.interpolate(varphi0) # varphi = J * phi
phi_n.assign(phi0) 
J_fun = Function(Vp, name="J")
J_fun.interpolate(det(grad(u) + Identity(dim)))
phi_curr = Function(Vphi, name="phi_spatial")
phi_curr.interpolate(phi/J_fun)

# Initialize and iterate
i = 0
anderson = AndersonAcceleration(sol.sub(2), order_forw, delay, restart=False)
FF = getForwardProblem(time, dt, ramp_time, theta, p_source, functions, formulation, False)
FFres = getForwardProblem(time, dt, ramp_time, theta, p_source, functions, formulation, True)
time.assign(ramp_time)
res_vec = assemble(FFres, bcs=bcs)
err0 = sqrt(res_vec.vector().inner(res_vec.vector()))
time.assign(0.0)
if do_forward:
    outfile = File("output/pm-one-phase-3d-forward.pvd")
    outfile.write(*sol.subfunctions, J_fun, phi_curr, t=0)
    converged = False
    t = dt
    while not converged:
        time.assign(t)
        solve(FF==0, sol, bcs=bcs, solver_parameters=parameters)
        if t >= ramp_time and i%accelEveryForw==0: 
            anderson.get_next_vector(sol.sub(2))

        assemble(FFres, bcs=bcs, tensor=res_vec)
        err = sqrt(res_vec.vector().inner(res_vec.vector()))/err0
        if i % printEvery == 0: print(f"Forward: It {i:4}, time {t:4.3f}, err={err:4.2e}")
        if err < ref_tol_forw and t>= ramp_time:
            converged=True
            break
        phi_n.assign(sol.subfunctions[2])
        if i % saveEvery == 0: 
            J_fun.interpolate(det(grad(u) + Identity(dim)))
            phi_curr.interpolate(phi/J_fun)
            outfile.write(*sol.subfunctions, J_fun, phi_curr, t=t)
        i, t = i + 1, t+dt
print("Done")
