from firedrake import *
from AndersonAcceleration import AndersonAcceleration

# Physical params
phi0_art = 0.1 #* 0.2
kappa = 2e-7
k = 5e4 # quasi-incomp
q1a = 22 #1.333
q2a = 1009 # 550
q3a = 10 # 10
p_sink = 1e3

PRIMAL=1
MIXEDP=2
MIXEDU=3

def parprint(*args):
    if COMM_WORLD.rank == 0:
        print("[=]", *args, flush=True)


class Functions:
    def __init__(self, mesh, formulation, u_deg, phi_deg, p_deg=1, splitting=False, fibers=False, incompressibility=True):
        self.mesh = mesh
        #if bubble:
        #    P1 = FiniteElement("CG", cell=mesh.ufl_cell(), degree=1)
        #    B = FiniteElement("B", cell=mesh.ufl_cell(), degree=4)
        #    mini = P1 + B
        #    Vu = VectorFunctionSpace(mesh, mini)
        #else:
        Vu = VectorFunctionSpace(mesh, 'CG', u_deg)
        #phi_deg = 2
        Vphi = FunctionSpace(mesh, 'CG', phi_deg)
        Vp = FunctionSpace(mesh, 'CG', p_deg)
        self.Vu = Vu
        self.Vphi = Vphi
        self.Vp = Vp
        if formulation==PRIMAL:
            if splitting: 
                u = Function(Vu, name="u")
                u0 = Function(Vu, name="u0")
                v = TestFunction(Vu)
                du = TrialFunction(Vu)
                if incompressibility:
                    Vpphi = Vp * Vphi
                    pphi = Function(Vpphi, name="pphi")
                    pphi0 = Function(Vpphi, name="pphi0")
                    q, qa = TestFunctions(Vpphi)
                    dp, dphi = TrialFunction(Vpphi)
                    self.pphi = pphi
                    self.pphi0 = pphi0
                    p, phi = split(pphi)
                    p0, phi0 = split(pphi0)
                    phi_fun = pphi.sub(1)
                    phi0_fun = pphi0.sub(1)
                else: # Compressible
                    phi = Function(Vphi, name="phi")
                    phi0 = Function(Vphi, name="phi0")
                    qa = TestFunction(Vphi)
                    dphi = TrialFunction(Vphi)
                    phi_fun = phi
                    phi0_fun = phi0
            else:
                if incompressibility:
                    V = Vu * Vp * Vphi
                    sol = Function(V, name="sol")
                    sol0 = Function(V, name="sol0")
                    u, p, phi = split(sol)
                    u0, p0, phi0 = split(sol0)
                    v, q, qa = TestFunctions(V)
                    du, dp, dphi = TrialFunctions(V)
                    phi_fun = sol.sub(2)
                    phi0_fun = sol0.sub(2)
                else: 
                    V = Vu * Vphi
                    sol = Function(V, name="sol")
                    sol0 = Function(V, name="sol0")
                    u, phi = split(sol)
                    u0, phi0 = split(sol0)
                    v, qa = TestFunctions(V)
                    du, dphi = TrialFunctions(V)
                    phi_fun = sol.sub(1)
                    phi0_fun = sol0.sub(1)
        elif formulation==MIXEDP:
            Vmu = FunctionSpace(mesh, 'CG', phi_deg)
            if splitting:
                if incompressibility: 
                    Vpphimu = Vp * Vphi * Vmu
                    self.Vpphimu = Vpphimu
                    u = Function(Vu, name="u")
                    u0 = Function(Vu, name="u0")
                    v = TestFunction(Vu)
                    du = TrialFunction(Vu)

                    pphimu = Function(Vpphimu, name="pphimu")
                    pphimu0 = Function(Vpphimu, name="pphimu0")
                    p, phi, mu = split(pphimu)
                    p0, phi0, mu0 = split(pphimu0)
                    q, qa, eta = TestFunctions(Vpphimu)
                    dp, dphi, dmu = TrialFunctions(Vpphimu)
                    phi_fun = pphimu.sub(1)
                    phi0_fun = pphimu0.sub(1)
                    self.pphimu = pphimu
                    self.pphimu0 = pphimu0
                else:
                    Vphimu = Vphi * Vmu
                    self.Vphimu = Vphimu
                    u = Function(Vu, name="u")
                    u0 = Function(Vu, name="u0")
                    v = TestFunction(Vu)
                    du = TrialFunction(Vu)

                    phimu = Function(Vphimu, name="phimu")
                    phimu0 = Function(Vphimu, name="phimu0")
                    phi, mu = split(phimu)
                    phi0, mu0 = split(phimu0)
                    qa, eta = TestFunctions(Vphimu)
                    dphi, dmu = TrialFunctions(Vphimu)
                    phi_fun = phimu.sub(0)
                    phi0_fun = phimu0.sub(0)
                    self.phimu = phimu
                    self.phimu0 = phimu0
            else: # monolithic
                self.Vmu = Vmu
                if incompressibility:
                    V = Vu * Vp * Vphi * Vmu
                    sol = Function(V, name="sol")
                    sol0 = Function(V, name="sol0")
                    u, p, phi, mu = split(sol)
                    u0, p0, phi0, mu0 = split(sol0)
                    v, q, qa, eta = TestFunctions(V)
                    du, dp, dphi, dmu = TrialFunctions(V)
                    phi_fun = sol.sub(2)
                    phi0_fun = sol.sub(2)
                else:
                    V = Vu * Vphi * Vmu
                    sol = Function(V, name="sol")
                    sol0 = Function(V, name="sol0")
                    u, phi, mu = split(sol)
                    u0, phi0, mu0 = split(sol0)
                    v, qa, eta = TestFunctions(V)
                    du, dphi, dmu = TrialFunctions(V)
                    phi_fun = sol.sub(1)
                    phi0_fun = sol.sub(1)
            self.mu = mu
            self.eta = eta
            self.mu0 = mu0
            self.dmu = dmu
        else: # formulation==MIXEDU
            Vv = VectorFunctionSpace(mesh, 'CG', 2)
            if splitting: 
                if incompressibility: 
                    u = Function(Vu, name="u")
                    u0 = Function(Vu, name="u0")
                    v = TestFunction(Vu)
                    du = TrialFunction(Vu)
                    self.u = u
                    self.u0 = u0

                    Vpphivu = Vp * Vphi * Vv
                    self.Vpphivu = Vpphivu
                    pphivu = Function(Vpphivu, name="pphivu")
                    pphivu0 = Function(Vpphivu, name="pphivu0")
                    p, phi, vu = split(pphivu)
                    p, phi0, vu0 = split(pphivu0)
                    q, qa, vv = TestFunctions(Vpphivu)
                    dp, dphi, dvu = TrialFunctions(Vpphivu)
                    phi_fun = pphivu.sub(1)
                    phi0_fun = pphivu0.sub(1)
                    self.pphivu = pphivu
                    self.pphivu0 = pphivu0

                else: # compressible
                    u = Function(Vu, name="u")
                    u0 = Function(Vu, name="u0")
                    v = TestFunction(Vu)
                    du = TrialFunction(Vu)
                    self.u = u
                    self.u0 = u0

                    Vphivu = Vphi * Vv
                    self.Vphivu = Vphivu

                    phivu = Function(Vphivu, name="phivu")
                    phivu0 = Function(Vphivu, name="phivu0")
                    phi, vu = split(phivu)
                    phi0, vu0 = split(phivu0)
                    qa, vv = TestFunctions(Vphivu)
                    dphi, dvu = TrialFunctions(Vphivu)
                    phi_fun = phivu.sub(0)
                    phi0_fun = phivu0.sub(0)
                    self.phivu = phivu
                    self.phivu0 = phivu0

            else: # monolithic
                if incompressibility:
                    self.Vv = Vv
                    V = Vu * Vp * Vphi * Vv
                    sol = Function(V, name="sol")
                    sol0 = Function(V, name="sol0")
                    u, p, phi, vu = split(sol)
                    u0, p0, phi0, vu0 = split(sol0)
                    v, q, qa, vv = TestFunctions(V)
                    du, dp, dphi, duv = TrialFunctions(V)
                    phi_fun = sol.sub(2)
                    phi0_fun = sol.sub(2)
                else:
                    self.Vv = Vv
                    V = Vu * Vphi * Vv
                    sol = Function(V, name="sol")
                    sol0 = Function(V, name="sol0")
                    u, phi, vu = split(sol)
                    u0, phi0, vu0 = split(sol0)
                    v, qa, vv = TestFunctions(V)
                    du, dphi, duv = TrialFunctions(V)
                    phi_fun = sol.sub(1)
                    phi0_fun = sol.sub(1)
            self.vu = vu
            self.vv = vv
            self.vu0 = vu0
            self.duv = duv

        if not splitting:
            self.V = V
            self.sol = sol
            self.sol0 = sol0
        self.u = u
        self.phi = phi
        self.u0 = u0
        self.phi0 = phi0
        self.v = v
        self.qa = qa
        self.du = du
        self.dphi = dphi
        self.phi_n = Function(Vphi, name="phi")
        self.phi0_n = Function(Vphi, name="phi0")
        self.phi_fun = phi_fun
        self.phi0_fun = phi0_fun
        if incompressibility:
            self.p = p
            self.p0 = p0
            self.q = q
            self.dp = dp
        if fibers:
            self.f0 = fibers[0]
            self.s0 = fibers[1]
            self.n0 = fibers[2]
            self.fibers = fibers

    def setCardiac(self, ds_epi, ds_endo, ds_base, p_endo):
        self.ds_epi = ds_epi
        self.ds_endo = ds_endo
        self.ds_base = ds_base
        self.p_endo = p_endo

class PMSplittingSolver:
    def __init__(self, solA, solB, FFA, FFB, JA, JB, JpA, JpB, params_A, params_B, atol, rtol, maxit, bcA=None, bcB=None, nullspaceB=None):
        self.solA = solA
        self.FFA = FFA
        self.JA = JA
        self.JpA = JpA
        self.bcA = bcA
        self.params_A = params_A
        self.solB = solB
        self.FFB = FFB
        self.JB = JB
        self.JpB = JpB
        self.bcB = bcB
        self.params_B = params_B

        # Internal residuals
        self.resA = None
        self.resB = None


        # Solvers
        if bcA:
            problemA = NonlinearVariationalProblem(FFA, solA, bcs=bcA, J=JA, Jp=JpA)
        else:
            problemA = NonlinearVariationalProblem(FFA, solA, J=JA, Jp=JpA)
        solverA = NonlinearVariationalSolver(problemA, solver_parameters=params_A)
        self.solverA = solverA
        if bcA:
            problemB = NonlinearVariationalProblem(FFB, solB, bcs=bcB, J=JB, Jp=JpB)
        else:
            problemB = NonlinearVariationalProblem(FFB, solB, J=JB, Jp=JpB)
        if nullspaceB:
            solverB = NonlinearVariationalSolver(problemB, solver_parameters=params_B, nullspace=nullspaceB)
        else:
            solverB = NonlinearVariationalSolver(problemB, solver_parameters=params_B)
        self.solverB = solverB
        self.atol = atol
        self.rtol = rtol
        self.maxit = maxit

        # Initialize errors
        errA0, errB0 = self.computeErrors()
        self.errA0 = errA0 if errA0 > 1e-10 else 1.0
        self.errB0 = errB0 if errB0 > 1e-10 else 1.0

    def computeErrors(self):
        if self.resA and self.resB:
            resA = assemble(self.FFA, bcs=self.bcA, tensor=self.resA)
            resB = assemble(self.FFB, bcs=self.bcB, tensor=self.resB)
        else:
            resA = assemble(self.FFA, bcs=self.bcA)
            self.resA = resA
            resB = assemble(self.FFB, bcs=self.bcB)
            self.resB = resB
        errA = sqrt(resA.vector().inner(resA.vector()))
        errB = sqrt(resB.vector().inner(resB.vector()))
        return errA, errB

    def computeAbsRelError(self):
        errA, errB = self.computeErrors()
        err_abs = max(errA, errB)
        err_rel = max(errA/self.errA0, errB/self.errB0)
        return err_abs, err_rel


    def solve(self):
        err_abs = err_rel = 1
        it = 0
        parprint(f"  Split it {it:4.0f}, err_abs={err_abs:4.2e}, err_rel={err_rel:4.2e}")
        while err_abs > self.atol and err_rel > self.rtol and it < self.maxit:
            it += 1
            self.solverA.solve()
            self.solverB.solve()
            err_abs, err_rel = self.computeAbsRelError()
            parprint(f"  Split it {it:4.0f}, err_abs={err_abs:4.2e}, err_rel={err_rel:4.2e}")



def pa_fun(_phi):
    return q1a * exp(q3a * _phi) + q2a * ln(q3a * _phi)

def dpa_fun(_phi):
    return q3a * q1a * exp(q3a * _phi) + q2a / _phi

def psi_vol(x, x0): 
    return Constant(k) * (x - x0) * ln(x/x0)

def p_sol(phis, phis0):
    return Constant(k) * (ln(phis/phis0) + (phis - phis0) / phis)


def psi_mech(F, dim):
    J = det(F)
    Fbar = J**(-1/dim) *F
    # Usyk,. mc Culloch 2002
    Cg = 0.88e3
    b = 6 # Just use isotropic value

    E = 0.5*(Fbar.T*Fbar - Identity(dim))

    Q = Constant(b) * E**2
    WP = 0.5*Constant(Cg)*(exp(Q)-1)

    WV = psi_vol(J, 1)
    return WP + WV



def psi_mech_heart(F, dim, fibers):
    f0, s0, n0 = fibers
    J = det(F)
    Fbar = J**(-1/dim) *F
    # Usyk,. mc Culloch 2002
    Cg = 0.88e3
    bf = 8
    bs = 6
    bn = 3
    bfs = 12
    bsn = 3
    bfn = 3

    E = 0.5*(Fbar.T*Fbar - Identity(dim))
    Eff, Efs, Efn = inner(E*f0, f0), inner(E*f0, s0), inner(E*f0, n0)
    Esf, Ess, Esn = inner(E*s0, f0), inner(E*s0, s0), inner(E*s0, n0)
    Enf, Ens, Enn = inner(E*n0, f0), inner(E*n0, s0), inner(E*n0, n0)

    Q = Constant(bf) * Eff**2 \
        + Constant(bs) * Ess**2 \
        + Constant(bn) * Enn**2 \
        + Constant(bfs) * 2.0 * Efs**2 \
        + Constant(bfn) * 2.0 * Efn**2 \
        + Constant(bsn) * 2.0 * Esn**2

    WP = 0.5*Constant(Cg)*(exp(Q)-1)

    WV = psi_vol(J, 1)
    return WP + WV


def getBackwardProblemResidual(time, dt, ramp_time, functions, theta, p_source, stationaryResidual, formulation, cardiac, imex, split, incompressibility):

    u = functions.u0
    v = functions.v
    phi = functions.phi
    phi0 = functions.phi0
    phi0_n = functions.phi0_n
    dim = u.ufl_shape[0]
    ramp = conditional(le(time, ramp_time), time/Constant(ramp_time), 1)
    if incompressibility:
        p = functions.p0
        q = functions.q

    # Mechanics
    I = Identity(dim)
    f = I + grad(u)
    j = det(f)
    F = variable(inv(f))
    J = 1/j

    phis = 1 - phi
    phis0 = 1 - phi0
    p_s = p_sol(J * phis, J * phis0) 
    if cardiac:
        P = diff(psi_mech_heart(F, dim, functions.fibers), F)
    else:
        P = diff(psi_mech(F,dim), F)
    pressure = p_s - p if incompressibility else p_s
    sigma = j * P * F.T + pressure * Identity(dim)

    Fmech =  inner(sigma, grad(v)) * dx
    if cardiac: 
        nn = FacetNormal(functions.mesh)
        k_perp = 2e5
        c_perp = 5e3
        ts_robin = outer(nn, nn)*k_perp*u + (Identity(3) - outer(nn, nn)) * \
            k_perp/10*u  # flip signs for inverse displacement

        Fmech += - ramp * Constant(-functions.p_endo) * dot(nn, v) * functions.ds_endo - dot(ts_robin, v) * functions.ds_epi


    # Incompressibility
    if incompressibility:
        Fmech += q * (J * (1 - phi) - (1 - phi0)) * dx

    # Porous media
    qa = functions.qa
    if imex: 
        phis0_n = 1 - phi0_n 
        p_s_n = p_sol(J * phis, J * phis0_n) 
        p0_n = pa_fun(J*phi0_n)
        press_n = p - p_s_n if incompressibility else -p_s
        pa_n = pa_fun(J*phi) - p0_n + press_n
        incr_phi0 = phi0 - phi0_n
        # We ignore p_s because it is zero... (hard coded)
        dpa_n = -dpa_fun(J*phi0_n) * J * incr_phi0
        #pa = pa_n + dpa_n
        pa = pa_n
    else:
        p0 = pa_fun(J*phi0)
        pa = pa_fun(J*phi) - p0 - pressure
    source = ramp * Constant(-theta) * (pa - Constant(p_source))
    if cardiac: 
        source += ramp * Constant(-theta) * (pa - Constant(p_sink))

    idt = 1/Constant(dt)
    dphi0dt = Constant(-1) * idt * (phi0 - phi0_n)

    # Backward problem with inverted time derivative
    if formulation == PRIMAL:
        if stationaryResidual:
            Fa = (dot(kappa * grad(pa), grad(qa))  - source * qa ) * dx
            return Fa
        else:
            Fa = dphi0dt * qa * dx + (dot(kappa * grad(pa), grad(qa))  - source * qa ) * dx
    elif formulation == MIXEDP:
        # Backward problem with inverted time derivative
        mu = functions.mu0
        Fmu = (mu - pa) * qa * dx 
        # Redefine source with mu
        source = ramp * Constant(-theta) * (mu - Constant(p_source))
        if cardiac: 
            source += ramp * Constant(-theta) * (mu - Constant(p_sink))
        if stationaryResidual:
            eta = functions.eta
            Fa = (dot(kappa * grad(mu), grad(eta))  - source * eta ) * dx
            return Fa + Fmu
        else:
            eta = Constant(dt) * functions.eta # Scale for symmetry
            Fa = dphi0dt * eta * dx + (dot(kappa * grad(mu), grad(eta))  - source * eta ) * dx
        Fa = Fa + Fmu
    else: # MIXEDU
        vu = functions.vu0
        vv = functions.vv
        FU = (dot(inv(kappa) * vu, vv) - pa * div(vv)) * dx
        if stationaryResidual:
            Fa = (div(vu) * qa  - source * qa ) * dx
            return Fa + FU
        else:
            Fa = dphi0dt * qa * dx + (div(vu) * qa  - source * qa ) * dx
        Fa = Fa + FU

    if split:
        return Fmech, Fa
    else:
        return Fmech +  Fa

def getBackwardProblemJacobianForm(time, dt, ramp_time, functions, theta, formulation, cardiac, imex):

    u = functions.u0
    p = functions.p0
    v = functions.v
    q = functions.q
    phi = functions.phi
    phi0 = functions.phi0
    phi0_n = functions.phi0_n
    dim = u.ufl_shape[0]
    ramp = conditional(le(time, ramp_time), time/Constant(ramp_time), 1)

    # Trial
    du = functions.du
    dp = functions.dp
    dphi = functions.dphi
    # Variables
    u = variable(u)
    phi0 = variable(phi0)
    p = variable(p)

    # Mechanics
    I = Identity(dim)
    f = I + grad(u)
    j = det(f)
    F = variable(inv(f))
    J = 1/j

    phis = 1 - phi
    phis0 = 1 - phi0_n if imex else 1-phi0
    p_s = p_sol(J * phis, J * phis0) 
    if cardiac:
        P = diff(psi_mech_heart(F, dim, functions.fibers), F)
    else:
        P = diff(psi_mech(F,dim), F)
    sigma = j * P * F.T + p_s * Identity(dim) # Add p -> dp in next step
    dsigma = dot(diff(sigma, u), du)

    Jmech =  inner(dsigma, grad(v))*dx  - dp * div(v) * dx
    if cardiac: 
        nn = FacetNormal(functions.mesh)
        k_perp = 2e5
        c_perp = 5e3
        ts_robin = outer(nn, nn)*k_perp*du + (Identity(dim) - outer(nn, nn)) * \
            k_perp/10*du  # flip signs for inverse displacement

        Jmech += - ramp * Constant(-functions.p_endo) * dot(nn, v) * functions.ds_endo - dot(ts_robin, v) * functions.ds_epi


    # Incompressibility
    Jincom = q * inner(diff(j,u), du) * phi0 * dx

    # Porous media
    qa = functions.qa
    p0 = pa_fun(J*phi0_n) if imex else pa_fun(J * phi0_n)
    pa = pa_fun(J*phi) - p0 + p - p_s
    dpa = diff(pa, phi0) * dphi 
    source = ramp * Constant(-theta) * dpa    
    if cardiac: 
        source += ramp * Constant(-theta) * dpa

    idt = 1/Constant(dt)
    dphi0dt = Constant(-1) * idt * dphi

    # Backward problem with inverted time derivative
    if formulation == PRIMAL:
        Ja = dphi0dt * qa * dx + (dot(kappa * grad(dpa), grad(qa))  - source * qa ) * dx
    elif formulation == MIXEDP:
        # Backward problem with inverted time derivative
        mu = functions.mu0
        mu = variable(mu)
        dmu = functions.dmu
        eta = functions.eta
        Jmu = (dmu - dpa) * qa * dx
        Ja = dphi0dt * eta * dx + (dot(kappa * grad(dmu), grad(eta))  - source * eta ) * dx
        Ja = Ja + Jmu
    else: # MIXEDU
        vu = functions.vu0
        vu = variable(vu)
        vv = functions.vv
        dvu = functions.dvu
        JU = (dot(inv(kappa) * dvu, vv) - dpa * div(vv)) * dx
        Ja = dphi0dt * qa * dx + (div(dvu) * qa  - source * qa ) * dx
        Ja = Ja + JU
    return Jmech + Jincom + Ja


def getForwardProblemResidual(time, dt, ramp_time, functions, theta, p_source, stationaryResidual, formulation, cardiac, imex, split, incompressibility):
    u = functions.u
    v = functions.v
    if incompressibility:
        p = functions.p
        q = functions.q
    phi = functions.phi
    phi0 = functions.phi0
    phi_n = functions.phi_n
    dim = u.ufl_shape[0]
    I = Identity(dim)    # Identity tensor
    F = I + grad(u)             # Deformation gradient
    F = variable(F)
    J = det(F)
    ramp = conditional(le(time, ramp_time), time/Constant(ramp_time), 1)

    phis = J - phi_n if imex else J - phi
    phis0 = J - phi0
    p_s = p_sol(phis, phis0) 

    # Mechanics
    if cardiac:
        P = diff(psi_mech_heart(F, dim, functions.fibers), F)
    else:
        P = diff(psi_mech(F,dim), F)

    pressure = p_s - p if incompressibility else p_s
    P += pressure * J * inv(F).T

    Fmech = inner(P, grad(v)) * dx
    if cardiac:
        nn = FacetNormal(functions.mesh)
        cof = J * inv(F).T
        cofnorm = sqrt(dot(cof*nn, cof*nn))
        NN = 1/cofnorm * cof*nn
        k_perp = 2e5
        c_perp = 5e3
        ts_robin = -outer(NN, NN)*k_perp*u - (Identity(3) - outer(NN, NN)) * \
            k_perp/10*u

        Fmech += - ramp * Constant(-functions.p_endo) * dot(cof * nn, v) * functions.ds_endo - dot(ts_robin, v) * cofnorm * functions.ds_epi

    # Incompressibility
    if incompressibility:
        Fmech += q * (J - phi - 1 + phi0) * dx

    # Porous media
    qa = functions.qa
    p0 = pa_fun(phi0)
    pa_fun_val = pa_fun(phi_n) if imex else pa_fun(phi)
    pa = pa_fun_val - p0 - pressure
    source = ramp * Constant(-theta) * (pa - Constant(p_source))
    if cardiac: 
        source += ramp * Constant(-theta) * (pa - Constant(p_sink))

    idt = 1/Constant(dt)
    dphidt = idt * (phi - phi_n)
    Kappa = J * inv(F) * kappa * inv(F).T
    if formulation == PRIMAL:
        if stationaryResidual:
            Fa = dot(Kappa * grad(pa), grad(qa)) * dx - J * source * qa * dx
            return Fa
        else:
            Fa = dphidt * qa * dx + dot(Kappa * grad(pa), grad(qa)) * dx - J * source * qa * dx
    elif formulation == MIXEDP: 
        mu = functions.mu
        eta = functions.eta
        Fmu = (mu - pa) * qa * dx
        if stationaryResidual:
            Fa = dot(Kappa * grad(mu), grad(eta)) * dx - J * source * eta * dx
            return Fa + Fmu
        else:
            Fa = dphidt * eta * dx + dot(Kappa * grad(mu), grad(eta)) * dx - J * source * eta * dx
        Fa += Fmu
    else: 
        vu = functions.vu
        vv = functions.vv
        FU = (dot( inv(Kappa) * vu, vv) - pa * div(vv)) * dx
        if stationaryResidual:
            Fa = (div(vu) * qa  - source * qa ) * dx
            return Fa + FU
        else:
            Fa = dphidt * qa * dx + (div(vu) * qa  - source * qa ) * dx
        Fa += FU
    if split:
        return Fmech, Fa
    else:
        return Fmech + Fa


def getBackwardProblem(time, dt, ramp_time, theta, p_source, functions, formulation, stationaryResidual=False, cardiac=False, imex=False, split=False, incompressibility=True):
    if cardiac and not functions.fibers:
        assert False, "If cardiac, fibers must be set"
    FF = getBackwardProblemResidual(time, dt, ramp_time, functions, theta, p_source, stationaryResidual, formulation, cardiac, imex, split, incompressibility)
    return FF

def getBackwardProblemJacobian(time, dt, ramp_time, theta, functions, formulation, cardiac=False, imex=False):
    if cardiac and not functions.fibers:
        assert False, "If cardiac, fibers must be set"
    FF = getBackwardProblemJacobianForm(time, dt, ramp_time, functions, theta, formulation, cardiac, imex)
    return FF

def getForwardProblem(time, dt, ramp_time, theta, p_source, functions, formulation, stationaryResidual=False, cardiac=False, imex=False, split=False, incompressibility=True):
    if cardiac and not functions.fibers:
        assert False, "If cardiac, fibers must be set"
    FF = getForwardProblemResidual(time, dt, ramp_time, functions, theta, p_source, stationaryResidual, formulation, cardiac, imex, split, incompressibility)
    return FF


