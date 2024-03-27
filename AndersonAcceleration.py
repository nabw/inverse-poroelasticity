import numpy as np
from petsc4py import PETSc
from mpi4py import MPI


class AndersonAcceleration:

    def __init__(self, x0, order, delay, restart=False):
        self.order = order
        self.delay = delay
        self.restart = restart
        self.k = 0
        self.F = []
        self.X = []
        self.G = []
        self.F0 = []  # For global vectors
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.d_fk0 = None
        self.fk0 = None

        self.xk = PETSc.Vec()
        self.gk = PETSc.Vec()
        self.fk = PETSc.Vec()
        self.gk_n = PETSc.Vec()
        self.fk_n = PETSc.Vec()
        self.delta_gk = PETSc.Vec()
        self.delta_fk = PETSc.Vec()

        # Initialize vectors
        for v in (self.xk, self.fk, self.gk, self.delta_fk, self.delta_gk, self.gk_n, self.fk_n):
            with x0.dat.vec_ro as w:
                w.copy(v)
            v.zeroEntries()
        # Init global vectors and scatterer
        if self.size>1: 
            with x0.dat.vec_ro as v:
                self.scatter, aux = PETSc.Scatter.toZero(v)
                self.d_fk0 = aux.copy()
                self.fk0 = aux.copy()
        # Initial vector
        with x0.dat.vec_ro as v:
            v.copy(self.xk)


    #def append_vectors(self, dgk, dfk):
    def append_vectors(self, dgk, dfk):
        self.G.append(dgk.copy())
        self.F.append(dfk.copy())
        if len(self.F) > self.order:
            self.F.pop(0)
            self.G.pop(0)

    def reset(self):
        self.F = []
        self.F0 = []
        self.G = []
        for v in (self.xk, self.fk, self.gk, self.delta_fk, self.delta_gk, self.gk_n, self.fk_n):
            v.zeroEntries()
        if self.size>1 and self.rank == 0:
            for v in (self.fk0, self.d_fk0):
                v.zeroEntries()
        self.k = 0

    # Result is stored in gk
    def get_next_vector(self, gk, fk=None):


        # Update internal vectors
        with gk.dat.vec_ro as v:
            v.copy(self.gk)
        if fk:
            with fk.dat.vec_ro as v:
                v.copy(self.fk)
        else:
            self.gk.copy(self.fk)
            self.fk.axpy(-1, self.xk)

        mk = min(self.k-self.delay, self.order)
        if mk > 0:  # If order>0 and k>0
            # Build increments
            self.fk.copy(self.delta_fk)
            self.gk.copy(self.delta_gk)
            self.delta_fk.axpy(-1, self.fk_n)
            self.delta_gk.axpy(-1, self.gk_n)
            
            # Iterate only if increment is big enough 
            if self.delta_fk.norm() < 1e-12:
                self.k -= 1
            else:
                self.append_vectors(self.delta_gk, self.delta_fk)
                if self.size > 1: # Parallel
                    self.scatter.scatter(self.delta_fk, self.d_fk0)
                    self.scatter.scatter(self.fk, self.fk0)
                    # Process only on first core, then scatter alpha
                    if self.rank == 0:
                        self.F0.append(self.d_fk0.copy())
                        if len(self.F0) > self.order:
                            self.F0.pop(0)
                        F = np.vstack(self.F0).T
                        alpha = np.linalg.lstsq(F, self.fk0, rcond=None)[0]
                    else:
                        alpha = None
                    alpha = self.comm.bcast(alpha, root=0)
                else: # Serial
                    F = np.vstack(self.F).T
                    out = np.linalg.lstsq(F, self.fk, rcond=None)
                    alpha = out[0]
                norm = np.linalg.norm(alpha)
                restart = False
                if norm > self.order and self.restart:
                    print(f"ANDERSON: RESTART")
                    alpha[:] = 0.0
                    restart = True
                        

                # Update xk
                self.gk.copy(self.xk)
                for i in range(mk):
                    self.xk.axpy(-alpha[i], self.G[i])
                with gk.dat.vec as v:
                    self.xk.copy(v)
                if restart:
                    for f, g in zip(self.F, self.G):
                        f.destroy()
                        g.destroy()
                    self.F = []
                    self.G = []
                    self.k = self.delay
        else:
            # Update xk
            self.gk.copy(self.xk)

        self.k += 1
        self.gk.copy(self.gk_n)
        self.fk.copy(self.fk_n)
