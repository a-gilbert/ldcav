"""Algorithms and methods to solve a lid driven Cavity problem"""
import matplotlib.pyplot as plt
import numpy as np 


class SimpleNSCavity(object):

    def __init__(self, hs, tstop, uws, Re, bco=1, meth='sor', dtype=np.longdouble):
        self.Re = Re
        self.uws = uws
        self.hs = hs
        self.ms = np.array([(1/self.hs[0])-1, (1.0/self.hs[1])-1, 
                           tstop/(self.hs[2])], dtype=np.int16)
        #intitialize grid functions
        self.vor = np.zeros((3, (self.ms[0] + 2)*(self.ms[1] + 2)), dtype=dtype)
        self.psi = np.zeros(((self.ms[0] + 2)*(self.ms[1] + 2),), dtype=dtype)
        self.t = 0.0
        self.nstep = 0
        self.set_bcs_o1()
        self.save_plot()
        #initialize second initial state
        self.euler_step()
        while self.t < tstop:
            #self.euler_step()
            self.cnlf_step()


    def save_plot(self):
        x = np.arange(0, self.ms[0]+2)*self.hs[0]
        y = np.flip(np.arange(0, self.ms[1]+2)*self.hs[1])
        Xm, Ym = np.meshgrid(x, y)
        S = np.reshape(self.psi, (self.ms[1] + 2, self.ms[0] + 2))
        U = np.gradient(S, self.hs[0], self.hs[1], edge_order=2)
        plt.close('all')
        plt.figure(figsize=(10, 10))
        plt.subplot(411)
        i = plt.imshow(-1*U[0], extent=(x[0], x[-1], y[-1], y[0]), Cmap='RdBu', 
                        vmin=-1.0, vmax=1.0)
        plt.colorbar(i)
        ax = plt.gca()
        ax.set_title(r"$U_{x}$")
        plt.subplot(412)
        i = plt.imshow(-1*U[1], extent=(x[0], x[-1], y[-1], y[0]), Cmap='RdBu', 
                        vmin=-1.0, vmax=1.0)
        plt.colorbar(i)
        ax = plt.gca()
        ax.set_title(r"$U_{y}$")
        
        i = plt.imshow(S, extent=(x[0], x[-1], y[-1], y[0]))
        plt.streamplot(Xm, Ym, -1*U[0].astype(np.float64), -1*U[1].astype(np.float64))
        plt.hlines(0.5, 0.5, 1)
        #plt.vlines(0.5, 0, 0.5)
        plt.colorbar(i)
        plt.title('Driven Cavity with Re=%.02f, t=%.02f' % (self.Re, self.t))
        plt.savefig('cavity%04d.png' % self.nstep)


    def _move_vor(self):
        """Move the most current voricity back, to enable updating."""
        self.vor[1, :] = self.vor[0, :]


    def _set_euler_rhs(self):
        a = self.hs[2]/(4.0*self.hs[0]*self.hs[1])
        Nc = self.ms[0] + 2
        Nr = self.ms[1] + 2
        #iterate over each row of system (rows=lines of constant y)
        #formulate such that vor[0, 0] = vor(x=0, y=1)
        for j in range(1, Nr-1):
            self.vor[2, j*Nc+1:(j+1)*Nc-1] = -1*a*(self.psi[(j-1)*Nc+1:j*Nc-1] -
                                        self.psi[(j+1)*Nc+1:(j+2)*Nc-1]) *\
                                        (self.vor[1, j*Nc+2:(j+1)*Nc] -\
                                         self.vor[1, j*Nc:(j+1)*Nc - 2])
            self.vor[2, j*Nc+1:(j+1)*Nc-1] +=  a*(self.psi[j*Nc+2:(j+1)*Nc] - \
                                             self.psi[j*Nc:(j+1)*Nc-2])*(
                                         self.vor[1, (j-1)*Nc+1:j*Nc-1] - \
                                         self.vor[1, (j+1)*Nc+1:(j+2)*Nc-1])
            self.vor[2, j*Nc+1:(j+1)*Nc-1] += self.vor[1, j*Nc+1:(j+1)*Nc-1]
    

    def _set_cnlf_rhs(self):
        #diffusion coefficients
        dx = self.hs[2]/(self.Re*self.hs[0]*self.hs[0])
        dy = self.hs[2]/(self.Re*self.hs[1]*self.hs[1])
        dc = 1 - 2*dx - 2*dy
        #advection coefficient
        a = self.hs[2]/(2.0*self.hs[0]*self.hs[1])
        Nr = self.ms[1] + 2
        Nc = self.ms[0] + 2
        for j in range(1, Nr-1):
            #diffusion/n-1 value 
            lhs = self.vor[2, j*Nc+1:(j+1)*Nc-1]
            lhs[:] = dy*self.vor[1, (j-1)*Nc+1:j*Nc-1]
            lhs[:] += dx*self.vor[1, j*Nc:(j+1)*Nc-2]
            lhs[:] += dc*self.vor[1, j*Nc+1:(j+1)*Nc-1]
            lhs[:] += dx*self.vor[1, j*Nc+2:(j+1)*Nc]
            lhs[:] += dy*self.vor[1, (j+1)*Nc+1:(j+2)*Nc-1]
            #advection
            lhs[:] -= a*(self.psi[(j-1)*Nc+1:j*Nc-1] - 
                      self.psi[(j+1)*Nc+1:(j+2)*Nc-1])*\
                      (self.vor[0, j*Nc+2:(j+1)*Nc] - 
                      self.vor[0, j*Nc:(j+1)*Nc-2])
            lhs[:] += a*(self.psi[j*Nc+2:(j+1)*Nc]-self.psi[j*Nc:(j+1)*Nc-2])*\
                      (self.vor[0, (j-1)*Nc+1:j*Nc-1] -
                       self.vor[0, (j+1)*Nc+1:(j+2)*Nc-1])


    def vor_solve(self, sor=True):
        print('vor solve')
        cy = self.hs[0]/(self.Re*self.hs[1]*self.hs[1])
        cx = self.hs[0]/(self.Re*self.hs[0]*self.hs[0])
        cdinv = 1.0/(1 + 2*cy + 2*cx)
        cy = cy*cdinv
        cx = cx*cdinv
        if sor == True:
            a = 2.0 - 2.0*np.pi*np.sqrt(self.hs[0]*self.hs[1])
        else:
            a = 1.0
        #scale coefficients by a*cdinv, and scale rhs
        cy = cy*a
        cx = cx*a
        self.vor[2, :] = a*cdinv*self.vor[2, :]
        #machine tolerance
        #if self.vor.dtype == np.longdouble:
        #    tol = 10*np.finfo(np.float64).eps
        #elif self.vor.dtype == np.float64:
        #    tol = np.finfo(np.float32).eps
        #elif self.vor.dtype == np.float16:
        #    tol = np.finfo(np.float16).eps
        tol = 0.5*(self.hs[0]**2 + self.hs[1]**2)
        r = tol + 0.1*tol
        Nr = self.ms[1] + 2
        Nc = self.ms[0] + 2
        while r > tol:
            r = tol - 0.1*tol
            for j in range(1, Nr-1):
                for i in range(1, Nc-1):
                    vprev = self.vor[0, i + Nc*j]
                    self.vor[0, i + Nc*j] = (1-a)*vprev + self.vor[2, i + Nc*j]
                    self.vor[0, i + Nc*j] += cy*(self.vor[0, i+Nc*(j-1)] + 
                                             self.vor[0, i+Nc*(j+1)])
                    self.vor[0, i + Nc*j] += cx*(self.vor[0, i+1+Nc*j] + 
                                             self.vor[0, i-1+Nc*j])
                    vprev -= self.vor[0, i + Nc*j]
                    vprev = np.abs(vprev)
                    if vprev > r:
                        r = vprev
            print(r)


    def stream_solve(self, sor=True):
        print('stream solve')
        cy = 1/(self.hs[1]*self.hs[1])
        cx = 1/(self.hs[0]*self.hs[0])
        cdinv = -1.0/(2*cx + 2*cy)
        cy = cdinv*cy
        cx = cdinv*cx
        if sor == True:
            a = 2.0 - 2.0*np.pi*np.sqrt(self.hs[0]*self.hs[1])
        else:
            a = 1.0
        cy = a*cy
        cx = a*cx
        #machine tolerance
        #if self.vor.dtype == np.longdouble:
        #    tol = 10*np.finfo(np.float64).eps
        #elif self.vor.dtype == np.float64:
        #    tol = np.finfo(np.float32).eps
        #elif self.vor.dtype == np.float16:
        #    tol = np.finfo(np.float16).eps
        tol = 0.5*(self.hs[0]**2 + self.hs[1]**2)
        r = tol + 0.1*tol
        Nr = self.ms[1] + 2
        Nc = self.ms[0] + 2
        while r > tol:
            r = tol - 0.1*tol
            for j in range(1, Nr-1):
                for i in range(1, Nc-1):
                    sprev = self.psi[i + Nc*j]
                    self.psi[i + Nc*j] = (1-a)*sprev - a*cdinv*self.vor[0, i+Nc*j]
                    self.psi[i + Nc*j] -= cy*(self.psi[i + Nc*(j-1)] + 
                                     self.psi[i + Nc*(j+1)])
                    self.psi[i + Nc*j] -= cx*(self.psi[i+1+Nc*j] + 
                                          self.psi[i-1+Nc*j])
                    sprev -= self.psi[i + Nc*j]
                    sprev = np.abs(sprev)
                    if sprev > r:
                        r = sprev
            print(r)


    def set_bcs_o1(self):
        """Set the vorticity boundary condtions, to first order accuracy."""
        c1 = 2.0/self.hs[:-1]
        c2 = c1/self.hs[:-1]
        Nc = self.ms[0] + 2
        Nr = self.ms[1] + 2
        #set top boundary condition
        self.vor[0, 1:Nc-1] = c2[1]*(self.psi[1:Nc-1]-self.psi[Nc+1:2*Nc-1]) \
                              - c1[1]*self.uws[0]
        #set right boundary conditon
        self.vor[0, 2*Nc-1:-1:Nc] = c2[0]*(self.psi[2*Nc-1:-1:Nc] - \
                                    self.psi[2*Nc-2:-2:Nc]) + c1[0]*self.uws[1]
        #set bottom boundary condition
        self.vor[0, Nc*(Nr-1) + 1:-1] = c2[1]*(self.psi[Nc*(Nr-1) + 1:-1] - \
                                        self.psi[Nc*(Nr-2) + 1:Nc*(Nr-1)-1]) +\
                                        c1[1]*self.uws[2]
        #set left boundary condition
        self.vor[0, Nc:(Nr-1)*Nc:Nc] = c2[0]*(self.psi[Nc:(Nr-1)*Nc:Nc] - \
                                       self.psi[Nc+1:(Nr-1)*Nc+1:Nc]) -\
                                       c1[0]*(self.uws[3])

    def euler_step(self):
        print('euler step')
        #move vorticity 
        self._move_vor()
        #set boundaries for second step
        self.set_bcs_o1()
        #set right hand side
        self._set_euler_rhs() 
        #solve for vorticity
        self.vor_solve()
        #solve for stream function
        self.stream_solve()
        self.t += self.hs[2]
        self.nstep += 1
        self.save_plot()

    def cnlf_step(self):
        print('cnlf step')
        self._set_cnlf_rhs()
        self.set_bcs_o1()
        self._move_vor()
        self.vor_solve(sor=True)
        self.stream_solve(sor=True)
        #self.set_bcs_o1()
        self.t += self.hs[2]
        self.nstep += 1
        self.save_plot()


class ComplexNSCavity(object):

    def __init__(self, hs, tstop, uws, Re, bco=1, sor=True, dtype=np.longdouble):
        self.Re = Re
        self.uws = uws
        self.hs = hs
        self.ms = np.array([(1/self.hs[0])-1, (1.0/self.hs[1])-1, 
                           tstop/(self.hs[2])], dtype=np.int16)
        #intitialize grid functions
        self.vor = np.zeros((3, (self.ms[0] + 2)*(self.ms[1] + 2)), dtype=dtype)
        self.psi = np.zeros(((self.ms[0] + 2)*(self.ms[1] + 2)), dtype=dtype)
        self.t = 0.0
        self.nstep = 0
        self.sniter = []
        self.vniter = []
        self.linf_dt = []
        self.set_complex_bcs_o1()
        self.save_plot()
        #initialize second initial state
        if sor:
            self.euler_step()
            while self.t < tstop:
                self.cnlf_step()
        else:
            self.euler_step(sor=False)
            while self.t < tstop:
                self.cnlf_step(sor=False)


    def save_plot(self, sor=True):
        plt.rcParams.update({'font.size': 22})
        x = np.arange(0, self.ms[0]+2)*self.hs[0]
        y = np.flip(np.arange(0, self.ms[1]+2)*self.hs[1])
        Xm, Ym = np.meshgrid(x, y)
        S = np.reshape(self.psi, (self.ms[1] + 2, self.ms[0] + 2))
        V = np.reshape(self.vor[0], (self.ms[1] + 2, self.ms[0]+2))
        U = np.gradient(S, self.hs[0], self.hs[1], edge_order=2)
        plt.close('all')
        f = plt.figure(figsize=(12, 40))
        #ux
        #uxmax = np.max(np.abs(U[0]))
        plt.subplot(411)
        i = plt.imshow(-1*U[0], extent=(x[0], x[-1], y[-1], y[0]), Cmap='RdBu', 
                        vmin=-1, vmax=1)
        Nrh = np.floor((self.ms[1] + 2)/2)
        Nch = np.floor((self.ms[0] + 2)/2)
        plt.hlines(Nrh*self.hs[1], self.hs[0]*Nch, self.hs[0]*(self.ms[0]+2))
        plt.vlines(Nch*self.hs[0], 0, Nrh*self.hs[1])
        plt.colorbar(i)
        ax = plt.gca()
        ax.set_title(r"$U_{x}$")
        #uy 
        #uymax = np.max(np.abs(U[1]))
        plt.subplot(412)
        i = plt.imshow(-1*U[1], extent=(x[0], x[-1], y[-1], y[0]), Cmap='RdBu', 
                        vmin=-1, vmax=1)
        Nrh = np.floor((self.ms[1] + 2)/2)
        Nch = np.floor((self.ms[0] + 2)/2)
        plt.hlines(Nrh*self.hs[1], self.hs[0]*Nch, self.hs[0]*(self.ms[0]+2))
        plt.vlines(Nch*self.hs[0], 0, Nrh*self.hs[1])
        plt.colorbar(i)
        ax = plt.gca()
        ax.set_title(r"$U_{y}$")
        #stream
        plt.subplot(413) 
        i = plt.imshow(S, extent=(x[0], x[-1], y[-1], y[0]))
        plt.streamplot(Xm, Ym, -1*U[0].astype(np.float64), -1*U[1].astype(np.float64))
        Nrh = np.floor((self.ms[1] + 2)/2)
        Nch = np.floor((self.ms[0] + 2)/2)
        plt.hlines(Nrh*self.hs[1], self.hs[0]*Nch, self.hs[0]*(self.ms[0]+2))
        plt.vlines(Nch*self.hs[0], 0, Nrh*self.hs[1])
        plt.colorbar(i)
        ax = plt.gca()
        ax.set_title(r'$\Psi$')
        #vor
        plt.subplot(414) 
        i = plt.imshow(V, extent=(x[0], x[-1], y[-1], y[0]))
        plt.streamplot(Xm, Ym, -1*U[0].astype(np.float64), -1*U[1].astype(np.float64))
        Nrh = np.floor((self.ms[1] + 2)/2)
        Nch = np.floor((self.ms[0] + 2)/2)
        plt.hlines(Nrh*self.hs[1], self.hs[0]*Nch, self.hs[0]*(self.ms[0]+2))
        plt.vlines(Nch*self.hs[0], 0, Nrh*self.hs[1])
        plt.colorbar(i)
        ax = plt.gca()
        ax.set_title(r'$\omega$')
        plt.tight_layout()
        f.suptitle('Driven Cavity with Re=%.02f, t=%.02f' % (self.Re, self.t))
        if sor:
            plt.savefig('SOR_cavity%04d.png' % self.nstep)
        else:
            plt.savefig('GS_cavity%04d.png' % self.nstep)


    def _move_vor(self):
        """Move the most current voricity back, to enable updating."""
        self.vor[1, :] = self.vor[0, :]


    def _set_euler_rhs(self):
        a = self.hs[2]/(4.0*self.hs[0]*self.hs[1])
        Nc = self.ms[0] + 2
        Nr = self.ms[1] + 2
        #iterate over each row of system (rows=lines of constant y)
        #formulate such that vor[0, 0] = vor(x=0, y=1)
        for j in range(1, Nr-1):
            self.vor[2, j*Nc+1:(j+1)*Nc-1] = -1*a*(self.psi[(j-1)*Nc+1:j*Nc-1] -
                                        self.psi[(j+1)*Nc+1:(j+2)*Nc-1]) *\
                                        (self.vor[1, j*Nc+2:(j+1)*Nc] -\
                                         self.vor[1, j*Nc:(j+1)*Nc - 2])
            self.vor[2, j*Nc+1:(j+1)*Nc-1] +=  a*(self.psi[j*Nc+2:(j+1)*Nc] - \
                                             self.psi[j*Nc:(j+1)*Nc-2])*(
                                         self.vor[1, (j-1)*Nc+1:j*Nc-1] - \
                                         self.vor[1, (j+1)*Nc+1:(j+2)*Nc-1])
            self.vor[2, j*Nc+1:(j+1)*Nc-1] += self.vor[1, j*Nc+1:(j+1)*Nc-1]
    

    def _set_cnlf_rhs(self):
        #diffusion coefficients
        dx = self.hs[2]/(self.Re*self.hs[0]*self.hs[0])
        dy = self.hs[2]/(self.Re*self.hs[1]*self.hs[1])
        dc = 1 - 2*dx - 2*dy
        #advection coefficient
        a = self.hs[2]/(2.0*self.hs[0]*self.hs[1])
        Nr = self.ms[1] + 2
        Nc = self.ms[0] + 2
        Nrh = int(np.floor(Nr/2))
        Nch = int(np.floor(Nc/2))
        for j in range(1, Nr-1):
            lhs = self.vor[2]
            #diffusion/n-1 value 
            lhs = self.vor[2, j*Nc+1:(j+1)*Nc-1]
            lhs[:] = dy*self.vor[1, (j-1)*Nc+1:j*Nc-1]
            lhs[:] += dx*self.vor[1, j*Nc:(j+1)*Nc-2]
            lhs[:] += dc*self.vor[1, j*Nc+1:(j+1)*Nc-1]
            lhs[:] += dx*self.vor[1, j*Nc+2:(j+1)*Nc]
            lhs[:] += dy*self.vor[1, (j+1)*Nc+1:(j+2)*Nc-1]
            #advection
            lhs[:] -= a*(self.psi[(j-1)*Nc+1:j*Nc-1] - 
                      self.psi[(j+1)*Nc+1:(j+2)*Nc-1])*\
                      (self.vor[0, j*Nc+2:(j+1)*Nc] - 
                      self.vor[0, j*Nc:(j+1)*Nc-2])
            lhs[:] += a*(self.psi[j*Nc+2:(j+1)*Nc]-self.psi[j*Nc:(j+1)*Nc-2])*\
                      (self.vor[0, (j-1)*Nc+1:j*Nc-1] -
                       self.vor[0, (j+1)*Nc+1:(j+2)*Nc-1])


    def vor_solve(self, sor=True):
        print('vor solve')
        cy = self.hs[0]/(self.Re*self.hs[1]*self.hs[1])
        cx = self.hs[0]/(self.Re*self.hs[0]*self.hs[0])
        cdinv = 1.0/(1 + 2*cy + 2*cx)
        cy = cy*cdinv
        cx = cx*cdinv
        if sor == True:
            a = 2.0 - 2.0*np.pi*np.sqrt(self.hs[0]*self.hs[1])
        else:
            a = 1.0
        #scale coefficients by a*cdinv, and scale rhs
        cy = cy*a
        cx = cx*a
        self.vor[2, :] = a*cdinv*self.vor[2, :]
        #machine tolerance
        #if self.vor.dtype == np.longdouble:
        #    tol = 10*np.finfo(np.float64).eps
        #elif self.vor.dtype == np.float64:
        #    tol = np.finfo(np.float32).eps
        #elif self.vor.dtype == np.float16:
        #    tol = np.finfo(np.float16).eps
        tol = 0.5*(self.hs[0]**2 + self.hs[1]**2)
        r = tol + 0.1*tol
        Nr = self.ms[1] + 2
        Nc = self.ms[0] + 2
        Nrh = int(np.floor(Nr/2.0))
        Nch = int(np.floor(Nc/2.0))
        niter = 0
        while r > tol:
            niter += 1
            r = tol - 0.1*tol
            for j in range(1, Nr-1):
                if j < Nrh:
                    for i in range(1, Nc-1):
                        vprev = self.vor[0, i + Nc*j]
                        self.vor[0, i + Nc*j] = (1-a)*vprev + self.vor[2, i + Nc*j]
                        self.vor[0, i + Nc*j] += cy*(self.vor[0, i+Nc*(j-1)] + 
                                                 self.vor[0, i+Nc*(j+1)])
                        self.vor[0, i + Nc*j] += cx*(self.vor[0, i+1+Nc*j] + 
                                                 self.vor[0, i-1+Nc*j])
                        vprev -= self.vor[0, i + Nc*j]
                        vprev = np.abs(vprev)
                        if vprev > r:
                            r = vprev
                elif j>= Nrh:
                    for i in range(1, Nch-1):
                        vprev = self.vor[0, i + Nc*j]
                        self.vor[0, i + Nc*j] = (1-a)*vprev + self.vor[2, i + Nc*j]
                        self.vor[0, i + Nc*j] += cy*(self.vor[0, i+Nc*(j-1)] + 
                                                 self.vor[0, i+Nc*(j+1)])
                        self.vor[0, i + Nc*j] += cx*(self.vor[0, i+1+Nc*j] + 
                                                 self.vor[0, i-1+Nc*j])
                        vprev -= self.vor[0, i + Nc*j]
                        vprev = np.abs(vprev)
                        if vprev > r:
                            r = vprev
            print(r)
        self.vniter.append(niter)


    def stream_solve(self, sor=True):
        print('stream solve')
        cy = 1/(self.hs[1]*self.hs[1])
        cx = 1/(self.hs[0]*self.hs[0])
        cdinv = -1.0/(2*cx + 2*cy)
        cy = cdinv*cy
        cx = cdinv*cx
        if sor == True:
            a = 2.0 - 2.0*np.pi*np.sqrt(self.hs[0]*self.hs[1])
        else:
            a = 1.0
        cy = a*cy
        cx = a*cx
        #machine tolerance
        #if self.vor.dtype == np.longdouble:
        #    tol = 10*np.finfo(np.float64).eps
        #elif self.vor.dtype == np.float64:
        #    tol = np.finfo(np.float32).eps
        #elif self.vor.dtype == np.float16:
        #    tol = np.finfo(np.float16).eps
        tol = 0.5*(self.hs[0]**2 + self.hs[1]**2)
        r = tol + 0.1*tol
        Nr = self.ms[1] + 2
        Nc = self.ms[0] + 2
        Nrh = int(np.floor(Nr/2.0))
        Nch = int(np.floor(Nc/2.0))
        niter = 0
        while r > tol:
            niter += 1
            r = tol - 0.1*tol
            for j in range(1, Nr-1):
                if j < Nrh:
                    for i in range(1, Nc-1):
                        sprev = self.psi[i + Nc*j]
                        self.psi[i + Nc*j] = (1-a)*sprev - a*cdinv*self.vor[0, i+Nc*j]
                        self.psi[i + Nc*j] -= cy*(self.psi[i + Nc*(j-1)] + 
                                         self.psi[i + Nc*(j+1)])
                        self.psi[i + Nc*j] -= cx*(self.psi[i+1+Nc*j] + 
                                              self.psi[i-1+Nc*j])
                        sprev -= self.psi[i + Nc*j]
                        sprev = np.abs(sprev)
                        if sprev > r:
                            r = sprev
                elif j >= Nrh:
                    for i in range(1, Nch-1):
                        sprev = self.psi[i + Nc*j]
                        self.psi[i + Nc*j] = (1-a)*sprev - a*cdinv*self.vor[0, i+Nc*j]
                        self.psi[i + Nc*j] -= cy*(self.psi[i + Nc*(j-1)] + 
                                         self.psi[i + Nc*(j+1)])
                        self.psi[i + Nc*j] -= cx*(self.psi[i+1+Nc*j] + 
                                              self.psi[i-1+Nc*j])
                        sprev -= self.psi[i + Nc*j]
                        sprev = np.abs(sprev)
                        if sprev > r:
                            r = sprev
 
            print(r)
        self.sniter.append(niter)


    def set_complex_bcs_o1(self):
        """Set the vorticity boundary condtions, to first order accuracy."""
        c1 = 2.0/self.hs[:-1]
        c2 = c1/self.hs[:-1]
        Nc = self.ms[0] + 2
        Nr = self.ms[1] + 2
        Nrh = int(np.floor(Nr/2))
        Nch = int(np.floor(Nc/2))
        vview = np.reshape(self.vor[0], (Nr, Nc))
        pview = np.reshape(self.psi, (Nr, Nc))
        #set top boundary condition
        vview[0, 1:Nc-1] = c2[1]*(pview[0, 1:Nc-1]-pview[1, 1:Nc-1]) -\
                                 c1[1]*self.uws[0]
        #set right upper wall bc
        vview[1:Nrh, -1] = c2[0]*(pview[1:Nrh, -1] - pview[1:Nrh, -2]) -\
                                 c1[0]*self.uws[1]
        #set horizontal middle wall (y=0.5, 0.5<= x<= 1) bc:
        vview[Nrh, Nch:-1] = c2[1]*(pview[Nrh, Nch:-1] - pview[Nrh-1, Nch:-1]) +\
                                   c1[0]*self.uws[2]
        #set vertical middle wall
        vview[Nrh:-1, Nch] = c2[0]*(pview[Nrh:-1, Nch] - pview[Nrh:-1, Nch-1]) -\
                                   c1[0]*self.uws[3]
        #set bottom left wall
        vview[-1, 1:Nch] = c2[1]*(pview[-1, 1:Nch] - pview[-2, 1:Nch]) +\
                                 c1[1]*self.uws[4]
        #set left wall
        vview[1:-1, 0] = c2[0]*(pview[1:-1, 0] - pview[1:-1, 1]) +\
                               c1[0]*self.uws[5]
        vview[Nrh+1:, Nch+1:] = 0
        pview[Nrh+1:, Nch+1:] = 0

    def euler_step(self, sor=True):
        print('euler step')
        #move vorticity 
        self._move_vor()
        #set boundaries for second step
        self.set_complex_bcs_o1()
        #set right hand side
        self._set_euler_rhs() 
        #solve for vorticity
        self.vor_solve(sor=sor)
        #solve for stream function
        self.stream_solve(sor=sor)
        self.linf_dt.append(np.max(np.abs(self.vor[0] - self.vor[1])))
        self.t += self.hs[2]
        self.nstep += 1
        self.save_plot(sor=sor)

    def cnlf_step(self, sor=True):
        print('cnlf step')
        self._set_cnlf_rhs()
        self.set_complex_bcs_o1()
        self._move_vor()
        self.vor_solve(sor=sor)
        self.stream_solve(sor=sor)
        self.linf_dt.append(np.max(np.abs(self.vor[0] - self.vor[1])))
        #self.set_bcs_o1()
        self.t += self.hs[2]
        self.nstep += 1
        self.save_plot(sor=sor)
