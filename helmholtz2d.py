# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 13:29:04 2021

@author: aidan
"""
 # NOTE: Dirichlet conditions override Neumann
 # NOTE: Constraints at the end of the list override ones at the start
 
import numpy as np
import os
from collections import OrderedDict
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from python_lib.helmholtz2d import helmholtz2d as pythonHelmholtz2d
#from comsol_lib.helmholtz2d import helmholtz2d as comsolHelmholtz2d

class helmholtz2d(pythonHelmholtz2d):
    """ Inserts Dirichlet constraints into unconstrained matrices and solves """
    def __init__(self, f, c, rho = 1, domain = 'all', coupling_masks = [], dirichlet_constraint_functions = [], neumann_constraint_functions = [], meshName = "FineMesh", builder = 'python', reorder = None):
        self.f = f
        self.c = c
        self.rho = rho # density
        
        if builder == 'python':
            pythonHelmholtz2d.__init__(self, domain = domain, meshName = meshName, coupling_masks = coupling_masks, neumann_constraints = neumann_constraint_functions, reorder = reorder)
        #if builder == 'comsol':
        #    comsolHelmholtz2d.__init__(self)
        
        # self.neumann_constraints = OrderedDict()
        # self.neumann_constraint_functions = neumann_constraint_functions
        
        # for n in self.neumann_constraint_functions:
        #     print("added constraint : {}".format(n.__name__)) ## TEMP
        #     self.add_neumann_constraint(n)
            
        self.dirichlet_constraints = OrderedDict()
        self.dirichlet_constraint_functions = dirichlet_constraint_functions
        
        for c in self.dirichlet_constraint_functions:
            print("added constraint : {}".format(c.__name__)) ## TEMP
            self.add_dirichlet_constraint(c)
        
        ## mask and vals for all constraints (all Dirchlet at this stage)
        self.all_dirichlet_constraints_mask = np.zeros(len(self.X))
        all_dirichlet_constraints_vals_full = np.zeros(len(self.X))
        
        # self.all_neumann_constraints_mask = np.zeros(len(self.X))
        # self.all_neumann_constraints = np.zeros(len(self.X))
        
        for c in self.dirichlet_constraints.keys():
            mask = self.dirichlet_constraints[c]['mask']
            vals = self.dirichlet_constraints[c]['values']
            
            self.all_dirichlet_constraints_mask += mask # counts num constraints per dof. Zero if not a boundary dof
            all_dirichlet_constraints_vals_full[mask] = vals
            
        ## TEMP - delete the Neumann stuff from here
        # for n in self.neumann_constraints.keys():
        #     mask = self.neumann_constraints[n]['mask']
        #     vals = self.neumann_constraints[n]['values']
            
        #     self.all_neumann_constraints_mask += mask # counts num constraints per dof. Zero if not a constrained dof
        #     self.all_neumann_constraints[mask] = vals
            
        # ## masks showing all constraints and all internal dofs
        self.all_dirichlet_constraints_mask = self.all_dirichlet_constraints_mask.astype(dtype=bool)
        self.all_dirichlet_constraints_vals = all_dirichlet_constraints_vals_full[self.all_dirichlet_constraints_mask]
        
        # self.all_neumann_constraints_mask = self.all_neumann_constraints_mask.astype(dtype=bool)
        
        self.all_internal_mask = ~self.all_dirichlet_constraints_mask # TODO: Add in Neumman constraints to this
        
        #self.L += self.all_neumann_constraints # add Neumann constraints to LHS ##TODO: remove all neumann stuff from here
        self.assemble()
        self.solve()
    
    def update_frequency(self, f):
        """ updates frequency and re-solves Helmholtz equation"""
        self.f = f
        self.assemble()
        self.solve()
        
    def assemble_unconstrained(self):
        k = 2*np.pi*self.f/self.c
        self.a_full = self.stiffness.copy() - k**2*self.mass.copy()
        self.F = np.zeros(len(self.X))
    
    def add_neumann_constraint(self, constraint_func):
        cname = constraint_func.__name__
        
        mask_func, vals_func = constraint_func()
        node_mask = mask_func(self.X, self.Y) # constraint node mask
        vals_full = np.ones(len(self.Y))*vals_func(self.X, self.Y)
        
        dof_mask = node_mask.copy() ## convert constraint node mask to dof mask
        vals = vals_full[dof_mask] ## remove vals calculated outside the constraint dofs
        
        self.neumann_constraints[cname] = {}
        self.neumann_constraints[cname]['mask'], self.neumann_constraints[cname]['values'] = dof_mask, vals      
        
    def add_dirichlet_constraint(self, constraint_func):
        cname = constraint_func.__name__
        
        mask_func, vals_func = constraint_func()
        node_mask = mask_func(self.X, self.Y) # constraint node mask
        vals_full = np.ones(len(self.Y))*vals_func(self.X, self.Y)
        
        dof_mask = node_mask.copy() ## convert constraint node mask to dof mask
        vals = vals_full[dof_mask] ## remove vals calculated outside the constraint dofs
        
        self.dirichlet_constraints[cname] = {}
        self.dirichlet_constraints[cname]['mask'], self.dirichlet_constraints[cname]['values'] = dof_mask, vals
        
    def assemble(self):
        self.assemble_unconstrained()
        a = self.a_full.copy()[~self.all_dirichlet_constraints_mask, :] ## remove rows corresponding to test functions at Dirchlet constraints
        dc_cols = a[:, self.all_dirichlet_constraints_mask].tocsr() # columns corresponding to Dirichlet constraints
        self.F = self.L[~self.all_dirichlet_constraints_mask] - dc_cols.dot(self.all_dirichlet_constraints_vals) # move constraints to LHS
        self.a = a.copy()[:, ~self.all_dirichlet_constraints_mask]
        
    def solve(self):
        u_nonDirichlet = spsolve(self.a.tocsr(), self.F) # solve for non-Dirchlet dofs
        
        ## Insert Dirichlet constraints into solution
        self.u = np.zeros(self.num_nodes)
        self.u[~self.all_dirichlet_constraints_mask] = u_nonDirichlet
        self.u[self.all_dirichlet_constraints_mask] = self.all_dirichlet_constraints_vals
        
    def plot(self, path = None):
        fig = plt.figure(figsize = (10, 10))
        ax = fig.add_subplot(projection="3d")
        sc = ax.scatter(self.X, self.Y, self.u, c=self.u, cmap="hsv")
        plt.title("Acoustic field (Pa)")
        plt.colorbar(sc)
        if path:
            plt.savefig(os.path.join(path, "pressureField.png"))
        else:
            plt.plot()
        
    def plot_matrix(self, matrix, title = ""):
        """convenience function for plotting matrices"""
        plt.imshow(matrix.todense())
        plt.colorbar()
        plt.title(title)
        plt.show()
