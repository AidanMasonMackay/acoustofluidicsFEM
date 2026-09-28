# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 16:31:48 2022

@author: aidan
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import os

from python_lib.elastic2d import elastic2d as pythonElastic2d
#from comsol_lib.elastic2d import elastic2d as comsolElastic2d

class electroelastic2d(pythonElastic2d): 
    """  
        Assembles matrix equation for computing electric and elastic fields in homogeneous piezoelectric material
    """
    def __init__(self, f, Neumann = {}, Dirichlet = {}, params = {}, builder = 'python', meshName = "ExtraFineMesh", domain = 'all', reorder = None):
        if builder == 'python':
            pythonElastic2d.__init__(self, meshName = meshName, domain = domain, Neumann = Neumann, params = params, reorder = reorder)
#        if builder == 'comsol':
#            comsolElastic2d.__init__(self)
        
        self.Dirichlet = Dirichlet
        self.omega = 2*np.pi*f

        # TODO : Fix density implementation
        # This is a work around until I work through how to implement a non-constant rho
        breakpoint()
        dictKeys = list(params.keys())
        if len(dictKeys) > 1:
            rhos = []
            for domain in dictKeys:
                rhos.append(params[domain]['rho'])
            if len(list(set(rhos))) > 1:
                print("Warning : Solver doesn't work with a non-constant density.")
        self.rho = params[dictKeys[0]]['rho']

        # identify where u_x, u_y, and potentials are in the dof vector -- unique to electroelastic
        self.num_dofs = self.num_nodes*3
        self.dof_families = {}
        self.add_dof_family('u_x', 0, self.num_nodes*2, 2)
        self.add_dof_family('u_y', 1, self.num_nodes*2, 2)
        self.add_dof_family('pot', self.num_nodes*2, self.num_nodes*3, 1)
        
        self.assemble_unconstrained()
        self.L = np.zeros(self.num_dofs) ## TODO: add forcing terms as optional inputs, just like the with the Dirichlet and Neumann constraints
        self.update_Dirichlet_constraints()
        
    def add_dof_family(self, name, idx_start = 0, idx_end = -1, n = 1): ## TODO: Move to FEM base class
        """ this is used for identifying ids of dofs for imposing node constraints
        name is a string to identify dof. Every nth dof between idx_start and idx_end is assigned to this family """
        self.dof_families[name] = {}
        self.dof_families[name]['idx_start'] = idx_start
        self.dof_families[name]['idx_end'] = idx_end
        self.dof_families[name]['n'] = n
    
    def add_Dirichlet_constraint(self, constraint, cname): ## TODO: Move to FEM base class
        """ calculates indices and values for a given Dirichlet constraint dictionary """
        mask_func = constraint['mask_func']
        vals_func = constraint['vals_func']
        
        family_name = constraint['dof_family']
        dof_family = self.dof_families[family_name]
        start_idx, end_idx, n = dof_family['idx_start'], dof_family['idx_end'],dof_family['n']
        
        ## Construct constraint masks
        node_mask = mask_func(self.X, self.Y) # with node indexing
        dof_mask = np.zeros(self.num_dofs, dtype = bool)
        dof_mask[start_idx:end_idx:n] = node_mask # with dof indexing
        num_constraints = np.sum(dof_mask)
        
        ## Compute constraint values and map to their dofs
        vals = np.ones(num_constraints)*vals_func(self.X[node_mask], self.Y[node_mask])
        
        self.Dirichlet_constraints[cname] = {}
        self.Dirichlet_constraints[cname]['mask'], self.Dirichlet_constraints[cname]['values'] = dof_mask, vals
    
    def update_Dirichlet_constraints(self): ## TODO: Move to FEM base class
        """ assemble all constraints into a single mask with single list of values """
        ## mask and vals for all constraints
        self.all_Dirichlet_mask = np.zeros(self.num_dofs)
        all_Dirichlet_vals_full = np.zeros(self.num_dofs)
        self.Dirichlet_constraints = {}
        
        for key in self.Dirichlet.keys():
            self.add_Dirichlet_constraint(self.Dirichlet[key], cname = key)
            mask = self.Dirichlet_constraints[key]['mask']
            vals = self.Dirichlet_constraints[key]['values']
            
            self.all_Dirichlet_mask += mask # counts num constraints per dof. Zero if not a constrained dof
            all_Dirichlet_vals_full[mask] = vals # insert values for these dofs
            
        ## masks showing all constraints and all non-Dirichlet dofs
        self.all_Dirichlet_mask = self.all_Dirichlet_mask.astype(dtype=bool)
        self.all_nonDirichlet_mask = ~self.all_Dirichlet_mask
        self.all_Dirichlet_vals = all_Dirichlet_vals_full[self.all_Dirichlet_mask]
        
        self.assemble()
        self.solve()
        
    def assemble_unconstrained(self):  ## Unique to electroelastic
        # TODO : different rho for different domains
        
        self.a_full = lil_matrix((self.num_dofs, self.num_dofs))
        self.a_full[:self.num_nodes*2, :self.num_nodes*2] = self.K_uu.copy() - self.rho*self.omega**2*self.mass.copy()
        self.a_full[self.num_nodes*2:, :self.num_nodes*2] = self.K_uphi.T.copy()
        self.a_full[:self.num_nodes*2, self.num_nodes*2:] = self.K_uphi.copy()
        self.a_full[self.num_nodes*2:, self.num_nodes*2:] = -self.K_phiphi.copy()

        ## replace with K_uu ... once verified still works for elastic
        
    def assemble(self):  ## TODO: Move to FEM base class
        a = self.a_full.copy()[self.all_nonDirichlet_mask, :] ## remove rows corresponding to test functions on the boundaries
        dc_cols = a[:, self.all_Dirichlet_mask].tocsr() # columns corresponding to Dirichlet constraints
        self.F = self.L[self.all_nonDirichlet_mask] - dc_cols.dot(self.all_Dirichlet_vals) # move constraints to LHS
        self.a = a.copy()[:, self.all_nonDirichlet_mask]
        
    def solve(self): ## TODO: Move to FEM base class
        u_nonDirichlet = spsolve(self.a.tocsr(), self.F) # solve for internal dofs
        
        ## Insert Dirichlet constraints into solution
        self.u = np.zeros(self.num_dofs)
        self.u[self.all_nonDirichlet_mask] = u_nonDirichlet
        self.u[self.all_Dirichlet_mask] = self.all_Dirichlet_vals
        self.u = self.u
        
        self.solutions = {}
        
        for family in self.dof_families.keys():
            dof_family = self.dof_families[family]
            start_idx, end_idx, n = dof_family['idx_start'], dof_family['idx_end'],dof_family['n']
            self.solutions[family] = self.u[start_idx:end_idx:n]
        
    def plot(self, path): ## TODO: Move to FEM base class
        ## Plot directly on the mesh triangles -- nothing is drawn outside the mesh (holes, concave boundaries)
        figsize = (10, 10)
        triang = mtri.Triangulation(self.X, self.Y, triangles = self.elements)

        for solution_name in self.solutions.keys():
            solution = self.solutions[solution_name]

            plt.figure(figsize = figsize)
            tpc = plt.tripcolor(triang, solution, shading = 'gouraud') # linear interpolation within each element
            plt.gca().set_aspect('equal')
            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            plt.title(solution_name)
            plt.colorbar(tpc)

            if path:
                plt.savefig(os.path.join(path, f"{solution_name}_interpolated.png"))
            else:
                plt.show()
        
    def plot_matrix(self, matrix, title = ""):  ## TODO: Move to FEM base class
        """convenience function for plotting matrices"""
        plt.imshow(matrix.todense())
        plt.colorbar()
        plt.title(title)
        plt.show()
