# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 12:36:58 2021

@author: aidan
"""

from python_lib.base_class2d import base_class2d

import numpy as np
from scipy.sparse import lil_matrix
# I don't like that Neumann BCs have to be known so early in asssembly

class helmholtz2d(base_class2d):
    """ assembles unconstrained LHS matrix and RHS loading vector (assumed all zeros at the moment) """
    def __init__(self, neumann_constraints = [], coupling_masks = [], meshName = "FineMesh", domain = 'all', reorder = None, *kwargs):
        super().__init__(meshName, domain = domain, reorder = reorder, *kwargs)
        self.neumann_ids, self.neumann_vals = self.get_neumann_BCs(constraint_funcs = neumann_constraints)
        self.coupling_ids = self.get_coupling_ids(coupling_masks)
        self.build_matrices()
        
        if domain != 'all':
            self.restrict_domain()
    
    def get_neumann_BCs(self, constraint_funcs):
        dof_ids = np.array([], dtype = int)
        vals = np.zeros(len(self.X))
    
        for constraint_func in constraint_funcs:
            mask_func, vals_func = constraint_func()
            node_mask = mask_func(self.X, self.Y) # constraint node mask
            dof_mask = node_mask.copy() ## convert constraint node mask to dof mask
    
            dof_ids = np.append(dof_ids, np.nonzero(node_mask)[0].astype(int)) # ids of boundary nodes
            vals[dof_ids] = vals_func(self.X[dof_mask], self.Y[dof_mask]) # Neumann BC val at nodes

        dof_ids = np.unique(dof_ids.copy())
        return dof_ids, vals
    
    def get_coupling_ids(self, coupling_masks):
        """ ids for nodes on acoustic-elastic interfaces """
        dof_ids = np.array([], dtype = int)
        for mask in coupling_masks:
            node_mask = mask(self.X, self.Y) # constraint node mask    
            dof_ids = np.append(dof_ids, np.nonzero(node_mask)[0].astype(int)) # ids of boundary nodes

        dof_ids = np.unique(dof_ids.copy())
        return dof_ids
    
    def build_matrices(self):
        neumann = np.zeros(len(self.X))
        self.coupling = np.array([0, 0, 0])
        
        M = lil_matrix((self.num_nodes, self.num_nodes)) ## Mass matrix
        K = lil_matrix((self.num_nodes, self.num_nodes)) ## Stiffness matrix
       
        ##  build mass and stiffness matrices
        for e in self.elements:
            area = self.integrate_element(e) # area of the triangle, for integration
            grad_xs, grad_ys = self.get_partials(e)
            
            # i, j are global node indices, idx, jdx are local to each element
            # loop over node pairs in element ...
            ### TODO: remove i and j loops following Ru's advice
            for idx, i in enumerate(e):
                for jdx, j in enumerate(e):
                    ## mass
                    if i==j:
                        M[i, j] += (1/6)*area
                    else:
                        M[i, j] += (1/12)*area
                        
                    # Neumann BCs
                    # These add twice - once for i, j and again for j, i
                    # To account for this, the values are divided by 2
                    if ((i in self.neumann_ids) and (j in self.neumann_ids)): # if both i and j are boundary nodes
                        dels = np.sqrt((self.X[j]-self.X[i])**2 + (self.Y[j]-self.Y[i])**2) # distance between nodes i and j  
                        neumann[i] += 0.5 * (1/3*self.neumann_vals[i] + 1/6*self.neumann_vals[j])*dels
                        neumann[j] += 0.5 * (1/6*self.neumann_vals[i] + 1/3*self.neumann_vals[j])*dels
                        
                    # Save info for acoustic-elastic coupling
                    if ((i!=j) and (i in self.coupling_ids) and (j in self.coupling_ids)):
                        dels = np.sqrt((self.X[j]-self.X[i])**2 + (self.Y[j]-self.Y[i])**2) # distance between nodes i and j
                        self.coupling = np.vstack([self.coupling, [i, j, dels]])
                        
                    ## stiffness
                    K[i, j] += (grad_xs[idx]*grad_xs[jdx] + grad_ys[idx]*grad_ys[jdx])*area         
                    
        if self.coupling.shape != (3,): # remove placeholder 
            self.coupling = self.coupling[1:]
        else:
            self.coupling = np.array([])
    
        self.stiffness = K
        self.mass = M
        self.L = neumann
        
    def restrict_domain(self):
        """ If only using a subdomain of the mesh, then matrices, nodes etc need to be restricted """
        nonzero = self.stiffness.nonzero() # if row / column is empty, then this dof should be removed
        nonzero = np.unique(np.hstack([nonzero[0], nonzero[1]]))
        
        self.nodes = self.nodes[nonzero]
        self.stiffness = self.stiffness[nonzero, :][:, nonzero]
        self.mass = self.mass[nonzero, :][:, nonzero]
        self.L = self.L[nonzero]

        ## Remove elements containing removed nodes, and renumber the rest to the new node indexing
        kept_nodes = nonzero # one dof per node, so dof indices are old node indices
        old_to_new = -np.ones(self.num_nodes, dtype = int) # -1 marks a removed node
        old_to_new[kept_nodes] = np.arange(len(kept_nodes))
        new_elements = old_to_new[self.elements]
        element_kept = np.all(new_elements >= 0, axis = 1)
        self.elements = new_elements[element_kept]
        self.element_mask = self.element_mask[element_kept]

        self.X = self.nodes[:, 0]
        self.Y = self.nodes[:, 1]
        self.num_nodes = len(self.nodes)
        self.num_dofs = self.num_nodes