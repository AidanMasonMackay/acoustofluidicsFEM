# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 09:53:18 2021

@author: aidan
"""

from python_lib.base_class2d import base_class2d

from scipy.sparse import lil_matrix
import numpy as np

## TODO: Fix eps shape problem -- I get errors when I make eps all zeros
## TODO: Don't construct matrices for piezo coupling if not necessary
## TODO: make masks for elements in different domains 
## TODO: Don't build everything and then restrict the domain, that's inefficient

class elastic2d(base_class2d):
    def __init__(self, Neumann = [], meshName = "ExtraFineMesh", domain = 'all', params = {}, *kwargs, reorder = None):
        """ Builds matrices for electroelastic FEM solver """
        super().__init__(meshName, domain = domain, *kwargs)
        ## Import mesh
        self.params = params
        self.neumann_ids, self.neumann_vals_x, self.neumann_vals_y  = self.get_neumann_BCs(constraint_funcs = Neumann)

        #self.dofs, self.boundaries, self.nodes, self.elements = loadMesh(meshName)
        #self.num_nodes = self.nodes.shape[0]
        if reorder:
            self.reorder_nodes(X = reorder[0], Y = reorder[1])
        self.num_dofs = self.num_nodes*2
        
        #self.load_params()
        self.build_matrices()
        
        if domain!='all':
            self.restrict_domain()
    
    def get_neumann_BCs(self, constraint_funcs):
        node_ids = np.array([], dtype = int)
        vals_x = np.zeros(len(self.X))
        vals_y = np.zeros(len(self.X))
    
        for constraint_func in constraint_funcs:
            mask_func, vals_func_x, vals_func_y = constraint_func()
            node_mask = mask_func(self.X, self.Y) # constraint node mask
    
            node_ids = np.append(node_ids, np.nonzero(node_mask)[0].astype(int)) # ids of boundary nodes
            vals_x[node_ids] = vals_func_x(self.X[node_mask], self.Y[node_mask]) # x Neumann BC val at nodes
            vals_y[node_ids] = vals_func_y(self.X[node_mask], self.Y[node_mask]) # y Neumann BC val at nodes

        node_ids = np.unique(node_ids.copy())
        return node_ids, vals_x, vals_y

    #def load_params(self):
    #    """ load material parameters (defined on elements) into vectors """
    #    self.youngs = np.zeros(self.element_mask.shape)
    #    self.poisson = self.youngs.copy()
    #    
    #    for key in self.params.keys():
    #        self.youngs[self.element_mask == key] = self.params[key]['E']
    #        self.poisson[self.element_mask == key] = self.params[key]['mu']
    
    ## Remove loops!!
    def build_matrices(self): ## TODO: tell these which subdomain to construct matrices for
        neumann = np.zeros(self.num_nodes*2)
        K_uu = lil_matrix((self.num_nodes*2, self.num_nodes*2)) ## follows piezo book
        K_uphi = lil_matrix((self.num_nodes*2, self.num_nodes))
        K_phiphi = lil_matrix((self.num_nodes, self.num_nodes)) ## follows piezo book
        M = lil_matrix((self.num_nodes*2, self.num_nodes*2))
       
        ##  build mass and stiffness matrices
        for e_idx, e in enumerate(self.elements):

            area = self.integrate_element(e) # area of the triangle, for integration
            grad_xs, grad_ys = self.get_partials(e)
            K_e, K_p, K_d = self.get_elementary_matrices(e, e_idx)

            # i, j are global node indices, idx, jdx are local to each element
            # loop over node pairs in element ...
            ### TODO: remove i and j loops
            for idx, i in enumerate(e): # u
                for jdx, j in enumerate(e): # test function
                    ## mass
                    if i==j:
                        M[2*i+1, 2*j+1] += (1/6)*area
                        M[2*i, 2*j] += (1/6)*area
                    else:
                        M[2*i, 2*j] += (1/12)*area          
                        M[2*i+1, 2*j+1] += (1/12)*area
                   
                    K_uu[2*i:2*i+2, 2*j:2*j+2] += K_e[2*idx:2*idx+2, 2*jdx:2*jdx+2]
                    K_uphi[2*i:2*i+2, j:j+1] += K_p[2*idx:2*idx+2, jdx:jdx+1]
                    K_phiphi[i:i+1, j:j+1] += K_d[idx:idx+1, jdx:jdx+1]
                    
                    if ((i in self.neumann_ids) and (j in self.neumann_ids)): # if both i and j are boundary nodes
                        dels = np.sqrt((self.X[j]-self.X[i])**2 + (self.Y[j]-self.Y[i])**2) # distance between nodes i and j
                        neumann[2*i] += (1/3*self.neumann_vals_x[i] + 1/6*self.neumann_vals_x[j])*dels ## TEMP
                        neumann[2*i+1] += (1/3*self.neumann_vals_y[i] + 1/6*self.neumann_vals_y[j])*dels ## TEMP

                        neumann[2*j] += (1/6*self.neumann_vals_x[i] + 1/3*self.neumann_vals_x[j])*dels
                        neumann[2*j+1] += (1/6*self.neumann_vals_y[i] + 1/3*self.neumann_vals_y[j])*dels

        self.stiffness = K_uu
        self.K_uu = K_uu
        self.K_uphi = K_uphi
        self.K_phiphi = K_phiphi
        self.mass = M
        self.L = neumann
        
    def restrict_domain(self): ## General -- can move to base_class2d
        """ If only using a subdomain of the mesh, then matrices, nodes etc need to be restricted """
        # This is a messy way to do things because it re-orders the mesh 
        nonzero = self.stiffness.nonzero() # if row / column is empty, then this dof should be removed
        nonzero = np.unique(np.hstack([nonzero[0], nonzero[1]]))
        
        dofs = np.zeros([len(self.nodes)*2, 3])
        dofs[::2] = self.nodes
        dofs[1::2] = self.nodes
        
        self.nodes = dofs[nonzero][::2]
        self.stiffness = self.stiffness[nonzero, :][:, nonzero]
        self.K_uu = self.K_uu[nonzero, :][:, nonzero]
        
        nonzero2 =  self.K_phiphi.nonzero() # E-field has only one dof per node, needs a different mask
        nonzero2 = np.unique(np.hstack([nonzero2[0], nonzero2[1]]))
        
        self.K_uphi = self.K_uphi[nonzero, :][:, nonzero2]
        self.K_phiphi =self. K_phiphi[nonzero2, :][:, nonzero2]
        self.mass = self.mass[nonzero, :][:, nonzero]
        self.L = self.L[nonzero]

        ## Remove elements containing removed nodes, and renumber the rest to the new node indexing
        kept_nodes = nonzero[::2]//2 # old indices of the nodes that remain
        old_to_new = -np.ones(self.num_nodes, dtype = int) # -1 marks a removed node
        old_to_new[kept_nodes] = np.arange(len(kept_nodes))
        new_elements = old_to_new[self.elements]
        element_kept = np.all(new_elements >= 0, axis = 1)
        self.elements = new_elements[element_kept]
        self.element_mask = self.element_mask[element_kept]

        self.X = self.nodes[:, 0]
        self.Y = self.nodes[:, 1]
        self.num_nodes = len(self.nodes)
        self.num_dofs = self.num_nodes*2
    
    ## Unique to electroelastic
    def get_elementary_matrices(self, e, e_idx): # Think of this as solving PDE just in a subdomain, i.e within a single element, 'elementary domain'
        """ returns elastic elementary matrix for element e (for 2d) """
        
        ## Check which subdomain e is in
        domain = int(self.element_mask[e_idx])

        C = self.params[domain]["C"]
        coupling = self.params[domain]["coupling"]
        eps = self.params[domain]["eps"]
        
        area = self.integrate_element(e)
        grad_xs, grad_ys = self.get_partials(e)
          
        num_nodes_temp = len(e)
        B_e = np.zeros([3, num_nodes_temp*2])
        B_e_phi = np.zeros([2, num_nodes_temp])
        
        for i in range(num_nodes_temp):
            B_e[0, 2*i] = grad_xs[i]
            B_e[1, 2*i+1] = grad_ys[i]    
            B_e[2, 2*i] = grad_ys[i]
            B_e[2, 2*i+1] = grad_xs[i]
            
            B_e_phi[0, i] = grad_xs[i]
            B_e_phi[1, i] = grad_ys[i]
            
        K_e = np.matmul(np.matmul(B_e.T, C), B_e)*area
        K_p = np.matmul(np.matmul(B_e.T, coupling), B_e_phi)*area
        K_d = np.matmul(np.matmul(B_e_phi.T, eps), B_e_phi)*area

        return K_e, K_p, K_d
