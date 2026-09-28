# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

# This class contains functions that are common accross all the different FEM solvers

from python_lib.mesh_lib import loadMesh

import numpy as np
import pandas as pd

class base_class2d:
    """ holds nodes and elements in order """
    def __init__(self, meshName = "FineMesh", domain = 'all', reorder = None):  
        ## Import comsol mesh from .txt file
        print("assembling in python")
        self.nodes, self.elements, self.element_mask = loadMesh(meshName)
        if reorder:
            self.reorder_nodes(X = reorder[0], Y = reorder[1])
        self.X, self.Y = self.nodes[:, 0], self.nodes[:, 1]
        self.num_nodes = self.nodes.shape[0]
        
        if domain != 'all': ## for when there's different physics in different domains on the mesh
            mask = [i in domain for i in self.element_mask]
            self.elements = self.elements[mask]
            self.element_mask = self.element_mask[mask]
    
    def reorder_nodes(self, X, Y):
        """ Re-orders to order of X, Y (X and Y must describe same mesh)
            Useful for making comparisons to comsol outputs """ 
        nodes_new = np.vstack([X, Y]).T
        node_mappings = [] ## mappings from python to comsol
        for ip, node_python in enumerate(self.nodes):
            for ic, node_comsol in enumerate(nodes_new):
               if np.all(np.isclose(node_python, node_comsol, atol = 1e-4)):
                    node_mappings.append([ip, ic])
                    break
        self.nodes = nodes_new.copy()
        src_array = np.array(node_mappings).T[0 ,:]
        dest_array = np.array(node_mappings).T[1 ,:]
        
        el_df = pd.DataFrame(self.elements)
        el_df = el_df.replace(src_array, dest_array)
        
        print("updated elements") ## TEMP
        #bd_df = pd.DataFrame(self.boundaries)
        #bd_df = bd_df.replace(src_array, dest_array)
        #self.boundaries = np.array(bd_df).flatten()
    
        #self.internal_nodes = self.dofs.copy()
        #for b in self.boundaries:
        #    self.internal_nodes = np.delete(self.internal_nodes, np.where(self.internal_nodes == b))
        
        self.X, self.Y = self.nodes[:, 0], self.nodes[:, 1]
    
    def integrate_element(self, e):
        """ get the area of an element """
        x1, x2, x3 = [self.nodes[i, 0] for i in e]
        y1, y2, y3 = [self.nodes[i, 1] for i in e]
        integral = 0.5*(-x2*y1 + x3*y1 + x1*y2 - x3*y2 - x1*y3 + x2*y3)
        return np.abs(integral)
    
    def get_partials(self, e):
        """ returns partial derivatives for each of the hat functions which are non-zero in e"""
        x1, x2, x3 = [self.nodes[i, 0] for i in e]
        y1, y2, y3 = [self.nodes[i, 1] for i in e]
        e_det = (x2*y3 - y2*x3) - (x1*y3 - y1*x3) + (x1*y2 - y1*x2)
        
        grad1_x = (y2 - y3)
        grad1_y = -(x2 - x3)
        
        grad2_x = -(y1 - y3)
        grad2_y = (x1 - x3)
        
        grad3_x = (y1 - y2)
        grad3_y = -(x1 - x2)
        
        grad_xs = np.array([grad1_x, grad2_x, grad3_x])/e_det
        grad_ys = np.array([grad1_y, grad2_y, grad3_y])/e_det
        
        return grad_xs, grad_ys