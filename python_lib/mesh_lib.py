# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:08:50 2021

@author: aidan
"""
import re
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path

meshPath = Path("meshes") # absolute path to mesh .txt files
## TODO: WIll get problems with more than 10000 elements - find a way to deal with spaces better
def loadMesh(meshName):
    """ builds node, element, and element label arrays from comsol .nas file converted to .txt file """ 
    meshDomain_path = Path("{}//{}.txt".format(meshPath, meshName))
    f = open(meshDomain_path)

    # initialise arrays
    nodes = np.array([0, 0, 0])
    elements = np.array([0, 0, 0])
    element_domain_mask = np.array([])
    
    el_type = 'CTRIA' ## this will eventually be a list - multiple element types

    for line in f:
        if 'GRID' in line:
            node1 = line.split('   ')[-1]
            node = [node1[:-17], node1[-17:-9], node1[-9:]]
            nodes = np.vstack([nodes, node])
            
        if el_type in line:
            temp = line.split('    ')[-4:]
            elements = np.vstack([elements, temp[1:]])
            element_domain_mask = np.append(element_domain_mask, temp[0])
            
    # for line in f: ## this one fails x / y more than 0.999
    #     if 'GRID' in line:
    #         node1 = line.split()[-1]
    #         node = [node1[:8], node1[8:16], node1[16:]]
    #         nodes = np.vstack([nodes, node])
            
    #     if el_type in line:
    #         temp = line.split()[2:]
    #         elements = np.vstack([elements, temp[1:]])
    #         element_domain_mask = np.append(element_domain_mask, temp[0])
    
    nodes = nodes.astype(float)[1:] 
    elements = elements[1:].astype(int) - 1 # elements start at 0 not 1 (python matlab difference)
    element_domain_mask = element_domain_mask.astype(int)
    
    return nodes, elements, element_domain_mask

# def loadMesh(meshName):
#     """ builds index, node, and element and element arrays from comsol .txt file"""
#     meshBoundary_path = Path("{}//{}Boundary.txt".format(meshPath, meshName))
#     meshDomain_path = Path("{}//{}Domain.txt".format(meshPath, meshName))

#     fDomain = open(meshDomain_path)
#     fBoundary = open(meshBoundary_path)
    
#     node_flag = element_flag = 0
    
#     for line in fDomain:
#         ## retrieve metadata
#         if '% Dimension:' in line:
#             dimension = int(re.findall(r'\d+', line)[0])
#         if '% Nodes:' in line:
#             num_nodes = int(re.findall(r'\d+', line)[0])
#         if '% Elements:' in line:
#             num_elements = int(re.findall(r'\d+', line)[0])
#             break
    
#     ## initialise arrays
#     nodes = np.zeros([num_nodes, dimension], np.float64)
#     indices = np.arange(num_nodes)
#     elements = np.zeros([num_elements, dimension+1], dtype = np.uint)
    
#     element_counter = 0
#     node_counter = 0
    
#     for line in fDomain:
#         if '% Coordinates' in line:
#             node_flag = 1
#             continue
#         if '% Elements' in line:
#             node_flag = 0
#             element_flag = 1
#             continue
#         if node_flag == 1:
#             nodes[node_counter, :] = line.split()
#             node_counter+=1     
#         if element_flag == 1:
#             elements[element_counter, :] = line.split()
#             element_counter+=1
    
#     ## get indicies of boundary nodes
#     for line in fBoundary:
#         ## retrieve number boundary nodes
#         if '% Nodes:' in line:
#             numBoundary_nodes = int(re.findall(r'\d+', line)[0])
#             break
    
#     boundary_nodes = np.zeros([numBoundary_nodes, dimension], np.float64)
#     boundary_node_counter = 0
#     boundary_node_flag = 0
    
#     for line in fBoundary:
#         if '% Coordinates' in line:
#             boundary_node_flag = 1
#             continue
#         if boundary_node_flag == 1:
#             boundary_nodes[boundary_node_counter, :] = line.split()
#             boundary_node_counter+=1
#         if boundary_node_counter == numBoundary_nodes:
#             break
#     boundaries = np.array([np.where((nodes == i).all(axis=1))[0][0] for i in boundary_nodes])
    
#     return indices, boundaries, nodes, elements-1 ## elements should start from 0 not 1

def showMeshPlot(nodes, elements, boundaries = None):
    """ plots mesh """
    # https://stackoverflow.com/questions/52202014/how-can-i-plot-2d-fem-results-using-matplotlib
    y = nodes[:,0]
    z = nodes[:,1]

    #https://stackoverflow.com/questions/49640311/matplotlib-unstructered-quadrilaterals-instead-of-triangles
    def quatplot(y,z, quatrangles, ax=None, **kwargs):

        if not ax: ax=plt.gca()
        yz = np.c_[y,z]
        verts= yz[quatrangles]
        pc = matplotlib.collections.PolyCollection(verts, **kwargs)
        ax.add_collection(pc)
        ax.autoscale()

    plt.figure()
    plt.gca().set_aspect('equal')

    quatplot(y,z, np.asarray(elements), ax=None, color="crimson", facecolor="None") 
    plt.plot(y,z, marker="o", ls="", color="crimson")

    plt.title('Mesh')
    plt.xlabel('y')
    plt.ylabel('x')
    
    if boundaries.any():
        plt.plot(nodes[boundaries, 0], nodes[boundaries, 1], linestyle = '', marker = 'x', color = 'black')
    plt.show()