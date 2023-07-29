#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from geopandas import GeoDataFrame as gdf
import networkx as nx
import osmnx as ox
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
ox.config(use_cache=True, log_console=True)


# # Retrieve regional map from OSM.

# In[ ]:


ox.config(use_cache=True, log_console=True)

place1 = 'Destel, Germany' 
place2 = 'Petershagen, Germany'
place3 = 'Hannover, Germany'
place4 = 'Elze, Germany'
place5 = 'Berlin, Germany'
place6 = 'North Rhine-Westphalia, Germany'
place7 = 'Kreis Minden-Lübbecke, Germany'
place8 = 'Rahden,Germany'

cf = '["highway"~"motorway|primary|secondary"]'

#G = ox.graph_from_place(place4, network_type='drive', custom_filter=cf)
G = ox.graph_from_place(place7, network_type='drive', custom_filter=cf)
G = ox.project_graph(G)  #G was Gp
#G = ox.simplification.simplify_graph(G, strict=True, remove_rings=True)
G = ox.simplification.consolidate_intersections(G, tolerance=150, rebuild_graph=True, dead_ends=False, reconnect_edges=True)

G = ox.utils_graph.get_undirected(G)
#G.add_edge(1, 15)

fig, ax = ox.plot_graph(G, node_color="r",node_size=20, edge_color="black",edge_linewidth=1,bgcolor='white', show=False, close=False)
#ax.set_title('"drive" network of district Minden-Lübbecke, Germany before simplification')
plt.savefig('photo 1', dpi =1000,  bbox_inches='tight')
plt.show()


# In[ ]:


import osmnx as ox
import pandas as pd
import folium
place = ['Kreis Minden-Lübbecke, Germany']
gdf_nodes = gdf_edges = None
for place in place:
    G = ox.graph_from_place(place,custom_filter=cf, simplify = True, network_type='drive')
    n_, e_ = ox.graph_to_gdfs(G)
    n_["place"] = place
    e_["place"] = place
    if gdf_nodes is None:
        gdf_nodes = n_
        gdf_edges = e_
    else:
        gdf_nodes = pd.concat([gdf_nodes, n_])
        gdf_edges = pd.concat([gdf_edges, e_])

gdf_edges.explore(column="place", height=500, width=700)


# # Number of node

# In[ ]:


C = nx.to_numpy_matrix(G)   # for loop

number_of_node = int((np.sqrt(C.size)))
print(number_of_node)

for i in range(number_of_node):
    for j in range(number_of_node):
        if C[i,j] != 0:
            C[i,j] = 1
            
# matrix to excel 
marks_data = pd.DataFrame(C)
file_name = 'C.xlsx'
  
# saving the excel
marks_data.to_excel(file_name)


# # Retrieve number of lane for each route and delete unclassified routes

# In[ ]:


# Highway classification:
# motorway, trunk  - Autobahn, 3 lanes
# primary - Bundesstraßen, 2 lanes
# secondary  - Landesstraßen/Staatsstraßen, 1 lane
# teritary - Landesstraßen/Staatsstraßen, 1 lane
countm = 0
countp = 0
counts = 0


lane_matrix = np.zeros((number_of_node, number_of_node))
nodes, edges = ox.graph_to_gdfs(G)
#print(np.shape(edges))


for i in range (number_of_node):
    for j in range (number_of_node):
        if C[i,j] == 1 and i<j:
            orig = list(G)[i]
            dest = list(G)[j]
            route = nx.shortest_path(G, orig, dest, weight='travel_time')

            route_type = ((ox.utils_graph.get_route_edge_attributes(G, route, "highway")))
            
            #print(route_type)
            if route_type == ['motorway'] or route_type == ['trunk']:
                countm = countm+1
                lane_matrix[i,j] = 3
            if route_type == ['primary']:
                countp = countp+1
                lane_matrix[i,j] = 2
            elif route_type == ['secondary'] or route_type == ['tertiary']:
                lane_matrix[i,j] = 1
                counts = counts+1
            #else:

                #G.remove_edge(orig, dest)
                #G.remove_edge(dest, orig)
                
  

# matrix to excel 
marks_data = pd.DataFrame(lane_matrix)
file_name = 'lane_matrix.xlsx'
  
# saving the excel
marks_data.to_excel(file_name)

print('number of motorway routes:', countm)
print('number of primary routes:', countp)
print('number of secondary routes:', counts)
fig, ax = ox.plot_graph(G)        
 


# # Remove nodes with degree of freedom less than 1.Plot retrieved map

# In[ ]:


remove = [node for node, degree in G.degree() if degree < 1]
G.remove_nodes_from(remove)

fig, ax = ox.plot_graph(G)


# # Number of nodes and Number of routes

# In[ ]:


nodes, edges = ox.graph_to_gdfs(G)
k = np.shape(edges)
k = list(k)
number_of_edge = k[0]
number_of_edge = int(number_of_edge)
print('number of edges:', number_of_edge)


number_of_node = int((np.sqrt(C.size)))
print('number of nodes:', number_of_node)


# In[ ]:


# impute missing edge speeds and add travel times
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)


# # Normal travel time (normal edge weight)

# In[ ]:


import math
travel_time_matrix = np.zeros((number_of_node,number_of_node))

for i in range (number_of_node):
    for j in range (number_of_node):
        if C[i,j] == 1 and i<j:
            orig = list(G)[i]
            dest = list(G)[j]
            travel_time = nx.shortest_path_length(G, orig, dest, weight='travel_time')
            travel_time_matrix[i,j] = round(travel_time)
        else:
            travel_time_matrix[i,j] = math.inf
            

# matrix to excel 
marks_data = pd.DataFrame(travel_time_matrix)
file_name = 'travel_time_matrix.xlsx'
  
# saving the excel
marks_data.to_excel(file_name)

print(np.shape(travel_time_matrix))


# # Normal travel distance (edge lenght)

# In[ ]:


travel_distance_matrix = np.zeros((number_of_node,number_of_node))
for i in range (number_of_node):
    for j in range (number_of_node):
        if C[i,j] == 1 and i<j:
            
            
            orig = list(G)[i]
            dest = list(G)[j]
            travel_distance = nx.shortest_path_length(G, orig, dest, weight='length')
            travel_distance_matrix[i,j] = round(travel_distance)
            

# matrix to excel 
marks_data = pd.DataFrame(travel_distance_matrix)
file_name = 'travel_distance_matrix.xlsx'
  
# saving the excel
marks_data.to_excel(file_name)



all_routes_lenght = 0
for i in range (number_of_node):
    for j in range (number_of_node):
        if C[i,j] == 1 and i<j:
            all_routes_lenght = all_routes_lenght + travel_distance_matrix[i,j] # the result is in metr
print(all_routes_lenght) # in m


# # Normal travel velocity matrix

# In[ ]:


travel_velosity_matrix = 3.6*(travel_distance_matrix)/(travel_time_matrix)  # multiplied by 3.6 to transfer from m/s to km/h

# matrix to excel 
marks_data = pd.DataFrame(travel_velosity_matrix)
file_name = 'travel_velosity_matrix.xlsx'
  
# saving the excel
marks_data.to_excel(file_name)


# # Average node connectivity

# In[ ]:


import copy
G1 = copy.deepcopy(G)
nx.average_node_connectivity(G1)


# In[ ]:


sensitivity = np.zeros((number_of_node,number_of_node))
for i in range(number_of_node):
    for j in range(number_of_node):
        if C[i,j] == 1 and i<j:
            G1 = copy.deepcopy(G)
            G1.remove_edge(i,j)
            sensitivity[i,j] = nx.average_node_connectivity(G1)


# In[ ]:


# matrix to excel 
marks_data = pd.DataFrame(sensitivity)
file_name = 'sensitivity.xlsx'
  
# saving the excel
marks_data.to_excel(file_name)


# In[ ]:


m = np.min(sensitivity)
print(m)


# # Largest connected cluster 

# In[ ]:


G_lcc = copy.deepcopy(G)
G_lcb = copy.deepcopy(G)


# In[ ]:


#Betweenness centrality
clusterb = []
x_axis = []
counter = 0
# measure attack strategies
bc = nx.betweenness_centrality(ox.get_digraph(G), weight="travel_time")
ka = len(bc)
for i in range(ka-1):
    m = max(bc, key=bc.get)
    del bc[m]
    G_lcc.remove_node(m)
    largest_cc = max(nx.connected_components(G_lcc), key=len)
    clusterb.append(len(largest_cc))
    x_axis.append(counter)
    counter = counter+1


# In[ ]:


#Closeness centrality
clusterc = []
x_axis = []
counter = 0
# measure attack strategies
bc = nx.closeness_centrality(G_lcb)
ka = len(bc)
for i in range(ka-1):
    m = max(bc, key=bc.get)
    del bc[m]
    G_lcb.remove_node(m)
    largest_cc = max(nx.connected_components(G_lcb), key=len)
    clusterc.append(len(largest_cc))
    x_axis.append(counter)
    counter = counter+1


# In[ ]:


plt.rcParams["figure.figsize"] = [10, 7]
plt.plot(x_axis, clusterb, 'r')
plt.plot(x_axis, clusterc, 'b', '-.')


plt.title('Attack strategy and Largest connected cluster')
plt.xlabel('attack iteration')
plt.ylabel('Largest connected cluster')

plt.plot([1,2,3],'r-',label='Largest connected cluster, after attack with betweenness centrality')
plt.plot([0.5,2,3.5],'b-',label='Largest connected cluster, after attack with closeness centrality')
plt.legend()
plt.savefig('attack strategy North Rhine-Wesphalia', dpi =1000,  bbox_inches='tight')
plt.show()


# # Climate data

# In[ ]:


Baden_Wurttemberg
Bremen
Mecklenburg_West
Saxony
Bavaria
Hamburg
North_Rhine_Westphalia
Saxony_Anhalt
Berlin
Rhineland_Palatinate
Hesse
Schleswig_Holstein
Brandenburg
Lower_Saxony
Saarland
Thuringia


# # Node dictionary. Key = id, value = int

# In[ ]:


nodes, edges = ox.graph_to_gdfs(G)


# In[ ]:


# id - int dictionary
i=0
iddictionary = {}
for node in (G.nodes):
    iddictionary[node] = i 
    i = i+1
#print(iddictionary)


# In[ ]:


def node_id_to_int(id1):
    #id to number dictionary
    
    return iddictionary[id1]


# # Node dictionary. Key = id, value = int

# In[ ]:


# int - id dictionary
i = 0
intdictionary = {}
for node in (G.nodes):
    
    intdictionary[i] = node
    i = i+1
#print(intdictionary)


# In[ ]:


def node_int_to_id(id2):
    #id to number dictionary
    
    return intdictionary[id2]


# # Node neighbours by id

# In[ ]:


def node_neighbour_by_id(id3):
    # input: node id, output: neighbour nodes id
    dictionary = {}

    for node in (G.nodes):
        #print(node)
        instant = []
        neighbor_list = [n for n in G.neighbors(node)]
        #print(f"Neighbor({node})=({neighbor_list})")
        for i in (neighbor_list):
            #print ('i=',i)
            instant.append(i)
        dictionary[node] = neighbor_list
    
    
    #print(dictionary[id])
    return dictionary[id3]


# In[ ]:


id3 = 43
a = node_neighbour_by_id(id3)
print(a)
print(a[1])


# # Mathematical model to determine velosity change after increasing density

# In[ ]:


def FlowDensity (velosity):

    if velosity < 48:   # km/h
        Flow = 700     # veh/hour/lane
        Density = 42    # veh/km
    elif velosity >=48 and velosity < 74:  # km/h
        Flow = 900    
        Density = 34
    elif velosity >=74 and velosity < 87:  # km/h
        Flow = 1325
        Density = 23
    elif velosity >=87 and velosity < 92:  # km/h
        Flow = 1700
        Density = 17
    elif velosity >=92 and velosity < 97:  # km/h
        Flow = 1925
        Density = 10
    else: 
        Flow = 2200
        Density = 6
    return Density


# In[ ]:


a = drop_degraded_time(-1)


# # Mathematical model for PCI 

# In[ ]:


#Random age

import random
from copy import copy
age_matrix = np.zeros((number_of_node,number_of_node))


for i in range (number_of_node):
    for j in range (number_of_node):
        if C[i,j] == 1 and i<j:
            age_matrix[i,j] = random.randint(2,15)

# matrix to excel 
marks_data = pd.DataFrame(age_matrix)
file_name = 'age_matrix.xlsx'
  
# saving the excel
marks_data.to_excel(file_name)


# In[ ]:


import copy
age_matrix1 = copy.deepcopy(age_matrix)


# In[ ]:


age_matrix = copy.deepcopy(age_matrix1)


# In[ ]:


#Read climate data
df = pd.read_excel(r'C:\Users\User\Desktop\thesis\code jupyter notebook\Destel\climate.xlsx')

print(df)

print(df.iloc[0][1])


# In[ ]:


from array import *
PCI_matrix = np.zeros((number_of_node,number_of_node))

r = region = 7 #North Rhine-Westphalia (7) so retrieve climate data for this region

AAS  = df.iloc[2][r] 
AART = df.iloc[0][r] - df.iloc[1][r]


# In[ ]:


def PCI_from_age(age_matrix):
    

    counter = 0
    
    

    pci = []
    age = []

    k = 0

    for i in range (number_of_node):
        for j in range (number_of_node):
            if C[i,j] == 1 and i<j and PCI_matrix[i,j] != 15:

                PCI_matrix[i,j] = 152.75-(6.17*age_matrix[i,j])-(0.0091*AAS)-(2.76*AART)+(0.0570*AART*AART)
                if PCI_matrix[i,j] > 99:
                    PCI_matrix[i,j] = 99
                if PCI_matrix[i,j] < 26:
                    PCI_matrix[i,j] = 26
                PCI_matrix_gold = copy.deepcopy(PCI_matrix)
                pci.append(PCI_matrix[i,j])
                age.append(age_matrix[i,j])
                k=k+1

    #print(PCI_matrix)
    #plt.scatter(age,pci)


    # matrix to excel 
    marks_data = pd.DataFrame(PCI_matrix)
    file_name = 'PCI_matrix.xlsx'

    # saving the excel
    marks_data.to_excel(file_name)
    return PCI_matrix


# In[ ]:


PCI_matrix = PCI_from_age(age_matrix)
PCI_matrix_gold = copy.deepcopy(PCI_matrix)


# In[ ]:


#quantitative representation of the network by value PCI
orig, dest = list(G)[0], list(G)[-1]
route = []
route_color = []
route_linewi = []
for iroute in range(number_of_node):
    for jroute in range (number_of_node):
        if C[iroute,jroute] == 1 and iroute<jroute:
            
        
            orig = list(G)[iroute]
            dest = list(G)[jroute]
            ro = nx.shortest_path(G, orig, dest, weight='length')
            route.append(ro)
            route_linewi.append(2)
            if PCI_matrix[iroute,jroute] < 50.0:
                route_color.append("r")
            if PCI_matrix[iroute,jroute] >= 50.0 and PCI_matrix[iroute,jroute] < 75.0:
                route_color.append("y")
            if PCI_matrix[iroute,jroute] >= 75.0:
                route_color.append("g")


# plot the routes
from matplotlib.pyplot import figure


fig, ax = ox.plot_graph_routes(G, routes = route, route_linewidths=route_linewi ,orig_dest_size =10, route_colors = route_color, node_size=0, edge_linewidth=0.2)
figure = plt.gcf()
plt.show()
fig.savefig('colored_PCI.png', dpi =1000,  bbox_inches='tight')


# # Speed and Travel time after pavement deterioration

# In[ ]:


degraded_velosity_matrix = np.zeros((number_of_node,number_of_node))
degraded_travel_time_matrix = np.zeros((number_of_node,number_of_node))


# In[ ]:


def degraded_time(PCI_matrix):
    

    
    a = 1385.406
    b = 15.985

    a50 = 1275.372
    b50 = 14.997

    for i in range (number_of_node):
        for j in range (number_of_node):
          
                
            if C[i,j] == 1 and i<j and PCI_matrix[i,j] >= 25 and PCI_matrix[i,j] <= 86 and PCI_matrix[i,j] != 15:
                deltav = ((a/PCI_matrix[i,j])-b)
                degraded_velosity_matrix[i,j] = travel_velosity_matrix[i,j]-deltav
                degraded_travel_time_matrix[i,j] = 3.6*travel_distance_matrix[i,j]/degraded_velosity_matrix[i,j]

            if C[i,j] == 1 and i<j and PCI_matrix[i,j] > 86:
                degraded_velosity_matrix[i,j] = travel_velosity_matrix[i,j]
                degraded_travel_time_matrix[i,j] = 3.6*travel_distance_matrix[i,j]/degraded_velosity_matrix[i,j]

            if C[i,j] == 1 and i<j and PCI_matrix[i,j] == 100:
                degraded_velosity_matrix[i,j] = travel_velosity_matrix[i,j]
                degraded_travel_time_matrix[i,j] = 3.6*travel_distance_matrix[i,j]/travel_velosity_matrix[i,j]
            if C[i,j] != 1:
                degraded_travel_time_matrix[i,j] = math.inf
                
    # matrix to excel 
    marks_data = pd.DataFrame(degraded_velosity_matrix)
    file_name = 'degraded_velosity_matrix.xlsx'
  
    # saving the excel
    marks_data.to_excel(file_name)


    #plt.scatter(degraded_velosity,PCI)

    # matrix to excel 
    marks_data = pd.DataFrame(degraded_travel_time_matrix)
    file_name = 'degraded_travel_time_matrix.xlsx'
  
    # saving the excel
    marks_data.to_excel(file_name)



    plt.scatter(degraded_velosity_matrix, PCI_matrix, c='r', marker='s', label='-1' )
    plt.scatter(travel_velosity_matrix, PCI_matrix, c='g', marker='s', label='-1')
    
    
    return degraded_travel_time_matrix, degraded_velosity_matrix


# # Speed and Travel time after pavement deterioration and traffic redistribution

# In[ ]:


def drop_time(PCI_matrix, drop_degraded_travel_time_matrix, drop_degraded_velosity_matrix):
    

    
    a = 1385.406
    b = 15.985

    a50 = 1275.372
    b50 = 14.997

    for i in range (number_of_node):
        for j in range (number_of_node):
            if C[i,j] == 1 and i<j and PCI_matrix[i,j] >= 25 and PCI_matrix[i,j] <= 86 and PCI_matrix[i,j] != 15:
                deltav_gold = ((a/PCI_matrix_gold[i,j])-b)
                deltav = ((a/PCI_matrix[i,j])-b)
                deltavelo = deltav-deltav_gold
                drop_degraded_velosity_matrix[i,j] = drop_degraded_velosity_matrix[i,j]-deltavelo
                drop_degraded_travel_time_matrix[i,j] = 3.6*travel_distance_matrix[i,j]/drop_degraded_velosity_matrix[i,j]
                

    
    return drop_degraded_travel_time_matrix, drop_degraded_velosity_matrix


# # Influence of pavement condition on speed change

# In[ ]:


PCI = np.array([25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90])
a85 = 1385.406
b85 = 15.985

a50 = 1275.372
b50 = 14.997

deltav50 = ((a50/PCI)-b50)
deltav85 = ((a85/PCI)-b85)


plt.plot(PCI, deltav50, 'g', linewidth=1, label='line 1')
plt.plot(PCI, deltav85, 'r', linewidth=1, label='line 2')
#plt.title('title name')
plt.xlabel('PCI')
plt.ylabel('Change in speed (km/h)')
plt.savefig('speed')
plt.show()


# # Ascending order of routes by its PCI

# In[ ]:


import copy
modified_PCI_matrix = copy.deepcopy(PCI_matrix)


# In[ ]:


#choose 10000m route with least PCI and maintain

#find the first smallest PCI


si1 = []
sj1 = []

for k in range (number_of_node-1):
    smallest_pci = 200.0
    for i in range (number_of_node):
        for j in range (number_of_node):
            if C[i,j] == 1 and i<j:
                if modified_PCI_matrix[i,j] < smallest_pci and modified_PCI_matrix[i,j]>20 :
                    smallest_pci = modified_PCI_matrix[i,j]
                    ii = i
                    jj = j
    #modified_PCI_matrix[ii] = 0
    modified_PCI_matrix[ii,jj] = 200
    
    ni = G.degree[ii]
    nj = G.degree[jj]
    
    si1.append(ii)
    sj1.append(jj)
#print(si1,sj1)  #array with ascending order od PCI value
#print(len(si1))


# # zone 1. We have ascending order by PCI. Now delete neighbour edges 

# In[ ]:


node_degree_general = np.array(list(G.degree))
#print(node_degree_general[5,1])

count = 0
si2 = []
sj2 = []


for i in si1:
    if i not in si2: # or node_degree_general[si1[count],1] < 3:
        si2.append(i)
        sj2.append(sj1[count])
    count = count + 1
    
#print(si1) 
#print(sj1)

count = 0
si = []
sj = []
for i in sj2:
    if i not in sj: # or node_degree_general[sj2[count],1] < 3:
        sj.append(i)
        si.append(si2[count])
    count = count + 1
    
#print(si) 
#print(sj)


# # Sequence of the roads by custom PCI value

# In[ ]:


import copy
modified_PCI_matrix = copy.deepcopy(PCI_matrix)


# In[ ]:


#choose 10000m route with least PCI and maintain

#find the first smallest PCI


si1 = []
sj1 = []


for i in range (number_of_node):
    for j in range (number_of_node):
        if C[i,j] == 1 and i<j:
            if modified_PCI_matrix[i,j] > 55 and modified_PCI_matrix[i,j] < 70:
                si1.append(i)
                sj1.append(j)

                modified_PCI_matrix[ii,jj] = 0
    
    ni = G.degree[ii]
    nj = G.degree[jj]
    
    
#print(si1,sj1)  #array with ascending order od PCI value
#print(len(si1))


# # We have a sequence by custom PCI value. Now delete neighbour edges

# In[ ]:


node_degree_general = np.array(list(G.degree))
#print(node_degree_general[5,1])

count = 0
si2 = []
sj2 = []


for i in si1:
    if i not in si2: # or node_degree_general[si1[count],1] < 3:
        si2.append(i)
        sj2.append(sj1[count])
    count = count + 1
    
#print(si1) 
#print(sj1)

count = 0
si = []
sj = []
for i in sj2:
    if i not in sj: # or node_degree_general[sj2[count],1] < 3:
        sj.append(i)
        si.append(si2[count])
    count = count + 1
    
#print(si) 
#print(sj)


# # Color map representation of routes in maintenance by its PCI

# In[ ]:


Nroutes = len(si)  #number_of_node
route = []
route_color = []
route_linewi = []

for iroute in range(Nroutes):
    orig = list(G)[si[iroute]]
    dest = list(G)[sj[iroute]]
    ro = nx.shortest_path(G, orig, dest, weight='length')
    route.append(ro)
    route_linewi.append(2)
    i = si[iroute]
    j = sj[iroute]
    if PCI_matrix[i,j] < 50.0:
        route_color.append("r")
    if PCI_matrix[i,j] >= 50.0 and PCI_matrix[i,j] < 75.0:
        route_color.append("y")
    if PCI_matrix[i,j] >= 75.0:
        route_color.append("g")


# plot the routes
fig, ax = ox.plot_graph_routes(G, routes = route, route_linewidths=route_linewi ,orig_dest_size =10, route_colors = route_color, node_size=0, edge_linewidth=0.6)

figure = plt.gcf()
plt.show()
#fig.savefig('scenario3map.png', dpi =300,  bbox_inches='tight')


# # Efficiency

# In[ ]:


status_in_maint = copy.deepcopy(C)
degraded_travel_time_matrix, degraded_velosity_matrix = degraded_time(PCI_matrix)
drop_degraded_travel_time_matrix = copy.deepcopy(degraded_travel_time_matrix)
drop_degraded_velosity_matrix = copy. deepcopy(degraded_velosity_matrix)


# In[ ]:


coordinate= -1


# In[ ]:


#efficiency travel time
def efficiency():

    sum=0
    for i in range (number_of_node):
        for j in range (number_of_node):
            if i<j and  travel_time_matrix[i,j] !=0.0 :
                
                sum = sum + (1/travel_time_matrix[i,j]) 
    #N number of nodes in network
    N = number_of_node
    #Network weighted efficiency
    normal_eff = sum/(N*(N-1))

    #efficiency degraded travel time
    sum=0
    for i in range (number_of_node):
        for j in range (number_of_node):
            
            if i<j and degraded_travel_time_matrix[i,j] !=0.0 :
               
                sum = sum + (1/degraded_travel_time_matrix[i,j])      

    #N number of nodes in network
    N= number_of_node
    #Network weighted efficiency
    degraded_eff = sum/(N*(N-1))
    #print('degraded sum', sum)

    
#++++++++++++++++++++++++++++++++++  efficiency after traffic load redistribution ++++++++++++++++++++++++++++++++++++++   
    #efficiency drop degraded travel time
    #drop_degraded_travel_time_matrix = drop_degraded_time(coordinate)
    sum=0
    for i in range (number_of_node):
        for j in range (number_of_node):
            if i<j and drop_degraded_travel_time_matrix[i,j] !=0.0 :
                
                sum = sum + (1/drop_degraded_travel_time_matrix[i,j]) 
    #N number of nodes in network
    N= number_of_node
    #Network weighted efficiency
    drop_degraded_eff = sum/(N*(N-1))
    #print('drop sum', sum)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    eff_resut_degraded = (100*degraded_eff)/normal_eff
    eff_resut_drop_degraded = (100*drop_degraded_eff)/normal_eff

    #print(result)
    return eff_resut_degraded, eff_resut_drop_degraded


# In[ ]:


degraded, drop = efficiency()
print('degraded efficiency before maintenance: ', degraded)
print('drop degraded efficiency before maintenance: ', drop)
print('average PCI before maintenance:', np.sum(PCI_matrix)/number_of_edge)


# # +Maintenance initial confition, which routes should be maintenanced first+

# In[ ]:


status_in_maint = copy.deepcopy(C)
status_duration = np.zeros((number_of_node,number_of_node))

#1 if not in maintenance
#2 if in maintenance


# In[ ]:


#betweenness centrality measure
def bc_node_coeff(node_int):
    # know centrallity coefficient of the node point
    bc = nx.betweenness_centrality(ox.get_digraph(G), weight="travel_time")
    nodeidd =  node_int #node_int_to_id(node_int)
    key = bc[nodeidd]
    first_key = list(bc)[0]
    centrality_coeff = list(bc.values())[node_int]    
    #print(centrality_coeff)
    return key


# In[ ]:


bc_node_coeff(40)


# In[ ]:


#closeness centrality measure
def cc_node_coeff(node_int):
    cc = nx.closeness_centrality(G)
    key = cc[node_int]
    return key


# In[ ]:


cc_node_coeff(40)


# In[ ]:


flag = 0 # equal redistribution
flag = 1 # redistribution by betweeness centrality of neighbor nodes


# In[ ]:


flag = 1
i = int(0)
eff_drop = []
eff = []
coordinate = -1
e, ed = efficiency()
eff.append(e) # 1st condition of efficiency
eff_drop.append(ed)
while ed > 70:  # boundary condition for effciency is 75%
    
    #identify road lenght and define maintenance duration
    route_lenght = travel_distance_matrix[si[i],sj[i]]
    if PCI_matrix[si[i],sj[i]] > 25 and PCI_matrix[si[i],sj[i]] <=40:
        status_duration[si[i], sj[i]] = (route_lenght/1000)* (100/(24*360)) # in year
    if PCI_matrix[si[i],sj[i]] > 40 and PCI_matrix[si[i],sj[i]] <=55:
        status_duration[si[i], sj[i]] = (route_lenght/1000)* (75/(24*360)) # in year
    if PCI_matrix[si[i],sj[i]] > 55 and PCI_matrix[si[i],sj[i]] <=70:
        status_duration[si[i], sj[i]] = (route_lenght/1000)* (55/(24*360)) # in year
    if PCI_matrix[si[i],sj[i]] > 70 and PCI_matrix[si[i],sj[i]] <=85:
        status_duration[si[i], sj[i]] = (route_lenght/1000)* (13/(24*360)) # in year
    if PCI_matrix[si[i],sj[i]] > 85 and PCI_matrix[si[i],sj[i]] <=100:
        status_duration[si[i], sj[i]] = (route_lenght/1000)* (3/(24*360)) # in year
    
    degraded_travel_time_matrix[si[i],sj[i]] = math.inf
    PCI_matrix[si[i],sj[i]] = 15
    #---------------------------------------------------
    #coordinate = i
    coordinate = -1
    
    node1int = si[i]
    node2int = sj[i]
    node1id = node_int_to_id(node1int)
    node2id = node_int_to_id(node2int)
    node1nes = node_neighbour_by_id(node1id)
    node2nes = node_neighbour_by_id(node2id)
    
    #+_+_+equal redistribution coefficient
    re_route = len(node1nes) + len(node2nes) - 2
    vm = copy.deepcopy(drop_degraded_travel_time_matrix[si[i], sj[i]])
    dm = FlowDensity(vm)
    Fm = dm * vm
    deltadensity = dm/2
    #+_+_
    coeffs1 = 0
    #traffic redistribution
    bc = nx.betweenness_centrality(ox.get_digraph(G), weight="travel_time")
    for q1 in node1nes:
        if q1 != node2id:
            nid1 = node_id_to_int(q1)
            coeffs1 = coeffs1 + bc_node_coeff(nid1)
            
    coeffs2 = 0
    for q2 in node2nes:
        if q2 != node1id:
            nid2 = node_id_to_int(q2)
            coeffs2 = coeffs2 + bc_node_coeff(nid2)
    
    
    partion = 1
    for s in node1nes:
        if s != node2id:
            ints1 = node_id_to_int(s)
            if ints1 < node1int and drop_degraded_travel_time_matrix[ints1, node1int] != math.inf:
                vn = drop_degraded_velosity_matrix[ints1, node1int]
                dn = FlowDensity(vn)
                Fn = dn*vn
                
                if flag == 0:
                    deltadensity = deltadensity/(len(node1nes)-1)
                if flag == 1:
                    if coeffs1 != 0:
                        partion = (bc_node_coeff(ints1)) / coeffs1
                    if coeffs1 == 0:
                        partion = 0.01
                    deltadensity = deltadensity * partion
                    
                dnew = dn + deltadensity
                #dnew = dnew/lane_matrix[ints1, node1int]
                v = Fn/dnew 
                drop_degraded_velosity_matrix [ints1, node1int] = v #travel_time_matrix[ints1, node1int] * 1
                drop_degraded_travel_time_matrix [ints1, node1int] = 3.6 * travel_distance_matrix [ints1, node1int]/drop_degraded_velosity_matrix [ints1, node1int]

            if ints1 > node1int and drop_degraded_travel_time_matrix[node1int, ints1] != math.inf:
                vn = drop_degraded_velosity_matrix[node1int, ints1]
                dn = FlowDensity(vn)
                Fn = dn*vn
                if flag == 0:
                    deltadensity = deltadensity/(len(node1nes)-1)
                if flag == 1:
                    if coeffs1 != 0:
                        partion = (bc_node_coeff(ints1)) / coeffs1
                    if coeffs1 == 0:
                        partion = 0.01
                    deltadensity = deltadensity * partion
                dnew = dn + deltadensity
                #dnew = dnew/lane_matrix[node1int, ints1]
                v = Fn/dnew 
                drop_degraded_velosity_matrix[node1int, ints1] = v #travel_time_matrix[node1int, ints1] * 1.3
                drop_degraded_travel_time_matrix[node1int, ints1] = 3.6 * travel_distance_matrix[node1int, ints1]/drop_degraded_velosity_matrix[node1int, ints1]
    for q in node2nes:
        if q != node1id:
            ints2 = node_id_to_int(q)
            if ints2 < node2int and drop_degraded_travel_time_matrix[ints2, node2int] != math.inf:
                vn = drop_degraded_velosity_matrix[ints2, node2int]
                dn = FlowDensity(vn)
                Fn = dn*vn
                if flag == 0:
                    deltadensity = deltadensity/(len(node2nes)-1)
                if flag == 1:
                    if coeffs2 != 0:
                        partion = (bc_node_coeff(ints2)) / coeffs2
                    if coeffs2 == 0:
                        partion = 0.01
                    deltadensity = deltadensity * partion
                dnew = dn + deltadensity
                #dnew = dnew/lane_matrix[ints2, node2int]
                v = Fn/dnew 
                drop_degraded_velosity_matrix[ints2, node2int] = v #travel_time_matrix[ints2, node2int] * 1.3
                drop_degraded_travel_time_matrix[ints2, node2int] = 3.6 * travel_distance_matrix[ints2, node2int]/drop_degraded_velosity_matrix[ints2, node2int]
            if node2int < ints2 and drop_degraded_travel_time_matrix[node2int, ints2] != math.inf:
                vn = drop_degraded_velosity_matrix[node2int, ints2]
                dn = FlowDensity(vn)
                Fn = dn*vn
                if flag == 0:
                    deltadensity = deltadensity/(len(node2nes)-1)
                if flag == 1:
                    if coeffs2 != 0:
                        partion = (bc_node_coeff(ints2)) / coeffs2
                    if coeffs2 == 0:
                        partion = 0.01
                    deltadensity = deltadensity * partion
                dnew = dn + deltadensity
                #dnew = dnew/lane_matrix[node2int, ints2]
                v = Fn/dnew 
                drop_degraded_velosity_matrix[node2int, ints2] = v #travel_time_matrix[node2int, ints2] * 1.3
                drop_degraded_travel_time_matrix[node2int, ints2] = 3.6 * travel_distance_matrix[node2int, ints2]/drop_degraded_velosity_matrix[node2int, ints2]

    drop_degraded_travel_time_matrix[si[i], sj[i]] = math.inf
    #---------------------------------------------------
    status_in_maint[si[i], sj[i]] = int(2)    #changing from not in maintaenance to in maintenance
    e, ed = efficiency()
    print('degraded eff', e)
    print('drop eff', ed)
    i = i + 1
    
eff.append(e) # 2nd condition of efficiency (just after starting mantenance in the routes)
eff_drop.append(ed)
number_of_maintaining_routes = copy.deepcopy(i)


# In[ ]:


# matrix to excel 
marks_data = pd.DataFrame(drop_degraded_travel_time_matrix)
file_name = 'drop_degraded_travel_time_matrix.xlsx'

# saving the excel
marks_data.to_excel(file_name)

# matrix to excel 
marks_data = pd.DataFrame(status_duration)
file_name = 'status_duration.xlsx'
  
# saving the excel
marks_data.to_excel(file_name)

# matrix to excel 
marks_data = pd.DataFrame(status_in_maint)
file_name = 'status_in_maint.xlsx'
  
# saving the excel
marks_data.to_excel(file_name)


# In[ ]:


print("Maintenance has started in :",number_of_maintaining_routes+1, "routes" )
drop_PCI_matrix = copy.deepcopy(PCI_matrix)


# # Maintenance measure

# In[ ]:


#Loop for a year. dt = 10 days. Loop over 1 year = 360 days.

period = 700 #days = year 
dt = 10  # 10 days
ooops = int(period/dt) # check every ooops days
per = []
counter = 0
for i in range (ooops):
    
    
    a = 1385.406
    b = 15.985

    # upgrade age of the pavement in each ooops days
    for l in range(number_of_node): 
        for v in range(number_of_node):
            if status_in_maint[l,v] != 2 and C[l,v] == 1 and l<v and PCI_matrix[l,v] >= 25 and PCI_matrix[l,v] <= 86:
                age_matrix[l,v] = age_matrix[l,v] + (dt*24)/8640
                PCI_matrix[l,v] = 152.75-(6.17*age_matrix[l,v])-(0.0091*AAS)-(2.76*AART)+(0.0570*AART*AART)
                if PCI_matrix[l,v] > 99:
                    PCI_matrix[l,j] = 99
                if PCI_matrix[l,v] < 26:
                    PCI_matrix[l,v] = 26
                
                
                deltav = ((a/PCI_matrix[l,v])-b)
                degraded_velosity_matrix[l,v] = travel_velosity_matrix[l,v]-deltav
                degraded_travel_time_matrix[l,v] = 3.6*travel_distance_matrix[l,v]/degraded_velosity_matrix[l,v]
                    
                    
    #degraded_travel_time_matrix, degraded_velosity_matrix = degraded_time(PCI_matrix)
    #drop_degraded_travel_time_matrix, drop_degraded_velosity_matrix = drop_time(PCI_matrix_new, drop_degraded_travel_time_matrix,drop_degraded_velosity_matrix)

    #---------------
    
    
                
    # improve pci if maintenance is completed otherwise wait
    # worsen other routes condition by formula again.
    # update 1. degraded time = inf, 2. degraded time improved for maintenanced route, 3. degrade time worsened for other routes.
    for j in range (number_of_maintaining_routes): #should be for all edge but for start did only for edges in maintenance
        if status_duration[si[j], sj[j]] > 0 and status_in_maint[si[j], sj[j]]==2 :    
            status_duration[si[j], sj[j]] = status_duration[si[j], sj[j]] - (10*24)/8640 #10 days in year
        if status_duration[si[j], sj[j]] < 0 and status_in_maint[si[j], sj[j]]==2 :
            status_duration[si[j], sj[j]] = 0
            status_in_maint[si[j], sj[j]] = 1
            PCI_matrix[si[j], sj[j]] = 100
            degraded_travel_time_matrix[si[j], sj[j]] = travel_time_matrix[si[j], sj[j]]
            #---------------------------
            coordinate = -1
            drop_degraded_travel_time_matrix[si[j], sj[j]] = travel_time_matrix[si[j], sj[j]]
            node1int = si[j]
            node2int = sj[j]
            node1id = node_int_to_id(node1int)
            node2id = node_int_to_id(node2int)
            node1nes = node_neighbour_by_id(node1id)
            node2nes = node_neighbour_by_id(node2id)
            for s in node1nes:
                if s != node2id:
                    ints1 = node_id_to_int(s)
                    if ints1 < node1int:
                        drop_degraded_travel_time_matrix[ints1, node1int] = degraded_travel_time_matrix[ints1, node1int]
                    if ints1 > node1int:
                        drop_degraded_travel_time_matrix[node1int, ints1] = degraded_travel_time_matrix[node1int, ints1]
            for q in node2nes:
                if q != node1id:
                    ints2 = node_id_to_int(q)
                    if ints2 < node2int:
                        drop_degraded_travel_time_matrix[ints2, node2int] = degraded_travel_time_matrix[ints2, node2int]
                    if node2int < ints2:
                        drop_degraded_travel_time_matrix[node2int, ints2] = degraded_travel_time_matrix[node2int, ints2]
            
            #---------------------------
            e, ed = efficiency()
            print('degraded eff:', e)
            print('drop eff:', ed)
            eff.append(e)
            eff_drop.append(ed)
            per.append(counter)
            counter = counter+10
      
    
    


# In[ ]:


a = 10*len(eff)
print('The maintenance has completed in:', a, 'days')


# In[ ]:


counter = counter+1
per.append(counter)
counter = counter+1
per.append(counter)


# In[ ]:


from matplotlib.pyplot import figure
print('average PCI after maintenance:', np.sum(PCI_matrix)/number_of_edge)

figure(figsize=(8, 6), dpi=80)
plt.title('Efficiency tendency')
plt.xlabel('period in days')
plt.ylabel('efficieny of the network')

plt.plot(per, beff, 'b', linewidth=1, label = 'The efficiency tendency without traffic redistribution')
plt.plot(per, ceff, 'r', linewidth=1, label = 'The efficiency tendency with traffic redistribution')
#plt.axis([180, 280, 85, 100])
plt.legend()
plt.savefig('scenario3 curve', dpi =500,  bbox_inches='tight')
plt.show()


# ### Node degree test

# # Find the node in the map :)

# In[ ]:


orig, dest = list(G)[0], list(G)[-1]


orig = list(G)[60]


nc = ["r" if node == orig else "w" for node in G.nodes]
ns = [80 if node == orig else 15 for node in G.nodes]
fig, ax = ox.plot_graph(G, node_size=ns, node_color=nc, node_zorder=2)


# # Centralities

# ### Closeness centrality

# In[ ]:


centrality_closeness = nx.closeness_centrality(G)

s = centrality_closenss = pd.DataFrame.from_dict(centrality_closeness, orient='index',columns=['centrality'])

#print(s)

#find the max index of the centrality
s.iat[0,0]
max_closeness_c_index = 0.0
node_most_central_closeness =1

for i in range (number_of_node):
    if s.iat[i,0] > max_closeness_c_index:
        node_most_central_closeness = i
        max_closeness_c_index = s.iat[i,0]
       

#print(node_most_central_closeness)   


# ### Betweennes centrality

# In[ ]:


# calculate betweenness with a digraph of G (ie, no parallel edges)
bc = nx.betweenness_centrality(ox.get_digraph(G), weight="travel_time")
max_node, max_bc = max(bc.items(), key=lambda x: x[1])
max_node, max_bc
print(type(bc))

# the highest betweenness
nc = ["r" if node == max_node else "w" for node in G.nodes]
ns = [80 if node == max_node else 15 for node in G.nodes]
fig, ax = ox.plot_graph(G, node_size=ns, node_color=nc, node_zorder=2)

# add the betweenness centraliy values as new node attributes, then plot
nx.set_node_attributes(G, bc, "bc")
print(bc)

