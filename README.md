# Bachelor Thesis - Study Topic: Identification of Infrastructure Life-Cycle Strategies Based on Resilience

Abstract of the Task Sheet:
The increasing size and complexity of technical systems that are critical to the stable wealth
and development of modern societies requires new methodologies in engineering to
quantify their resilience [1]. In order to address the complexity of such a process,
researchers are challenged to develop sophisticated numerical and high-end computational
tools to provide an adequate basis for comprehensive decision-making in terms of the
authorities’ strategies. Consequently, current research aims for encompassing and rigorous
simulation frameworks for complex systems, such as infrastructure networks.
This bachelor’s thesis is concerned with the establishment of a simulation framework for an
road infrastructure network in its fundamental form. Correspondingly, a manuscript and
presentation should illustrate the theoretical validation of the applied models and a
numerical study of feasibility in terms of computational efficiency and complexity. The
student should discuss the challenges from an engineering perspective and give an outlook
for future challenges and next developments.

Reference:
[1] Salomon, Julian, et al. "Resilience decision-making for complex systems."
ASCE-ASME J Risk and Uncert in Engrg Sys Part B Mech Engrg 6.2 (2020).

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Contact](#contact)

## Introduction (Very short description of the approach. A more detailed one can be found in the bachelor thesis.)

The goal of the bachelor thesis is to create a simulation framework for a road network in its basic form and to find 
life cycle strategies (more precisely, maintenance strategies over time) that are based on a certain level of resilience and their costs.

Maintenance can be divided into two categories. One is preventive maintenance and the other is corrective maintenance. 
The quality of maintenance is divided into three categories (sparse, moderate, expensive). 

Due to the fact that the brute force method is used to find the optimal maintenance strategies, the simulation scope is limited. 
A period of 30 years is considered with a time step of one year. This period is divided into three equal intervals. 
In each time interval, a combination of the quality levels for Preventive and Corrective Maintenance is used.
For instance, one of the strategies is (sparse, moderate), (moderate, moderate), (moderate, extensive).
For each strategy, the network efficiency is sampled n times and the resilience is determined for the entire period.
At the end of the program, the most appropriate strategy is selected based on resilience and expected costs.

Network Efficiency depends on the travel time from each edge. The travel time on the respective edge is in turn dependent 
on the Pavement Condition Index (PCI) and whether any maintenance measures take place on the edge. The deterioration of 
the pavement is determined by a weighted linear combination of degradation processes (two Gamma Processes). One of the processes 
represents deterioration due to traffic and the other represents deterioration due to environment.

## Installation

The Python (3.10) code was created and tested under Windows 10.
Simply unzip the project folder and open it with an IDE (e.g. PyCharm) or any other program you like.
The IDE should automatically read the requirements.txt (it contains all package dependencies).

## Usage

- Extract a road network structure with the package `network_import/`. You can create a fictional graph with `networkx_import.py` 
or retrieve a graph based on Open Street Map Data with `osmnx_import.py`. The graph is automatically saved in a subfolder.
For the exact procedure see the comments in the corresponding file
- In `main.py` the simulation takes place. Import a road network and set some simulation conditions (e.g. sample size etc.).
The results are stored in `results/`. Likewise, the results are displayed in the terminal.
- See the comments and the docstring for more information.

## File Structure

- `Archive/` - Old files from the development process. This file can be ignored.
- `experiments/` - Some experiments to estimate the parameters of the program. This folder does not affect the main program.
- `function_library/` - Collection of functions.
- `network_import/` - This package is used to extract road networks from OSM.
- `results/` - The results from main.py are stored here.
- `main.py` - This is the main program to find maintenance strategies based on resilience and costs.
- `requirements.txt` - Contains all package dependencies.

## Contact

- Mark Hoyer
- mark.hoyer@stud.uni-hannover.de

---

© 2023, Mark Hoyer
