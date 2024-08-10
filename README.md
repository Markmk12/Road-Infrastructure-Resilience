# Bachelor Thesis - Study Topic: Identification of Infrastructure Life-Cycle Strategies Based on Resilience

Road networks play an important role in the transportation of people and goods. They
are critical infrastructure, especially for a nation’s economy. In Germany, nearly 80
percent of goods are transported by road. Therefore, high demands are made on maintenance to ensure the efficiency and resilience of a road network, which results in high
maintenance costs. This bachelor thesis proposes a decision making approach that allows authorities to select a cost-effective maintenance strategy considering a resilience
threshold. The strategies are a combination of preventive and corrective maintenance,
with varying intensities over time. To calculate resilience for each strategy, a Monte
Carlo simulation is utilized. The pavement degradation of the roads is modeled by
a linear combination of two gamma processes. This approach is applied to different
road networks in Germany. The Python package OSMnx, which can extract network
topologies from Open Street Map, is utilized. The results are discussed from an engineering perspective in terms of feasibility and computational complexity. Additionally,
the thesis provides an outlook on future challenges and potential developments.

Reference:
[1] Salomon, Julian, et al. "Resilience decision-making for complex systems."
ASCE-ASME J Risk and Uncert in Engrg Sys Part B Mech Engrg 6.2 (2020).

Keywords: Resilience; Monte Carlo Simulation; Maintenance Strategies; Road Network; Pavement Degradation

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
