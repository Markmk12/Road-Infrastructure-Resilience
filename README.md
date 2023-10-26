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
- [Input & Output](#input--output)
- [Testing](#testing)
- [Contact](#contact)

## Introduction

The aim of the bachelor thesis is to create a simulation framework for a road network in its basic form.
With the help of this simulation framework, life cycle strategies (more precisely, maintenance strategies) are to be identified,
which are evaluated on the basis of a certain resilience level and their costs.
Maintenance is divided into two categories. One is preventive maintenance and the other is corrective maintenance. 
The quality of maintenance is divided into three categories (sparse, moderate, expensive).

## Installation

The Python (3.10) code was created and tested under Windows 10.
Simply unzip the project folder and open it with an IDE (e.g. PyCharm) or any other program you like.
The IDE should automatically read the requirements.txt (it contains all package dependencies).

## Usage

- How to run the code.
- Example commands and their expected outputs.

## File Structure

- `Archive/` - Old files from the development process. This file can be ignored.
- `experiments/` - Some experiments to estimate the parameters of the program. This folder does not affect the main program.
- `function_library/` - Collection of functions.
- `network_import/` - This package is used to extract road networks from OSM.
- `results/` - The results from main.py are stored here.
- `main.py` - This is the main program to find maintenance strategies based on resilience and costs.
- `requirements.txt` - Contains all package dependencies.

## Input & Output

Description of the expected inputs and produced outputs.

## Testing

How to run tests and what they cover.

## Contact

- Mark Hoyer
- mark.hoyer@stud.uni-hannover.de

---

© 2023, Mark Hoyer
