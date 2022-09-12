# Code for Path Integral Stochastic Optimal Control for Sampling Transition Paths

This is the codebase accompanying the _Path Integral Stochastic Optimal Control for Sampling Transition Paths_ paper.

For each molecule evaluated in the paper we provide a separate test file (both in general .py and notebook format).
These generally follow the same structure.

## Codebase structure

In addition to the available scripts for learning the policies for the different molecular structures, the codebase has
the following structure:

- /potential: Contains wrappers for the OpenMM interface. Each wrapper handles the initialization of the OpenMM
  configuration for each molecule.
- /Dynamics: Implementation of the dynamics as described by equation 4 and 5 in the paper. Includes abstract methods for
  implementing f, G and phi according to the problem specification.
- /policies: Multiple simple Neural Network policies
- /solvers: Contains the main implementation of the PICE training algorithm.

### Implementation of OpenMM

To understand the interaction between our PICE and OpenMM implementation is it important to note that we have
implemented PICE to work with the generic linear dynamics description of equation 4 and 5 in the paper. Ie. the
uncontrolled molecular dynamics (f) and controlled policy (G(u)) are calculated independently and then added to get the
next state of the system.

However, this structure does not work well within the optimized setup of the OpenMM framework. To make most efficient
use of the functionality of OpenMM, we thus decided to have OpenMM keep track of the state of the system, and only
providing it with the bias potential at each step.

To highlight how this is realized, we have supplied a standalone jupyter notebook **(ExampleAlanine.ipynb)** that shows the
interaction between OpenMM and PICE in a simplified manner. 

### Contact information
For further question, please reach out to larsholdijk@gmail.com or open an issue. 

