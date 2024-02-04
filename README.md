# Ego-Group-Partition
Code for paper "Ego Group Partition: A Novel Framework for Improving Ego Experiments in Social Networks".

## Description

### Files
- simulation.py: simulation to compare different ego approaches.
- sigma_distribution.ipynb: show sigma distribution for different ego approaches.
- simulation_results.ipynb: show simulation results.
  
### Folders
- dataset: one network topology dataset used for simulations.
- src: simulation function, ego cluster and egp.
  
### Usage
To run simulations:
```
python3 simulation.py
```

use simulation_results.ipynb to show simulation results.

### Dependencies
- Networkx >= 2.8.4
- Scipy >= 1.9.3
- Numpy >=1.23.4
- Pandas >= 1.5.1
