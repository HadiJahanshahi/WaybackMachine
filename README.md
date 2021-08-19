# Wayback Machine
This is a repository for the paper `Wayback Machine: A tool to capture the evolutionarybehaviour of the bug reports and their triage process inopen-source software systems`. 

## Folders and their contents 

### bin
It includes the bug dependency graph (`BDG`), defined in the paper. 

It includes graph operations, e.g., adding or removing arcs and nodes, together with graph related updates, e.g., updating depth, degree, severity, and priority of the bugs in the BDG.

### dat
It includes all the datasets used in the paper. The datasets are related to the extracted bugs from three software projects, Mozilla, LibreOffice, and EclipseJDT.

### imgs
It includes the images used in the paper in a vector format.

### simulator
This folder contains two important files: `main.py` and `wayback.py`.

* `main.py` is needed to run the Wayback Machine. 
*  `wayback.py` codes whole the process of the wayback machine and its elements. The main variables are
  *  `keep_track_of_resolved_bugs` which keeps all the info related to the resolved bugs during the testing phase.
  *  `track_BDG_info` which keep track of the BDG during the life span of the project.
  *  `verbose` which defines how to print the output during the running time, e.g.. `nothing`, `some`, or `all` the information should be printed.
  It has also some important methods, including
  * `acceptable_solving_time` which determines the acceptable solving time based on the IQR.
  * `possible_developers` which finds the list of feasible developers at the end of the training phase.
  * `fixing_time_calculation` which uses bug infor and evolutionary database to calculate fixing time based on the Costriage paper.
  * `track_and_assign` which assigns the bugs to proper developers and track the info of the assigned/fixed bug.
  * `triage` module to apply triage algorithms. Researchers can manipulate this method and add their own triage algorithms to the wayback machine. `DABT` `RABT`, `CosTriage`, `CBR`, `Actual` and `Random` triage are already implemented.
  * `prioritization` module to apply prioritization algorithms. Researchers can manipulate this method and add their own prioritization algorithms to the wayback machine. `max_priority` `max_severity`, `cost_estimation`, `priority_estimation`, `cost_priority_estimation`, `Actual` and `Random` prioritization are already implemented.



The folder `output` includes the output of experiments under different conditions. The folder `scripts` includes all python scripts related to the paper. More details on that is given in the folder's readme. 

Prerequisites:
 * networkx 
 * random
 * tqdm import tqdm 
 * time
 * collections
 * statistics
 * numpy
 * pandas
 * datetime
 * os
 * ast
 * json
 * pickle
 * copy


To run the code, this command should be run. 

```python
python Main.py --stra=random --run=5 --n_develop=3
```

``--stra`` defines the **strategy** to take and ``--run`` defines the number of repetitions. ``n_develop`` indicates the number of developers in the system.

The options for ``stra`` are:
* childern_degree 
* childern_severity 
* max_severity
* max_degree
* max_depth
* max_degree_plus_max_depth
* max_degree_plus_severity
* random

The output of each run will be saved in output folder automatically. 
