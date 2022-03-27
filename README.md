# Wayback Machine
This is a repository for the paper `Wayback Machine: A tool to capture the evolutionary behavior of the bug reports and their triage process in open-source software systems`. A draft of the paper can be found on Arxiv: (https://arxiv.org/pdf/2011.05382.pdf).

The paper is published in Journal of Systems and Software (https://www.sciencedirect.com/science/article/abs/pii/S0164121222000565). 

## Folders and their contents 

### bin
It includes the bug dependency graph (`BDG`), defined in the paper. 

It includes graph operations, e.g., adding or removing arcs and nodes, together with graph-related updates, e.g., updating depth, degree, severity, and priority of the bugs in the BDG.

### components
It includes two main classes: **developers** and **bugs**. 
* `assignee.py` has the `Assignee` class which includes the information of the developers and track the assigned bugs to them and the accuracy of those assignments. It also keeps their LDA experience and the time limit $L$ of them.
* `bug.py` has the `Bug` class which includes all the essential information of each bugs, including ID, severity, priority, depth, degree, status, summary, description, fixing time, and so on. It has its methods to track the assigned developer and assignment time, compute the accuracy of the assignment, check the validity of the bugs for assignment based on preprocessing steps in the paper, update its blocking information, and change its status to fixed or reopenned. 


### dat
It includes all the datasets used in the paper. The datasets are related to the extracted bugs from three software projects, Mozilla, LibreOffice, and EclipseJDT.

### imgs
It includes the images used in the paper in a vector format.

### simulator
This folder contains two important files: `main.py` and `wayback.py`.

*  `wayback.py` codes the process of the Wayback machine and its elements. The main variables are
  *  `keep_track_of_resolved_bugs` which keeps all the info related to the resolved bugs during the testing phase.
  *  `track_BDG_info` keeps track of the BDG during the life span of the project.
  *  `verbose` defines how to print the output during the running time, e.g.. `nothing`, `some`, or `all` the information should be printed.
  It has also some important methods, including
  * `acceptable_solving_time` which determines the acceptable solving time based on the IQR.
  * `possible_developers` which finds the list of feasible developers at the end of the training phase.
  * `fixing_time_calculation` which uses bug info and evolutionary database to calculate fixing time according to the Costriage paper.
  * `track_and_assign` which assigns the bugs to proper developers and tracks the info of the assigned/fixed bug.
  * `triage` module to apply triage algorithms. Researchers can manipulate this method and add their own triage algorithms to the Wayback Machine. `DABT` `RABT`, `CosTriage`, `CBR`, `Actual` and `Random` triage are already implemented.
  * `prioritization` module to apply prioritization algorithms. Researchers can manipulate this method and add their own prioritization algorithms to the wayback machine. `max_priority` `max_severity`, `cost_estimation`, `priority_estimation`, `cost_priority_estimation`, `max_depth_degree`, `Actual` and `Random` prioritization are already implemented.


* `main.py` is needed to run the Wayback Machine. 
To run the code, a sample command might be as follows. 

```python
python simulator/main.py --project=Mozilla --resolution=max_depth_degree --n_days=7511 --prioritization_triage=prioritization --verbose=0
```
Regarding the options available for the `main.py` file:
  * `--resolution` defines the **strategy/algorithm** to take
  * `project` can be `Mozilla`, `LibreOffice`, or `EclipseJDT`. A user can also extract and add their own ITS database. 
  * `--n_days` defines the number of days from the beginning to the end of the lifespan. Based on our database, it should be 3438 days for LibreOffice,  7511 days for Mozilla, and 
  6644 days for EclipseJDT.
  * `prioritization_triage` can be set to either prioritization or triage, based on the resolution selected.
  * `verbose` indicates how to print the output and can be either: ```[0, 1, 2, nothing, some, all]```.


More details on the simulator are commented on in the files.

### utils
It contains `attention_decoder` for the DeepTriage strategy. `debugger` to search over the variables in case of a bug. `functions` which includes useful, fundamental functions. `prerequisites` including all the packages needed to run the Wayback Machine. `release_dates` that holds the release dates of the projects during their testing phase. If a user wants to add a new project, they have to manually add the release dates here. `report` gives a full report of the Wayback Machine outputs if needed.


## Prerequisites:
 * networkx 
 * random
 * tqdm
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
 * matplotlib
 * gensim 
 * nltk 
 * sklearn 
 * tensorflow
 * gurobipy 
 * plotly

____________
The output of each run will be saved in the output folder automatically. 

Any questions? Please do not hesitate to contact me: hadi . jahanshahi [at] ryerson.ca
