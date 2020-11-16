# bug-triage
This is a temporary repository for the paper submitted to PROMISE'20, and is use for blind submission. After the paper is accepted, all the materials will be moved to the permanent repository.

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
