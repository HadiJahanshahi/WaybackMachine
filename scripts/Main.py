from prerequisities_ import * #packages and these functions: isNaN, isnotNaN, 
                                                            #convert_string_to_array, mean_list, and string_to_time
#%matplotlib inline

#######################################
##                                   ##
##      author: XXXXXXX     ##
##     XX.XX@XX.XX    ##
##          XXX XX XX         ##
##                                   ##
#######################################

#######################################
#######################################
#####  Graph generating functions  ####
#######################################
####################################### 
def pred_time_to_solve (time, median_time, max_time, seed_ , base_ = 10):
    '''
    Predicting a solution time (rule-based approach)
    '''
    np.random.seed(int(seed_+1))
    if time == 0:
        return timedelta(np.random.rand(1)[0])
    elif time <= median_time:
        return timedelta((time/median_time)*base_) # between 0 to 10
    else:
        return timedelta(base_ + (time/max_time)*base_*4) # max 10 to 50

path_ = os.path.join(os.getcwd(),"..", "output", "Whole_dataset_total_updated.csv")
Whole_dataset = pd.read_csv(path_)
Whole_dataset.guassian_time_to_fix = [pd.Timedelta(i) for i in Whole_dataset.guassian_time_to_fix]
Whole_dataset.time_to_fix = [pd.Timedelta(i) for i in Whole_dataset.time_to_fix]
Whole_dataset["creation_time"] = [string_to_time(i, '%Y-%m-%d %H:%M:%S') for i in Whole_dataset["creation_time"]]
Whole_dataset["cf_last_resolved"] = [string_to_time(i, '%Y-%m-%d %H:%M:%S'
                                                   ) if not isNaN(i) else None for i in Whole_dataset["cf_last_resolved"]]

Whole_dataset = Whole_dataset.set_index(['id']) 

day_list = [i.days for i in Whole_dataset.time_to_fix]
median_solving_time = np.median(day_list)
mean_solving_time = np.mean(day_list)
max_solving_time = max(day_list)
Whole_dataset['pred_time_to_solve'] = None
Whole_dataset['pred_time_to_solve'] = [pred_time_to_solve(v, median_solving_time, max_solving_time, seed_=idx,
                                                         base_ = 5) for idx, v in enumerate(day_list)]
Whole_dataset['n_depends_on'] = Whole_dataset['depends_on'].apply(convert_string_to_array).apply(len)
Whole_dataset['n_blocks'] = Whole_dataset['blocks'].apply(convert_string_to_array).apply(len)

def sum_dict (dic1 , dic2):
    """
    sum of two dictionaries by keys
    """
    dic1 = OrderedDict(sorted(dic1.items()))
    dic2 = OrderedDict(sorted(dic2.items()))
    return {key: dic1.get(key, 0) + dic2.get(key, 0) for key in set(dic1) | set(dic2)}

def undirected_graph (graph, arcs='blocks'):
    '''
    This will convert a directed graph to undirected.
    '''
    backup_graph = copy.deepcopy(graph)
    for node1 in backup_graph.keys():
        for node2 in backup_graph[node1][arcs]:
            if node1 not in backup_graph[node2][arcs]:
                backup_graph[node2][arcs].append(node1)
    return backup_graph

def subgraph_finder (graph, node, cluster_num, visited, key='cluster', arcs='blocks'):
    '''
    This will find subgraphs in a graph.
    '''
    visited[node] = True
    graph[node][key] = cluster_num
    for node2 in graph[node][arcs]:
        if visited[node2] == False:
            subgraph_finder (graph, node2, cluster_num, visited)
    return graph, visited


def cluster_update(graph):
    '''
    Find the cluster id for all nodes in a graph 
    '''
    Undir_Graph = undirected_graph(graph)
    visited_ = {i: False for i in Undir_Graph.keys()}
    i = 0
    for bug in Undir_Graph.keys():
        if visited_[bug] == False:
            Undir_Graph, visited_ = subgraph_finder (Undir_Graph, bug, i, visited_)
            i = i+1
    for i in graph.keys():
        graph[i]['cluster'] = Undir_Graph[i]['cluster']
    return graph

def node_depth(dic_of_directed_Graph,index):
    '''
    Finding the depth of a node in a tree
    '''
    if (str(type(dic_of_directed_Graph)) == "<class 'networkx.classes.digraph.DiGraph'>") :
        dic_of_directed_Graph = nx.to_dict_of_lists(dic_of_directed_Graph)
    depth_i = []
    if len(dic_of_directed_Graph[index]["blocks"]) == 0:
        depth_ = 0
    else:
        depth_ = 1 + max(list(map(lambda X: node_depth(dic_of_directed_Graph,X),
                                  dic_of_directed_Graph[index]["blocks"])))
    return depth_ 

def depth_finder(dic_of_directed_Graph):
    '''
    Finding the depth of a tree
    '''
    if (str(type(dic_of_directed_Graph)) == "<class 'networkx.classes.digraph.DiGraph'>") :
        dic_of_directed_Graph = nx.to_dict_of_lists(dic_of_directed_Graph)
    depth_ = {}
    for i in sorted(dic_of_directed_Graph.keys()):
        depth_[i] = node_depth(dic_of_directed_Graph,i)
    return depth_

def max_dictionary(dic, seed_):
    """
    Finding the max value of a dictionary. In case of tie, it will choose randomly.
    """
    random.seed(seed_)
    max_dict = max(list(dic.values()))
    return random.choice([key for key, value in dic.items() if (value == max_dict)])

def update_bugs (list_idx, graph):
    '''
    Update depth and degree of the bugs in a given list
    '''
    for i in list_idx:
        graph[i]['depth'] = node_depth(graph, i)
        graph[i]['degree'] = len(graph[i]['blocks'])
    return graph

def create_graph(number_of_nodes, bug_dataset, expansion_factor = 1, output = "dictionary", 
                 min_change = -1, max_change = +1, seed=1):
    """
    expansion_factor is defined to say how much we need to increase the number of dependencies.
    bug_dataset is Whole_dataset
    """
    G = nx.DiGraph()
    G_dict = {}
    assert len(np.unique(bug_dataset.index)) == len(bug_dataset) # no duplicate bug should we have
    for i in range(number_of_nodes):
        G.add_node(i)
        G_dict[i] = {'blocks':[], 'depends_on':[], 'severity':None, 'votes': None,
                     'comment_count': None, 'time_to_solve': None, 'number_of_blocks': None}
        np.random.seed(seed+i)
        bug_index = np.random.choice(bug_dataset.index)
        random_change = np.random.choice(range(min_change,max_change+1))
        G_dict[i]['severity'] = bug_dataset.loc[bug_index,"severity_num"]
        G_dict[i]['votes'] = bug_dataset.loc[bug_index,"votes"]
        G_dict[i]['comment_count'] = bug_dataset.loc[bug_index,"comment_count"]
        actual_n_of_blocks = (bug_dataset.loc[bug_index,"n_blocks"])
        number_of_blocks = max((random_change + (actual_n_of_blocks * expansion_factor)),0) 
        diff = min ((number_of_blocks - actual_n_of_blocks), 5)
        G_dict[i]['time_to_solve'] = max((bug_dataset.loc[bug_index,"pred_time_to_solve"].days + diff), 1)
        G_dict[i]['number_of_blocks'] = number_of_blocks
    for i in range(number_of_nodes):
        number_of_blocks = G_dict[i]['number_of_blocks']
        try_ = 0
        n_added_edge = 0
        while ((try_ < 10) or (try_ < (number_of_blocks * 2))) and (n_added_edge != number_of_blocks):
            np.random.seed(seed+i+try_)
            random.seed(seed+i+try_)
            j = np.random.randint(number_of_nodes)
            G.add_edge(i,j)
            if nx.is_directed_acyclic_graph(G): # check whether it is acyclic
                if j not in G_dict[i]['blocks']:
                    G_dict[i]['blocks'].append(j)
                    assert i not in G_dict[j]['depends_on']
                    G_dict[j]['depends_on'].append(i)
                    n_added_edge += 1
            else:
                G.remove_edge(i,j)
            try_ += 1
    assert nx.is_directed_acyclic_graph(G)
    if output == "dictionary":
        return G_dict
    else:
        return G #nx.to_dict_of_lists(G)


def add_a_new_node(G_dict, vis_dict, hid_dict, bug_dataset, solved_bug, bug_index = None, partial_ratio = 0.5,
                   do_not_block = [], min_change = -1, max_change = +1,
                   expansion_factor = 1, output = "dictionary", seed=1):
    """
    This will add a node to the graph G_dict with the characteristics randomly coming from bug_dataset.
    do_not_block includes the list of bugs that this new bug should not block because they are already being solved.
    """
    G_dict_copy = G_dict.copy()
    vis_dict_copy = vis_dict.copy()
    hid_dict_copy = hid_dict.copy()
    G_dict_keys = list(G_dict.keys())
    solved_bug_keys = list([max(solved_bug)]) if len(solved_bug)>0 else list([0])
    m = {}
    for k in G_dict_keys:
        m[k] = G_dict_copy[k]['blocks']
    G = nx.DiGraph(m)
    last_key = max(G_dict_keys + solved_bug_keys)+1
    G.add_node(last_key)
    G_dict_copy[last_key] = {'blocks':[], 'depends_on':[], 'severity':None, 'votes': None, 
                             'comment_count': None, 'time_to_solve': None, 'number_of_blocks': None}
    np.random.seed(seed+last_key)
    if bug_index == None:
        bug_index = np.random.choice(bug_dataset.index)
    random_change = np.random.choice(range(min_change,max_change+1))
    G_dict_copy[last_key]['severity'] = bug_dataset.loc[bug_index,"severity_num"]
    G_dict_copy[last_key]['votes'] = bug_dataset.loc[bug_index,"votes"]
    G_dict_copy[last_key]['comment_count'] = bug_dataset.loc[bug_index,"comment_count"]
    actual_n_of_blocks = (bug_dataset.loc[bug_index,"n_blocks"])
    number_of_blocks = max((random_change + (actual_n_of_blocks * expansion_factor)),0) 
    diff = min ((number_of_blocks - actual_n_of_blocks), 5)
    G_dict_copy[last_key]['time_to_solve'] = max(
        (bug_dataset.loc[bug_index,"pred_time_to_solve"].days + diff), 1)
    G_dict_copy[last_key]['number_of_blocks'] = number_of_blocks
    vis_dict_copy[last_key] = copy.deepcopy(G_dict_copy[last_key])
    hid_dict_copy[last_key] = copy.deepcopy(G_dict_copy[last_key])
    try_ = 0
    n_added_edge = 0
    
    G_dict_keys_updated = list(G_dict_copy.keys())
    while ((try_ < 10) or (try_ < (number_of_blocks * 2))) and (n_added_edge != number_of_blocks):
        np.random.seed(seed+last_key+try_)
        random.seed(seed+last_key+try_)
        j = np.random.choice(G_dict_keys_updated)
        G.add_edge(last_key,j)
        # check whether it is acyclic and j is not blocking the bugs which are solving now
        if (nx.is_directed_acyclic_graph(G)) and (j not in do_not_block): 
            if j not in G_dict_copy[last_key]['blocks']: # if it is not already added, add it.
                G_dict_copy[last_key]['blocks'].append(j)
                G_dict_copy[j]['depends_on'].append(last_key)
                n_added_edge += 1
                random.seed(seed+last_key+try_+j)
                random_uniform = random.uniform(0, 1)
                if (random_uniform < partial_ratio):
                    hid_dict_copy[last_key]['blocks'].append(j)
                    hid_dict_copy[j]['depends_on'].append(last_key)
                    assert last_key not in vis_dict_copy[j]['depends_on']
                    assert j not in vis_dict_copy[last_key]['depends_on']
                else:
                    vis_dict_copy[last_key]['blocks'].append(j)
                    vis_dict_copy[j]['depends_on'].append(last_key)
                
        else:
            G.remove_edge(last_key,j)
        try_ += 1
    assert nx.is_directed_acyclic_graph(G)
    
    if output == "dictionary":
        return G_dict_copy, hid_dict_copy, vis_dict_copy
    else:    
        raise Exception("Sorry! Only dictionary output is available.")
        
def partially_obseravle_graph (dic_of_directed_Graph, prob, seed=1):
    #check whether the input is graph or list
    if (str(type(dic_of_directed_Graph)) == "<class 'networkx.classes.digraph.DiGraph'>") :
        dic_of_directed_Graph = nx.to_dict_of_lists(dic_of_directed_Graph)
    dict_of_lists ={}
    removed_links = {}
    for i in sorted(dic_of_directed_Graph.keys()):
        dict_of_lists [i] = dic_of_directed_Graph [i].copy()
        removed_links [i] = dic_of_directed_Graph [i].copy()
        dict_of_lists [i]['blocks'] = []
        removed_links [i]['blocks'] = []
        dict_of_lists [i]['depends_on'] = []
        removed_links [i]['depends_on'] = []
    for i in sorted(dic_of_directed_Graph.keys()):
        for j in dic_of_directed_Graph[i]['blocks']:
            random.seed(seed+j+i)
            random_uniform = random.uniform(0, 1)
            if (random_uniform < prob):
                removed_links[i]['blocks'].append(j)
                removed_links[j]['depends_on'].append(i)
            else:
                dict_of_lists[i]['blocks'].append(j)
                dict_of_lists[j]['depends_on'].append(i)
    return {"visible":dict_of_lists,"hidden":removed_links}

def find_parent(graph, bug_id, time_lost = timedelta(0), time_added = timedelta(1/4)):
    '''
    Find ancestors of a node. 
    '''
    first_parent = np.random.choice(graph[bug_id]['depends_on'])
    time_lost += time_added #losing a quarter of a day for finding parent
    if len(graph[first_parent]['depends_on']) == 0:
        return first_parent, time_lost
    else: #if parent has a grandparent
        #print(first_parent)
        return find_parent(graph, first_parent, time_lost, time_added)
    
def childern_metrics (bug_id, graph, severity_ = 0, degree_ = 0, severity_degree = 0, level_ = 0, 
                      method = 'exp', metric = 'severity'):
    if metric == 'severity':
        if method == 'exp':
            severity_ = ((1/np.exp(level_)) * graph[bug_id]['severity'])
        elif method == 'normal':
            severity_ = graph[bug_id]['severity']
        elif method == 'linear':
            severity_ = ((1/(1 + level_)) * graph[bug_id]['severity'])
        else:
            raise Exception('The method {} is not defined'.format(method))
        if len(graph[bug_id]['blocks']) > 0:
            level_ += 1
            for i in graph[bug_id]['blocks']:
                severity_ += childern_metrics(i, graph, severity_ = severity_, level_ = level_, 
                                               method = method, metric = metric)
        return severity_
    elif metric == 'degree':
        if method == 'exp':
            degree_ = ((1/np.exp(level_)) * len(graph[bug_id]['blocks']))
        elif method == 'normal':
            degree_ = len(graph[bug_id]['blocks'])
        elif method == 'linear':
            degree_ = ((1/(1 + level_)) * len(graph[bug_id]['blocks']))
        else:
            raise Exception('The method {} is not defined'.format(method))
        if len(graph[bug_id]['blocks']) > 0:
            level_ += 1
            for i in graph[bug_id]['blocks']:
                degree_ += childern_metrics(i, graph, degree_ = degree_, level_ = level_,
                                             method = method, metric = metric)
        return degree_
    
    elif metric == 'severity_degree':
        if method == 'exp':
            severity_degree = ((1/np.exp(level_)) * (graph[bug_id]['severity'] + len(graph[bug_id]['blocks']))) 
        elif method == 'normal':
            severity_degree = graph[bug_id]['severity'] + len(graph[bug_id]['blocks'])
        elif method == 'linear':
            severity_degree = ((1/(1 + level_)) * (graph[bug_id]['severity'] + len(graph[bug_id]['blocks']))) 
        else:
            raise Exception('The method {} is not defined'.format(method))
        if len(graph[bug_id]['blocks']) > 0:
            level_ += 1
            for i in graph[bug_id]['blocks']:
                severity_degree += childern_metrics(i, graph, severity_degree = severity_degree, level_ = level_,
                                                    method = method, metric = metric)
        return severity_degree
    else:
        raise Exception('The metric {} is not defined'.format(metric))

        
def select_bug_based_on_strategy(graph, strategy = 'random', method_ = 'exp', infeasible_list = [], seed = 0):
    '''
    Based on different strategies, which bugs should be chosen. 
    '''
    random.seed(seed)
    if strategy == "max_depth":
        start = max_dictionary({k: v['depth'] for k, v in graph.items() if k not in infeasible_list}, seed_ = seed)
    elif strategy == "max_degree":
        start = max_dictionary({k: v['degree'] for k, v in graph.items() if k not in infeasible_list}, seed_ = seed)
    elif strategy == "max_degree_plus_max_depth":
        depth_dict = ({k: v['depth'] for k, v in graph.items() if k not in infeasible_list})
        degree_dict = ({k: v['degree'] for k, v in graph.items() if k not in infeasible_list})
        sum_of_deg_and_dep = sum_dict(depth_dict, degree_dict)
        start = max_dictionary(sum_of_deg_and_dep, seed_ = seed)
    elif strategy == "max_severity":
        start = max_dictionary({k: v['severity'] for k, v in graph.items() if k not in infeasible_list}, seed_ = seed)
    elif strategy == "max_degree_plus_severity":
        degree_dict = ({k: v['degree'] for k, v in graph.items() if k not in infeasible_list})
        severity_dict = ({k: v['severity'] for k, v in graph.items() if k not in infeasible_list})
        sum_of_dep_and_sev = sum_dict(severity_dict, degree_dict)
        start = max_dictionary(sum_of_dep_and_sev, seed_ = seed)
    elif strategy == "childern_degree":
        degree_dict = {k: childern_metrics (k, graph, method = method_, metric = 'degree') for k, v in graph.items() 
                       if k not in infeasible_list}
        start = max_dictionary(degree_dict, seed_ = seed)
    elif strategy == "childern_severity":
        sev_dict = {k: childern_metrics (k, graph, method = method_, metric = 'severity') for k, v in graph.items() 
                    if k not in infeasible_list}
        start = max_dictionary(sev_dict, seed_ = seed)
    elif strategy == "random": 
        feasible_list = [i for i in list(sorted(graph.keys())) if i not in infeasible_list]
        start = random.choice(feasible_list)
    else: raise Exception("strategy not found")
    return start


def update_what_is_needed (graph_orig, list_of_bugs, solved_bugs):
    graph = copy.deepcopy(graph_orig)
    graph = cluster_update(graph)
    if ((len(graph)> 0) and (len(list_of_bugs)>0)):
        cluster_to_update = []
        try:
            list_of_bugs = list(set(list_of_bugs) - set(solved_bugs))
        except UnboundLocalError:
            pass
        for wh in list_of_bugs:
            cluster_to_update.append(graph[wh]['cluster'])

        what_to_update_ = [k for k, v in graph.items() if (v['cluster'] in cluster_to_update)]

        graph = update_bugs (what_to_update_, graph) #  Update depth and degree of the bugs
    return graph

def convert_graph_to_networkx (graphh):
    gg = {}
    for i in graphh:
        gg[i] = {}
        for j in graphh[i]['blocks']:
            gg[i][j] = {}
    return nx.DiGraph(gg)


def synthetic_simulation (n_developers, strategy_, 
						  graph_all, visible, hidden, bug_dataset, partial_ratio = 0.5, 
                          save_ = True, output_ = True,
                          diagnosing = False, reset_system = [False, None], method = 'exp', run_number = 0,
                          disable_tqdm = False, expansion_factor_ = 3):
    """
    graph all_, visible, and hidden, can be empty sets. It is better to define them before running.
    bug_dataset is the dataset of the real bugs mined from repository
    partial_ratio is the percentage of hidden links in the bug dependency graph
    reset_system can be used if you want to reset the bug dep graph every three months
    method is only used in children strategies as didscussed in the paper
    It is better to disable_tqdm if there is more than one run.
    expansion_factor_ will determine how dense the graph should be compared to the original one.
    """
    G = graph_all.copy()
    visible_backup = visible.copy()
    hidden_backup = hidden.copy()
    reported_date = [((date.timetuple().tm_year-2009)*(date.timetuple().tm_yday)) for date in bug_dataset.creation_time]
    bug_dataset['reported_date'] = reported_date
    number_of_ticks = max(reported_date)
    verbose_ = list(range(max(reported_date)))
    daily_metrics = pd.DataFrame(None, index=verbose_, columns = ['n_of_bugs', 'n_arcs','max_degree', 'mean_degree',
                                                                  'max_degree_centrality', 'mean_degree_centrality',
                                                                  'max_depth', 'mean_depth',
                                                                  'max_depth_centrality', 'mean_depth_centrality',
                                                                  'n_introduced', 'n_RESOLVED',
                                                                  'mean_subgraph_depth', 'mean_votes', 'mean_comments', 
                                                                  'mean_severity', 'mean_hub', 'max_hub',
                                                                  'mean_authority', 'max_authority',
                                                                  'mean_hub_norm', 'max_hub_norm',
                                                                  'mean_authority_norm', 'max_authority_norm',
                                                                  'mean_harmonic_centrality', 'max_harmonic_centrality',
                                                                  'day_counter'])
    daily_metrics_solved = pd.DataFrame(None, columns = ['date', 'bug_id', 'degree', 'depth',
                                                         'severity', 'votes', 'comment_count'])
    depth_recorder = pd.DataFrame(index=[verbose_], columns= ['day_counter'] + list(range(5)))
    task_table = {}
    all_resolved_bugs_strategy = {}
    solved_bug_list = []
    n_introduced = n_RESOLVED = 0
    
    for tick in tqdm(range(number_of_ticks), position=0, leave=True, disable = disable_tqdm):
        seed__ = int(tick)+ run_number
        what_to_update = []
        # if any developer is not busy, give them a task
        # if all the bugs are fixing, we won't have any other bug to fix!
        while (len(task_table) < n_developers) and (task_table.keys() != visible_backup.keys()):
            np.random.seed(seed__)
            selected_bug = select_bug_based_on_strategy(visible_backup, strategy = strategy_,  method_ = method,
                                                       infeasible_list = list(task_table.keys()), seed = seed__)
            if selected_bug not in task_table.keys():
                # add bug into to-do list with the time to fix (X many ticks later it should be solved)
                task_table[selected_bug] = tick + visible_backup[selected_bug]['time_to_solve']
        # is it time to solve a bug?
        if len(task_table) != 0:
            if (min (task_table.values()) <= tick):
                candidate_bug_list = [k for k, v in task_table.items() if v <= tick]
                for bug_to_solve in candidate_bug_list:
                    if len(visible_backup[bug_to_solve]['depends_on']) > 0:
                        '''
                        It has a blocker and should have not been selected
                        '''
                        time_lost_ = 0
                        time_added_ = 1/3
                        #remove the bug from list and return it to original list
                        del task_table[bug_to_solve]
                        what_to_update.append(bug_to_solve)
                        # replace it with one of its grand parents
                        bug_id, time_lost_ = find_parent (visible_backup, bug_to_solve, 
                                                          time_lost = time_lost_, time_added = time_added_)
                        if bug_id not in task_table.keys(): # if parent is not in task table
                            # add it with its fixing time
                            task_table[bug_id] = tick + visible_backup[bug_id]['time_to_solve'] + time_lost_
                    elif len(hidden_backup[bug_to_solve]['depends_on']) > 0: # if it has invisible blocker
                        bug_id = np.random.choice(hidden_backup[bug_to_solve]['depends_on'])
                        hidden_backup[bug_to_solve]['depends_on'].remove(bug_id)
                        hidden_backup[bug_id]['blocks'].remove(bug_to_solve)
                        visible_backup[bug_to_solve]['depends_on'].append(bug_id)
                        visible_backup[bug_id]['blocks'].append(bug_to_solve)
                        time_lost_ = 0
                        time_added_ = 1/3
                        #remove the bug from list and return it to original list
                        del task_table[bug_to_solve]
                        what_to_update.append(bug_to_solve)
                        if len(visible_backup[bug_id]['depends_on'])>0:
                            bug_id, time_lost_ = find_parent (visible_backup, bug_id,
                                                              time_lost = time_lost_, time_added = time_added_)
                        if bug_id not in task_table.keys(): # if parent is not in task table
                            # add it with its fixing time
                            task_table[bug_id] = tick + visible_backup[bug_id]['time_to_solve'] + time_lost_
                    else: # if it does not blocked by any visible or hidden bug
                        assert bug_to_solve not in all_resolved_bugs_strategy.keys()
                        daily_metrics_solved.loc[tick] = [tick,
                                                          bug_to_solve,
                                                          visible_backup[bug_to_solve]['degree'],
                                                          visible_backup[bug_to_solve]['depth'],
                                                          visible_backup[bug_to_solve]["severity"],
                                                          visible_backup[bug_to_solve]["votes"],
                                                          visible_backup[bug_to_solve]["comment_count"]
                                                         ]

                        what_to_update2 = [k for k, v in G.items() if 
                                           v['cluster'] == G[bug_to_solve]['cluster']]
                        what_to_update2.remove(bug_to_solve)
                        what_to_update = list(set(what_to_update2) | set(what_to_update))
                        G = update_bugs ([bug_to_solve], G)
                        try:
                            what_to_update.remove(bug_to_solve)
                        except ValueError:
                            pass
                        solved_bug_list.append(bug_to_solve)
                        # resolve it:
                        all_resolved_bugs_strategy[bug_to_solve] = G[bug_to_solve].copy()
                        del task_table[bug_to_solve]
                        for i in G[bug_to_solve]['blocks']:
                            if i not in what_to_update: what_to_update.append(i)
                            G[i]['depends_on'].remove(bug_to_solve)
                        del G[bug_to_solve]
                        for i in visible_backup[bug_to_solve]['blocks']:
                            if i not in what_to_update: what_to_update.append(i)
                            visible_backup[i]['depends_on'].remove(bug_to_solve)
                        del visible_backup[bug_to_solve] # if it gives error, do try except
                        for i in hidden_backup[bug_to_solve]['blocks']:
                            if i not in what_to_update: what_to_update.append(i)
                            hidden_backup[i]['depends_on'].remove(bug_to_solve)
                        del hidden_backup[bug_to_solve]
                        n_RESOLVED += 1

        # should I create a new bug?
        # do we have a new bug in this tick?
        for i in range(len(bug_dataset [tick == bug_dataset.reported_date])):
            bug_index_ = bug_dataset [tick == bug_dataset.reported_date].iloc[i].name
            G, visible_backup, hidden_backup = add_a_new_node(G, visible_backup, hidden_backup, Whole_dataset,
                                                              solved_bug_list, expansion_factor=expansion_factor_,
                                                              bug_index = bug_index_, partial_ratio = partial_ratio,
                                                              do_not_block = list(task_table.keys()))
            n_introduced +=1
            what_to_update.append(max(G.keys()))

        #####################    
        # Updating bug info #
        #####################
        G              = update_what_is_needed (G, what_to_update, solved_bug_list)
        visible_backup = update_what_is_needed (visible_backup, what_to_update, solved_bug_list)
        hidden_backup  = update_what_is_needed (hidden_backup, what_to_update, solved_bug_list)



        ####################    
        # Updating metrics #
        ####################
        if (G != {}) and (tick in verbose_):
            daily_metrics.loc[tick, "n_of_bugs"]    = len(G)
            daily_metrics.loc[tick, "n_arcs"]       = sum([len(G[i]['blocks']) for i in G])
            degree_list                             = ([v['degree'] for k, v in G.items()])
            depth_list                              = ([v['depth']  for k, v in G.items()])
            severity_list                           = ([v['severity']  for k, v in G.items()])
            votes_list                              = ([v['votes'] for k, v in G.items()])
            comment_count_list                      = ([v['comment_count'] for k, v in G.items()])
            Dir_Graph_strategy_networkx             = convert_graph_to_networkx(G)
            daily_metrics.loc[tick, "max_degree"]   = max(degree_list)
            daily_metrics.loc[tick, "mean_degree"]  = mean_list(degree_list)
            try:
                divide_by_n_bugs = 1/((len(G)-1))
            except ZeroDivisionError:
                divide_by_n_bugs = 1
            daily_metrics.loc[tick, "max_degree_centrality"]   = max(degree_list) * divide_by_n_bugs
            daily_metrics.loc[tick, "mean_degree_centrality"]  = mean_list(degree_list) * divide_by_n_bugs
            daily_metrics.loc[tick, "max_depth"]               = max(depth_list)
            daily_metrics.loc[tick, "mean_depth"]              = mean_list(depth_list)
            daily_metrics.loc[tick, "max_depth_centrality"]    = max(depth_list) * divide_by_n_bugs
            daily_metrics.loc[tick, "mean_depth_centrality"]   = mean_list(depth_list) * divide_by_n_bugs
            daily_metrics.loc[tick, "n_introduced"]            = n_introduced
            daily_metrics.loc[tick, "n_RESOLVED"]              = n_RESOLVED
            daily_metrics.loc[tick, "mean_votes"]              = mean_list(votes_list)
            daily_metrics.loc[tick, "mean_comments"]           = mean_list(comment_count_list)
            daily_metrics.loc[tick, "mean_severity"]           = mean_list(severity_list)
            try:
                hits__norm = nx.hits(Dir_Graph_strategy_networkx, max_iter = 200000, normalized=True, tol=1e-05)
                hits__     = nx.hits(Dir_Graph_strategy_networkx, max_iter = 200000, normalized=False, tol=1e-05)
                daily_metrics.loc[tick, "mean_hub"]           = mean_list(hits__[0].values())
                daily_metrics.loc[tick, "max_hub"]            = np.max(list(hits__[0].values()))
                daily_metrics.loc[tick, "mean_authority"]     = mean_list(hits__[1].values())
                daily_metrics.loc[tick, "max_authority"]      = np.max(list(hits__[1].values()))
                daily_metrics.loc[tick, "mean_hub_norm"]      = mean_list(hits__norm[0].values())
                daily_metrics.loc[tick, "max_hub_norm"]       = np.max(list(hits__norm[0].values()))
                daily_metrics.loc[tick, "mean_authority_norm"]= mean_list(hits__norm[1].values())
                daily_metrics.loc[tick, "max_authority_norm"] = np.max(list(hits__norm[1].values()))
            except ZeroDivisionError:
                daily_metrics.loc[tick, "mean_hub"]           = 0
                daily_metrics.loc[tick, "max_hub"]            = 0
                daily_metrics.loc[tick, "mean_authority"]     = 0 
                daily_metrics.loc[tick, "max_authority"]      = 0
                daily_metrics.loc[tick, "mean_hub_norm"]      = 0
                daily_metrics.loc[tick, "max_hub_norm"]       = 0
                daily_metrics.loc[tick, "mean_authority_norm"]= 0 
                daily_metrics.loc[tick, "max_authority_norm"] = 0

            harmonic_centrality_ = nx.harmonic_centrality(Dir_Graph_strategy_networkx).values()
            daily_metrics.loc[tick, "mean_harmonic_centrality"]= mean_list(harmonic_centrality_) * divide_by_n_bugs
            daily_metrics.loc[tick, "max_harmonic_centrality"] = np.max(list(harmonic_centrality_)) * divide_by_n_bugs
            daily_metrics.loc[tick, "day_counter"]             = tick           
            n_introduced = n_RESOLVED = 0
        #############################
        ####  Subgraphs' metrics ####
        #############################
            cluster_ids = np.array([G[i]['cluster'] for i in G])
            cluster_id, count = np.unique(cluster_ids, return_counts=True)
            for cl, cnt in zip(cluster_id, count):
                if cnt == 1:
                    for k , v in G.items():
                        if  v['cluster'] == cl: 
                            v['subgraph_depth'] = 0
                            break
                elif cnt == 2:
                    twice = 0.5
                    for k , v in G.items():
                        if  v['cluster'] == cl: 
                            v['subgraph_depth'] = 1
                            twice += 1
                        if twice >= cnt: break
                else: #cnt>2
                    max_depth = 0
                    more_than_once = 0
                    for k , v in G.items():
                        if  v['cluster'] == cl:
                            new_depth = node_depth(G, k)
                            more_than_once +=1
                            if new_depth > max_depth:
                                max_depth = new_depth
                            if more_than_once >= cnt: break
                    more_than_once = 0
                    for k , v in G.items():
                        if  v['cluster'] == cl:  
                            v['subgraph_depth'] = max_depth
                            more_than_once +=1
                            if more_than_once >= cnt: break
            sub_depth_ = np.array([G[i]['subgraph_depth'] for i in G])
            daily_metrics.loc[tick, "mean_subgraph_depth"]  = mean_list(sub_depth_)
            unique_, counts_ = np.unique(sub_depth_, return_counts=True)
            for dep, cnt in zip(unique_, counts_):
                depth_recorder.loc[tick, dep] = cnt
            depth_recorder.loc[tick, 'day_counter'] = tick
        #############################
        #############################

        if save_ and ((tick % 365 == 0) or (tick == (number_of_ticks-1))):
            try:
                os.mkdir(os.path.join(os.getcwd(),"..", "output", "Strategies_output_synthetic", strategy_))
                os.mkdir(os.path.join(os.getcwd(),"..", "output", "Strategies_output_synthetic", strategy_, 
                                      str(run_number)))
            except FileExistsError:
                try:
                    os.mkdir(os.path.join(os.getcwd(),"..", "output", "Strategies_output_synthetic", 
                                          strategy_, str(run_number)))
                except FileExistsError:
                    pass
            daily_metrics.to_csv(os.path.join(os.getcwd(),"..", "output", "Strategies_output_synthetic", 
                                              strategy_, str(run_number),
                                              "daily_metrics"+str(n_developers)+"_" + str(reset_system[1]) +".csv"),
                                index = False)
            daily_metrics_solved.to_csv(os.path.join(os.getcwd(),"..", "output", "Strategies_output_synthetic", 
                                                     strategy_, str(run_number),
                                            "daily_metrics_solved"+str(n_developers)+"_" + str(reset_system[1]) +".csv"), 
                                        index = False)
            depth_recorder.to_csv(os.path.join(os.getcwd(),"..", "output", "Strategies_output_synthetic",
                                               strategy_, str(run_number),
                                               "depth_recorder"+str(n_developers)+"_" + str(reset_system[1]) +".csv"),
                                 index=False)

            with open(os.path.join(os.getcwd(),"..", "output",  "Strategies_output_synthetic", strategy_, str(run_number),
                                   "all_resolved_bugs"+str(n_developers)+"_" + str(reset_system[1]) +".pickle"), 
                      'wb') as handle:
                pickle.dump(all_resolved_bugs_strategy, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(os.path.join(os.getcwd(),"..", "output", "Strategies_output_synthetic", strategy_, str(run_number),
                                   "Dir_Graph"+str(n_developers)+"_" + str(reset_system[1]) +".pickle"),
                      'wb') as handle2:
                pickle.dump(G, handle2, protocol=pickle.HIGHEST_PROTOCOL)

        if (reset_system[0]) and (tick % 90): # reset the system if necessary every 90 days
            #all_resolved_bugs_strategy= {} # list of resolved bug and their time to fix
            G = {}        # evolutionary graph
            hidden_backup = {}        # evolutionary graph
            visible_backups = {}        # evolutionary graph
            solved_bug_list = []
            n_introduced = n_RESOLVED = 0
            task_table = {}
            all_resolved_bugs_strategy = {}

    if output_:
        return [daily_metrics, daily_metrics_solved, depth_recorder, all_resolved_bugs_strategy, G]



number_of_nodes = 3
partial_ratio_ = 0.2
expansion_factor = 3
G = create_graph(number_of_nodes, Whole_dataset, min_change = -1, max_change = +1, expansion_factor = expansion_factor)
remove_by_perc = partially_obseravle_graph(G, partial_ratio_, seed=1) # make the graph partially observable
vis = remove_by_perc['visible']
hid = remove_by_perc['hidden']
# adding depth and degree to each
G = update_bugs (list(G.keys()), G)
vis = update_bugs (list(vis.keys()), vis)
hid = update_bugs (list(hid.keys()), hid)
# updating cluster numbers
G = cluster_update(G)
vis = cluster_update(vis)
hid = cluster_update(hid)
n_developers_ = 3
Whole_dataset = Whole_dataset[(Whole_dataset["creation_time"] >= '2010')]

import argparse
parser = argparse.ArgumentParser(description='My example explanation')
parser.add_argument(
    '--stra',
    type=str,
    #default='random',
    help='["childern_degree", "childern_severity", "max_severity", "max_degree", "max_depth", "max_degree_plus_max_depth", "max_degree_plus_severity", "random"]'
)

parser.add_argument(
    '--n_develop',
    default=3,
    type=int,
    help='number of developers'
)

parser.add_argument(
    '--run',
#    default=1,
    type=int,
    help='number of runs'
)

my_namespace = parser.parse_args()

for run_ in range(my_namespace.run):
    synthetic_simulation (my_namespace.n_develop, my_namespace.stra, G, vis, hid, Whole_dataset, 
                        partial_ratio = partial_ratio_, save_ = True, 
                        output_ = False, diagnosing = False, reset_system = [False,'whole'],
                        method = 'exp', run_number = run_, disable_tqdm = False,
                        expansion_factor_ = expansion_factor)
							   
## python COMMAND_LINE_TEST.py --stra=random --run=3 
