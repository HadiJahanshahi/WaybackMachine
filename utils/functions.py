from utils.prerequisites import * # Packages and these functions: isNaN, isnotNaN, 
                                # convert_string_to_array, mean_list, and string_to_time

class Functions:
    seed_        = 0
    tag_map      = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    @staticmethod
    def max_dictionary(dic):
        """
        Finding the max value of a dictionary. In case of tie, it will choose randomly.
        """
        max_dict         = max(list(dic.values()))
        Functions.seed_ += 1 
        random.seed(Functions.seed_)
        return random.choice([key for key, value in dic.items() if (value == max_dict)])
    @staticmethod
    def sum_dict (dic1 , dic2):
        """
        sum of two dictionaries by keys
        """
        dic1 = OrderedDict(sorted(dic1.items()))
        dic2 = OrderedDict(sorted(dic2.items()))
        return {key: dic1.get(key, 0) + dic2.get(key, 0) for key in set(dic1) | set(dic2)}
    @staticmethod
    def mean_list (x):
        '''
        Returns the mean of a list
        '''
        return np.mean(list(x))
    @staticmethod
    def undirected_network(network):
        '''
        This will convert a directed graph to undirected.
        '''
        undirected_net = copy.deepcopy(network)
        for node1 in undirected_net.network.keys():
            for node2 in undirected_net[node1]:
                assert node1 not in undirected_net[node2]
                undirected_net[node2].append(node1)
        return undirected_net

    @staticmethod
    def start_of_the_day(date):
        return np.datetime64(date)

    @staticmethod
    def end_of_the_day(date):
        return np.datetime64(date) + np.timedelta64(1, 'D')
    
    @staticmethod
    def read_files(project_name):
        try:
            list_of_developers = pickle.load(open(os.path.join('dat', project_name, 'list_of_developers.txt'), "rb"))
            time_to_fix_LDA    = pd.read_csv(os.path.join('dat', project_name, 'time_to_fix_LDA.csv'))
            time_to_fix_LDA.columns.values[0] = 'developer'
        except:
            list_of_developers = None
            time_to_fix_LDA    = None
        try:
            with open(os.path.join('dat', project_name, "feasible_bugs_actual.txt"), "rb") as fp:   #Pickling
                feasible_bugs_actual = pickle.load(fp)
        except:
            feasible_bugs_actual = None
            
        bug_evolution_db   = pd.read_csv(os.path.join('dat', project_name, 'bug_evolution_data_new.csv'))
        Whole_dataset      = pd.read_csv(os.path.join('dat', project_name, 'whole_dataset_new.csv'))
        # formating update
        try:
            bug_evolution_db.time       = bug_evolution_db.time.map(lambda x: string_to_time(x))
        except ValueError:
            bug_evolution_db.time       = bug_evolution_db.time.map(lambda x: string_to_time(x, '%m/%d/%Y %H:%M'))
        try:
            Whole_dataset.creation_time = Whole_dataset.creation_time.map(lambda x: string_to_time(x))
        except ValueError:
            Whole_dataset.creation_time = Whole_dataset.creation_time.map(lambda x: string_to_time(x,
                                                                                                   '%Y-%m-%d %H:%M:%S'))
        # setting index
        Whole_dataset = Whole_dataset.set_index(['id'])
        try:
            SVM_model = pickle.load(open(os.path.join('..','dat', project_name, 'SVM.sav'), 'rb'))
        except:
            SVM_model = None
        try:
            Whole_dataset['lemma'] = Whole_dataset['lemma'].map(eval)
        except:
            pass
        bug_evolution_db              = bug_evolution_db[bug_evolution_db.time<'2020'] # until the end of 2019
        """ Sorting evolutionary DB by time and status """
        custom_dict                   = {'introduced':0, 'NEW':1, 'ASSIGNED':2,  'assigned_to':3, 'RESOLVED':4, 'REOPENED':5,
                                         'UPDATE':6, 'blocks':7, 'depends_on':8, 'VERIFIED':9, 'CLOSED':10}
        bug_evolution_db['rank']      = bug_evolution_db['status'].map(custom_dict)
        bug_evolution_db.sort_values(['time', 'rank'], inplace= True)
        bug_evolution_db.drop('rank', axis=1, inplace=True)
        bug_evolution_db.reset_index(drop = True, inplace = True)
        #converting severity to numbers
        Whole_dataset['severity_num'] = Whole_dataset['severity'].replace(['normal', 'critical', 'major', 'minor',
                                                                           'trivial', 'blocker', 'enhancement', 'S1', 'S2', 'S3', 'S4', '--', None],
                                                                          [3,5,4,2,1,6,0,5,4,3,1,3,3])
        #converting priority to numbers >> P1 is the most important and P5 is the least important one
        if ('P1' in Whole_dataset.priority.unique()) and (len(Whole_dataset.priority.unique()) in [5,6]):
            Whole_dataset['priority_num'] = Whole_dataset['priority'].replace(['--', 'P5', 'P4', 'P3','P2','P1'],
                                                                              [0,1,2,3,4,5])
        elif ('highest' in Whole_dataset.priority.unique()) and (len(Whole_dataset.priority.unique())==5):
            Whole_dataset['priority_num'] = Whole_dataset['priority'].replace(['lowest', 'low', 'medium', 'high', 'highest'],
                                                                              [1,2,3,4,5])
        else:
            raise Exception (f'undefined priority levels {Whole_dataset.priority_num.unique()}')
        Whole_dataset['assigned_to_detail.email'] = Whole_dataset['assigned_to_detail.email'].map(lambda x: x.lower())

        glove_embeddings_index = dict()
        with open(os.path.join('dat', 'Embeddings', 'glove.6B.100d.txt'), encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                glove_embeddings_index[word] = coefs

        return [bug_evolution_db, Whole_dataset, list_of_developers, 
                time_to_fix_LDA, SVM_model, feasible_bugs_actual, glove_embeddings_index]
    
    @staticmethod
    def max_tuple_index(tuple_):
        return max(tuple_,key=itemgetter(1))[0]
    
    @staticmethod
    def model_kp(p, c, T, LogToConsole=False, verbose=False):
        p     = np.array(p)
        c     = np.round(np.array(c),6)
        model = Model()
        model.params.LogToConsole = LogToConsole
        model.params.TimeLimit = 60
        m, n  = np.array(p).shape
        "i=1 to m developers and j=1 to n bugs"
        assert p.shape == c.shape
        # x_{ij} \in \{0,1\}
        x    = model.addVars(m, n, vtype= GRB.BINARY, name='x')
        # sum_{i=1}^{m}{sum_{j=1}^{n}{P_{ij} X_{ij}}}
        model.setObjective(quicksum(p[i,j]* x[i,j] for i in range(m) for j in range(n)), GRB.MAXIMIZE)
        # sum_{j=1}^{n}{C_{ij} X_{ij}} \le T_i
        for i in range(m):
            model.addConstr(quicksum(c[i,j]* x[i,j] for j in range(n)) <= T[i])
        # sum_{i=1}^{m}{X_{ij}} \le 1
        for j in range(n):
            model.addConstr(quicksum(x[i,j] for i in range(m)) <= 1)
        if verbose:
            model.write('model_kp.lp')
        model.optimize()
        return model, x
    
    @staticmethod
    def lemmatizing(text, ls = True):
        tokens  = word_tokenize(text)
        if ls:
            lm_text = [WordNetLemmatizer().lemmatize(token, Functions.tag_map[tag[0]]) for token, tag in pos_tag(tokens) if (
                (token not in STOPWORDS) and (len(token) < 20) and (token.isalpha()))]
        else:
            lm_text = ''
            for token, tag in pos_tag(tokens):
                if ((token not in STOPWORDS) and (len(token) < 20) and (token.isalpha())):
                    lm_text += WordNetLemmatizer().lemmatize(token, Functions.tag_map[tag[0]]) + ' '
        return lm_text
    
    @staticmethod
    def convert_nan_to_space(text):
        if text != text:
            return ""
        else:
            return text.lower()

    @staticmethod
    def return_index(condition, list_):
        return [idx for idx, val in enumerate(list_) if (val == condition)]