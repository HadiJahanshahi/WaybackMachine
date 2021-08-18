# no import is needed

class Assignee:
    n_of_dev = 0
    def __init__(self, email, id_, name, LDA_experience, time_limit_orig):
        self.email               = email.lower()
        self.n                   = Assignee.n_of_dev
        self.id_                 = id_
        if name != name:
            name = ''
        self.name                = name.lower()
        self.bugs                = []
        self.working_time        = []
        self.components          = []
        self.components_tracking = []
        self.accuracy            = []        
        self.accuracyT           = []        
        self.time_limit_orig     = time_limit_orig # it will never change
        self.time_limit          = time_limit_orig # it will change
        self.LDA_experience      = LDA_experience
        self.n_assigned_bugs     = 0
        Assignee.add_dev()
        
    def assign_bug(self, bug, time_, mode_):
        """[Assign a bug to a developer]

        Args:
            bug (Bug): [Which bug to assign]
            time_ (int): [Assignment date]
            mode_ (str): [Whether it is tracking or not]
        """
        assert bug not in self.bugs
        self.bugs.append(bug)
        self.components.append(bug.component) 
        self.n_assigned_bugs += 1
        bug.assigned_to_rec   = self.email
        bug.assigned_time.append(time_)

    def assign_and_solve_bug(self, bug, time_, mode_, resolution_, T_or_P = 'triage'):
        """[Assign a bug to a developer]

        Args:
            bug (Bug): [Which bug to assign]
            time_ (int): [Assignment date]
            mode_ (str): [Whether it is tracking or not]
            resolution_ (str): [Whether it is Actual assignment or other bug triaging policies.]
            T_or_P (str): [Whether it is a Triage or Prioritization task]
        """
        if bug.component in self.components:
            # only having the same COMPONENT is enough to say the assignment is accurate.
            bug.assignment_accuracy = 1
            self.accuracy.append(1)
        else:
            bug.assignment_accuracy = 0    
            self.accuracy.append(0)
        if self.email.lower() == bug.assigned_to.lower():
            # It has to be assigned to the same DEVELOPER to be accurate.
            bug.assignment_accuracyT = 1
            self.accuracyT.append(1)
        else:
            bug.assignment_accuracyT = 0
            self.accuracyT.append(0)
        if bug not in self.bugs:
            self.n_assigned_bugs += 1
            self.bugs.append(bug)
            # time to solve based on LDA table
        if (resolution_ == 'Actual') or (T_or_P == 'prioritization'):
            solving_time_ = bug.time_to_solve
        else:
            solving_time_ = self.LDA_experience [bug.LDA_category]
        bug.solving_time_after_simulation_accumulated         = solving_time_ + (self.time_limit_orig - self.time_limit)
        bug.solving_time_after_simulation                     = solving_time_
        assert bug.solving_time_after_simulation             != None           
        assert bug.solving_time_after_simulation_accumulated != None           
        self.working_time.append(solving_time_)
        if mode_ == 'tracking':
            self.time_limit      -= solving_time_
            self.components_tracking.append(bug.component) 
        else:
            """we update components only in training phase"""
            self.components.append(bug.component) 
        bug.assigned_to_rec   = self.email
        bug.assigned_time.append(time_)
                
    def search_by_email(self, email_):
        if self.email == email_.lower():
            return True
        return False
    def search_by_id(self, id_):
        if self.id_ == id_:
            return True
        return False
    def increase_time_limit(self):
        """ at the end of the day, we need to add to the time_limit of each developer by 1 """
        # it cannot exceed time_limit_orig
        self.time_limit = min (self.time_limit+1, self.time_limit_orig)
    @classmethod
    def add_dev(cls): #class method not for an object
        cls.n_of_dev += 1
