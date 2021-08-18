from bin.BDG import BDG
from components.assignee import Assignee
from components.bug import Bug
from utils.functions import Functions
from utils.prerequisites import *  # Packages and these functions: isNaN, isnotNaN,
                                   # convert_string_to_array, mean_list, and string_to_time
from utils.report import Report

from simulator.wayback import Discrete_event_simulation


#######################################
##                                   ##
##      author: Hadi Jahanshahi      ##
##     hadi.jahanshahi@ryerson.ca    ##
##          Data Science Lab         ##
##                                   ##
#######################################


parser = argparse.ArgumentParser(description='The parser will handle hyperparameters of the model')

parser.add_argument(
    '--project', 
    default = 'Mozilla',
    type    = str,
    help    = 'it can be selected from this list: [LibreOffice, Mozilla, EclipseJDT]'
)

parser.add_argument(
    '--resolution',
    type    = str,
    help    = 'it can be selected from this list: [Actual, DABT, RABT, CosTriage, CBR]'
)

parser.add_argument(
    '--n_days',
    type    = int,
#	default = 6781,
    help    = 'How many days we need to train?'
)

parser.add_argument(
    '--prioritization_triage', 
    type    = str,
    help    = 'it can be selected from this list: [prioritization, triage, both]'
)
parser.add_argument(
    '--verbose',
    default = 0,
    type    = int,
    help    = 'it can be either: [0, 1, 2, nothing, some, all]'
)
wayback_param        = parser.parse_args()

project              = wayback_param.project
verbose              = wayback_param.verbose
file_name            = wayback_param.resolution
Tfidf_vect           = None
[bug_evolutionary_db, bug_info_db, list_of_developers, 
 time_to_fix_LDA, SVM_model, feasible_bugs_actual, embeddings] = Functions.read_files(project)

if wayback_param.prioritization_triage.lower() in ['prioritization', 'both']:
    simul_prioritization = Discrete_event_simulation(bug_evolutionary_db, bug_info_db, list_of_developers,
                                                    time_to_fix_LDA, SVM_model, Tfidf_vect, project, feasible_bugs_actual,
                                                    embeddings, resolution = file_name, verbose=verbose)
    stop_date            = len(pd.date_range(start=simul_prioritization.bug_evolutionary_db.time.min().date(), end='31/12/2019'))
if wayback_param.prioritization_triage.lower() in ['triage', 'both']:
    simul_triage         = Discrete_event_simulation(bug_evolutionary_db, bug_info_db, list_of_developers,
                                                    time_to_fix_LDA, SVM_model, Tfidf_vect, project, feasible_bugs_actual,
                                                    embeddings, resolution = file_name, verbose=verbose)
    stop_date            = len(pd.date_range(start=simul_triage.bug_evolutionary_db.time.min().date(), end='31/12/2019'))

try:
    for i in tqdm(range(stop_date-1), desc="simulating days", position=0, leave=True):
        if wayback_param.prioritization_triage.lower() in ['prioritization', 'both']:
            simul_prioritization.prioritization()
        if wayback_param.prioritization_triage.lower() in ['triage', 'both']:
            simul_triage.triage                ()
        #if (simul_Actual_time_half.date > simul_Actual_time_half.testing_time):
    
    if wayback_param.prioritization_triage.lower() in ['prioritization', 'both']:
        with open(f'dat/{project}/{file_name}_{i}_prioritization.pickle', 'wb') as file:
            pickle.dump(simul_prioritization, file) # use `json.loads` to do the reverse
    if wayback_param.prioritization_triage.lower() in ['triage', 'both']:
        with open(f'dat/{project}/{file_name}_{i}_triage.pickle', 'wb') as file:
            pickle.dump(simul_triage, file) # use `json.loads` to do the reverse
except Exception as e:
    # winsound.PlaySound("*", winsound.SND_ALIAS)
    raise e


# python3.7 simulator/main.py --project=Mozilla --resolution=Actual --n_days=7511 --prioritization_triage=prioritization
# python3.7 simulator/main.py --project=EclipseJDT --resolution=Actual --n_days=6644 --prioritization_triage=prioritization
# python3.7 simulator/main.py --project=LibreOffice --resolution=Actual --n_days=3438 --prioritization_triage=prioritization
# To find site-packages location: python -m site
# creating example.pth there 
# PATH="$PATH:/home/hadi/Hadi_progress-1/Scripts/Bugzilla_Mining/OOP/"