

# %%


from pathlib import Path
import pickle
import json


from pathlib import Path
import csv
import sys
import pickle
import xml.etree.ElementTree as ET
import numpy as np
import json
import pandas as pd
from copy import deepcopy

### update


def filter_assignment(assignment_parsed):
    return assignment_parsed


def parse_to_2d(assignment_parsed):
    data2d = dict()
    for key in ['emotion_order', 'randCondNum', 'dem_gender', 'dem_language', 'val_recognized', 'val_familiar', 'val_feedback', 'total_time', 'visible_area', 'browser', 'browser_version', 'ip', 'ipify_calls', 'request_succeeded', 'request_time', 'reply_succeeded', 'reply_time', 'workerId', 'workerIdAnon', 'assignmentId']:
        data2d[key] = assignment_parsed.get(key, 'empty')

    for key, val in assignment_parsed['validationRadio'].items():
        data2d[key] = val

    data2d['training_video_time_to_load'] = assignment_parsed['training_video_stats']['time_to_load']

    trial_data = list()
    for trial_raw in assignment_parsed['data_parsed']:

        trial = dict()
        for sub_field in ['workerId', 'workerIdAnon']:
            trial[sub_field] = data2d[sub_field]

        trial_ = deepcopy(trial_raw)

        trial.update(json.loads(trial_.pop('stimParam')))
        trial.update(trial_.pop('video_stats'))
        trial.update(trial_)
        trial_data.append(trial)

    return data2d, pd.DataFrame(trial_data)


def parse_assignment(assignment):
    sys.path.append(str(Path("/Users/dae/Documents/mturk_keys/")))
    from anonymization_function import get_anonymized_id

    # output = '\n\nSubject: ' + str(iSubject).zfill(3) + '\n'
    # botoOutput.write(output + '\n')
    # print(output)
    ###
    workerId = assignment['WorkerId']
    workerIdAnon = get_anonymized_id(workerId)
    assignmentId = assignment['AssignmentId']  # ID for this particular response
    answer = assignment['Answer'].encode('utf-8')
    ###
    output = 'The Worker with ID {} submitted assignment {} and gave the answer {}'.format(workerId, assignmentId, answer)
    # botoOutput.write(output + '\n')
    # print(output)
    ###
    # tree = ET.parse(answer) # if reading in from file
    root = ET.fromstring(answer)  # if reading in from string

    rooted = dict()
    for child in root:
        Fquestion = child[0].text
        Fanswer = child[1].text

        rooted[Fquestion] = json.loads(Fanswer)

    ###### generic ^^

    ###### Parser vv

    from copy import deepcopy
    parsed = dict()
    rooted.keys()
    data_parsed = list()
    for i_trial, trial in enumerate(rooted['data']):
        parsed = deepcopy(trial)

        data_parsed.append(parsed)

    rooted['data_parsed'] = data_parsed
    rooted['workerId'] = workerId
    rooted['workerIdAnon'] = workerIdAnon
    rooted['assignmentId'] = assignmentId

    return rooted


def load_hit(pkl_path=None, experiment_number=0, dataset_number=0, git_hash=''):

    with open(pkl_path, 'rb') as f:
        allresp = pickle.load(f)

    parsed_data, summaryrows, trialrows = list(), list(), list()
    for iSubject, assignment in enumerate(allresp):
        output = '\n\nSubject: ' + str(iSubject).zfill(3) + '\n'
        assignment_parsed = parse_assignment(assignment)
        assignment_filtered = filter_assignment(assignment_parsed)
        parsed_data.append(assignment_filtered)
        subjsummaryrow, subjtrialsrows = parse_to_2d(assignment_parsed)

        summaryrows.append(subjsummaryrow)
        trialrows.append(subjtrialsrows)

    ##### FILTER

    summary_df = pd.DataFrame(summaryrows)
    summary_df['experiment_number'] = experiment_number
    summary_df['dataset_number'] = dataset_number
    summary_df['git_hash'] = git_hash

    trials_df_list = list()
    for subjtrialsrows in trialrows:
        trials_df = pd.DataFrame(subjtrialsrows)
        trials_df['experiment_number'] = experiment_number
        trials_df['dataset_number'] = dataset_number
        trials_df['git_hash'] = git_hash
        trials_df_list.append(trials_df)

    # print(trials_df.columns)

    return parsed_data, summary_df, pd.concat(trials_df_list)


expbasedir = Path("/Users/dae/coding/-GitRepos/ITE_GoldenBalls/ite_gb/experiment12/boto3_data_dumps")
csv_path = Path("/Users/dae/coding/-GitRepos/ITE_GoldenBalls/ite_gb/php/serveCondition/servedConditions_exp12.csv")

subject_tracker_path = Path('/Users/dae/coding/-GitRepos/ITE_GoldenBalls/ite_gb/aws_sdk/all_subject_tracker.xlsx')
subj_out_path = Path("/Users/dae/coding/-GitRepos/ITE_GoldenBalls/ite_gb/experiment12/subject_tracker_gitignore.json")
all_subject_tracker_path = Path("/Users/dae/coding/-GitRepos/ITE_GoldenBalls/ite_gb/aws_sdk/subject_tracking")
all_subject_tracker_path.mkdir(exist_ok=True)
this_subject_tracker_path = all_subject_tracker_path / 'experiment12.csv'
batch_1_path = expbasedir / 'batch_1_composit.xlsx'
batch_2_path = expbasedir / 'batch_2_composit.xlsx'

hits = [
    dict(
        pkl_path=expbasedir / "12-1_3D17ECOUOFV3V62A7G3SCRRHI5913A/allresponses_3D17ECOUOFV3V62A7G3SCRRHI5913A.pkl",
        experiment_number=12,
        dataset_number=1,
        git_hash='7613193098ec9886c1e848135cb0ff5c12712381',
    ),
    dict(
        pkl_path=expbasedir / "12-2_3SCKNODZ0YGOAWAMEF4GMFG0Q9TN74/allresponses_3SCKNODZ0YGOAWAMEF4GMFG0Q9TN74.pkl",
        experiment_number=12,
        dataset_number=2,
        git_hash='b90453e395291020e0bc680be9d95467c01478c5',
    ),
    dict(
        pkl_path=expbasedir / "12-3_3B286OTISFHAA8HZ6KO83Z19IZ9AJX/allresponses_3B286OTISFHAA8HZ6KO83Z19IZ9AJX.pkl",
        experiment_number=12,
        dataset_number=3,
        git_hash='',
    ),
    dict(
        pkl_path=expbasedir / "12-4_3D5G8J4N5B4OGZG2T9T81S2V1DWTVL/allresponses_3D5G8J4N5B4OGZG2T9T81S2V1DWTVL.pkl",
        experiment_number=12,
        dataset_number=4,
        git_hash='',
    ),
    dict(
        pkl_path=expbasedir / "12-5_3JTPR5MTZTC8FMB0T9X3W3L05CF5KW/allresponses_3JTPR5MTZTC8FMB0T9X3W3L05CF5KW.pkl",
        experiment_number=12,
        dataset_number=5,
        git_hash='',
    ),
    dict(
        pkl_path=expbasedir / "12-6_3T2EL38U0NK3S8T5CK0VK8OTXITQXX/allresponses_3T2EL38U0NK3S8T5CK0VK8OTXITQXX.pkl",
        experiment_number=12,
        dataset_number=6,
        git_hash='',
    ),
    dict(
        pkl_path=expbasedir / "12-7_322ZSN9Z5HKPMMXN9DSLS22IOVV4TK/allresponses_322ZSN9Z5HKPMMXN9DSLS22IOVV4TK.pkl",
        experiment_number=12,
        dataset_number=7,
        git_hash='',
    ),
    dict(
        pkl_path=expbasedir / "12-8_3IYI9285WT0Y3NEXQYKZ9URXI66CJL/allresponses_3IYI9285WT0Y3NEXQYKZ9URXI66CJL.pkl",
        experiment_number=12,
        dataset_number=8,
        git_hash='',
    ),
    dict(
        pkl_path=expbasedir / "12-9_3L1EFR8WWU5G0NLYVOJNS6YXJFU9F8/allresponses_3L1EFR8WWU5G0NLYVOJNS6YXJFU9F8.pkl",
        experiment_number=12,
        dataset_number=9,
        git_hash='',
    ),
    dict(
        pkl_path=expbasedir / "12-10_3EKZL9T8Y9M2CHJ72YHEXHBI5FZCHF/allresponses_3EKZL9T8Y9M2CHJ72YHEXHBI5FZCHF.pkl",
        experiment_number=12,
        dataset_number=10,
        git_hash='',
    ),
]

parsed_data, summary_dfs, trials_dfs = list(), list(), list()
for hit in hits:
    parsed_hit_, summary_df_, trials_df_ = load_hit(**hit)
    parsed_data.extend(parsed_hit_)
    summary_dfs.append(summary_df_)
    trials_dfs.append(trials_df_)

summary_df = pd.concat(summary_dfs).reset_index(drop=True)
trials_df = pd.concat(trials_dfs).reset_index(drop=True)
trials_df.columns

client_dims_ = list()
for dims in trials_df['visible_area']:
    client_dims_.append(dims)
client_dims = np.vstack(client_dims_)


def plot_visible_area(dims):
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()

    fig, axs = plt.subplots([1, 2])

    iax = 0


v0 = -1 * np.ones((summary_df.shape[0], 3), dtype=int)
v0[:, 0] = summary_df['val_trainingvideo'] == '7510'
v0[:, 1] = summary_df['val_emomatch_contemptuous'] == 'disdainful'
v0[:, 2] = summary_df['val_expressionmatch_joyful'] == 'AF25HAS'

v1 = -1 * np.ones((summary_df.shape[0], 4), dtype=int)
v1[:, 0] = summary_df['payoff_comprehension_CD12_p1'] == '0'
v1[:, 1] = summary_df['payoff_comprehension_CD12_p2'] == '12000'
v1[:, 2] = summary_df['payoff_comprehension_DD42_p1'] == '0'
v1[:, 3] = summary_df['payoff_comprehension_CC42_p1'] == '21000'

summary_df.loc[:, 'valid'] = np.all(v0, axis=1).astype(int)
summary_df.loc[:, 'valid_1'] = np.all(v1, axis=1).astype(int)
summary_df.loc[:, 'valid_t'] = np.all(np.hstack([v0, v1]), axis=1).astype(int)


cols_included = [
    'workerId',
    'workerIdAnon',
    'randCondNum',
    'valid',
    'valid_1',
    'valid_t',
    'val_recognized',
    'val_familiar',
    'val_feedback',
    'dem_gender',
    'dem_language',
    'iwould_large',
    'iexpectother_large',
    'predicted_C_proport',
    'total_time',
    'visible_area',
    'browser',
    'browser_version',
    'emotion_order',
    ###
    'experiment_number',
    'dataset_number',
    ###
    'val_trainingvideo',
    'val_emomatch_contemptuous',
    'val_expressionmatch_joyful',
    'payoff_comprehension_CD12_p1',
    'payoff_comprehension_CD12_p2',
    'payoff_comprehension_DD42_p1',
    'payoff_comprehension_CC42_p1',
    ### mTurk vars
    'assignmentId',
    ### website performance
    'ip',
    'ipify_calls',
    'request_succeeded',
    'request_time',
    'reply_succeeded',
    'reply_time',
    'training_video_time_to_load',
    'git_hash'
]
summary_df_colfilter = summary_df.loc[:, cols_included]

cols_included = [
    'workerId',
    'workerIdAnon',
    'stimID',
    'stimulus',
    'pot',
    'stimFileName',
    'time_to_load',
    'time_to_serve',
    'active',
    'served',
    'disposed',
    'trial_number',
    'visible_video',
    'visible_video_base',
    'visible_area',
    'respTimer',
    ###
    'Annoyed',
    'Apprehensive',
    'Contemptuous',
    'Content',
    'Devastated',
    'Disappointed',
    'Disgusted',
    'Embarrassed',
    'Excited',
    'Furious',
    'Grateful',
    'Guilty',
    'Hopeful',
    'Impressed',
    'Jealous',
    'Joyful',
    'Proud',
    'Relieved',
    'Surprised',
    'Terrified',
    ###
    'experiment_number',
    'dataset_number',
    'git_hash'
]
trials_df_colfilter = trials_df.loc[:, cols_included]

trials_df_colfilter.to_excel(batch_1_path)
summary_df_colfilter.to_excel(batch_2_path)


conds = summary_df.loc[:, 'randCondNum'].apply(lambda x: int(x.split('set')[1])).to_numpy()
valid_conds = conds[summary_df.loc[:, 'valid'] == 1]

n_cond = max(132, np.max(conds))
n_counts = np.zeros((n_cond, 4), dtype=int)
for i_cond in range(n_cond):
    n_total_ = np.sum(conds == i_cond)
    n_valid_ = np.sum(valid_conds == i_cond)

    n_counts[i_cond, :] = np.array([i_cond, 0, n_total_, n_valid_])


# save numpy array as csv file
# define data
csv_data = np.asarray(n_counts)
# save to csv file
np.savetxt(csv_path, n_counts, fmt='%d', delimiter=',')

print(f"Missing conditions: {np.sum(csv_data[:,-1]==0)}")

these_subjects = summary_df.loc[:, 'workerId'].to_list()


datasheet_temp = pd.read_excel(subject_tracker_path, header=0, index_col=None)
worker_ids = datasheet_temp.loc[:, 'WorkerID'].unique()
prior_subjects = worker_ids.tolist()


all_subjects = [*prior_subjects, *these_subjects]
all_subjects.reverse()


with open(subj_out_path, 'w') as f:
    json.dump(all_subjects, f)


sdf = summary_df.loc[:, ('workerId', 'experiment_number', 'dataset_number')].copy()
sdf['submitted'] = 1
sdf.reset_index(drop=True, inplace=True)
wid_additional = ['A18T7E73TNGOKP', 'AZLRPIAD0ZHAA']
sdf_additional_list = list()
for wid in wid_additional:
    sdf_additional_list.append(dict(workerId=wid, experiment_number=12, dataset_number=-1, submitted=0))

this_tracker = pd.concat([sdf, pd.DataFrame(sdf_additional_list)])

this_tracker.to_csv(this_subject_tracker_path, index=False)

# %%

emotions = [
    'Annoyed',
    'Apprehensive',
    'Contemptuous',
    'Content',
    'Devastated',
    'Disappointed',
    'Disgusted',
    'Embarrassed',
    'Excited',
    'Furious',
    'Grateful',
    'Guilty',
    'Hopeful',
    'Impressed',
    'Jealous',
    'Joyful',
    'Proud',
    'Relieved',
    'Surprised',
    'Terrified',
]
valid_pid = summary_df_colfilter.loc[summary_df_colfilter['valid'] == 1, 'workerIdAnon']
assert valid_pid.unique().shape[0] == valid_pid.shape[0]

filtered_summ_df_list = list()
filtered_trial_df_list = list()
for i_pid, pid in enumerate(valid_pid):
    sub_summary_df = summary_df_colfilter.loc[summary_df_colfilter['workerIdAnon'] == pid, :].copy()
    sub_trials_df = trials_df_colfilter.loc[trials_df_colfilter['workerIdAnon'] == pid, :].copy()
    pid_ = f'mTurk{i_pid:04d}'
    sub_summary_df.loc[:, 'pid'] = pid_
    sub_trials_df.loc[:, 'pid'] = pid_
    filtered_summ_df_list.append(sub_summary_df)

    for emotion in emotions:
        sub_trials_df.loc[:, emotion] = sub_trials_df.loc[:, emotion].to_numpy() / 48.0

    filtered_trial_df_list.append(sub_trials_df)

assert np.unique([tempdf_.shape[0] for tempdf_ in filtered_summ_df_list]).shape[0] == 1
assert np.unique([tempdf_.shape[0] for tempdf_ in filtered_trial_df_list]).shape[0] == 1


cols_included = [
    'pid',
    'dem_gender',
    'dem_language',
    'iwould_large',
    'iexpectother_large',
    'predicted_C_proport',
]
filtered_summary_df = pd.concat(filtered_summ_df_list).loc[:, cols_included]


cols_included = [
    'pid',
    'stimulus',
    'pot',
    'trial_number',
    'respTimer',
    'Annoyed',
    'Apprehensive',
    'Contemptuous',
    'Content',
    'Devastated',
    'Disappointed',
    'Disgusted',
    'Embarrassed',
    'Excited',
    'Furious',
    'Grateful',
    'Guilty',
    'Hopeful',
    'Impressed',
    'Jealous',
    'Joyful',
    'Proud',
    'Relieved',
    'Surprised',
    'Terrified',
]
filtered_trials_df = pd.concat(filtered_trial_df_list).loc[:, cols_included]
