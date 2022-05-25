#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def import_responses_exp5(path_data, data_stats, data_load_param=None):
    import numpy as np
    import pandas as pd
    from pandas.api.types import CategoricalDtype

    if data_load_param is None:
        data_load_param = dict()

    #######
    ### Read in response data
    #######

    datasheet_temp = pd.read_csv(path_data,
                                 header=0, index_col=None,
                                 dtype={
                                     "subjectId": str,
                                     "stimulus": str,
                                     "pot": float,
                                     "trial_number": int,
                                     "BTS_actual_otherDecisionConfidence": int,
                                     "BTS_actual_thisDecisionConfidence": int,
                                     "BTS_actual_payoffQuadrant": int,
                                     "BTS_predicted_otherDecisionConfidence": int,
                                     "BTS_predicted_thisDecisionConfidence": int,
                                 },
                                 converters={},
                                 )

    data_stats['nresp_loaded'] = datasheet_temp.shape[0]

    return datasheet_temp


def import_participants_exp5(path_subjecttracker, data_load_param=None):
    import numpy as np
    import pandas as pd
    from pandas.api.types import CategoricalDtype

    datasheet_participants = pd.read_csv(path_subjecttracker,
                                         header=0, index_col=None,
                                         dtype={
                                             "subjectId": str,
                                             "validationRadio": str,
                                             "subjectValidation1": np.bool,
                                             "dem_gender": str,
                                             "dem_language": str,
                                             "val_recognized": str,
                                             "val_feedback": str,
                                             "HITID": str,
                                         },
                                         )

    return datasheet_participants


def calc_filter_criteria_exp5(datasheet_temp, datasheet_participants, data_stats, data_load_param=None):
    import numpy as np
    import pandas as pd
    import warnings

    ########
    # Test that all subjects have same number of responses
    ########
    unique_sub_batch2 = np.unique(datasheet_participants['subjectId'].values)
    unique_sub_batch1 = np.unique(datasheet_temp['subjectId'].values)

    nresponses = list()
    for subject in unique_sub_batch2:
        nresponses.append(np.sum(datasheet_temp['subjectId'] == subject))
    assert len(np.unique(nresponses)) == 1, "subjects have different numbers of responses"

    ########
    # Test that all responses are associated with a batch_2_ subject
    ########
    np.testing.assert_array_equal(unique_sub_batch1, unique_sub_batch2, err_msg=f"Subjects don't match \nbatch_1:\n{datasheet_temp['subjectId']}\nbatch_2:\n{datasheet_participants['subjectId']}")

    ##########
    ### Subject Filter
    ##########
    datasheet_participants['response_filter'] = np.ones((datasheet_participants.shape[0], 1), dtype=np.bool)

    validation_df = datasheet_participants.loc[:, ("subjectId", "subjectValidation1", "val_notrecognized", "response_filter")].copy()

    return validation_df


def import_empirical_data_wrapper(path_data, path_subjecttracker, data_load_param, import_responses_fn=None, import_participants_fn=None, calc_filter_criteria_fn=None, package_fn=None, followup_fn=None, plot_param=None, bypass_plotting=False, debug=False):
    import numpy as np
    import pandas as pd
    import warnings

    # dataout, data_stats = import_empirical_data_wrapper(path_data, path_subjecttracker, data_load_param, **import_fn_dict)

    # class EmpiricalImporter:
    #     def __init__(exp_label, ):
    #         self.label = exp_label

    data_stats = {
        'label': data_load_param.get('label', 'none'),
        'nsub_loaded': None,
        'nsub_retained': None,
        'nresp_loaded': None,
        'nresp_retained': None,
        'nresp_per_sub_retained': None,
        # '_nobs_unfiltered': None,
    }

    if debug:
        data_stats['debug'] = dict()

    ### fetch response data, filter out practice question
    datasheet_temp = import_responses_fn(path_data, data_stats, data_load_param=data_load_param)

    ### fetch participants data
    datasheet_participants = import_participants_fn(path_subjecttracker, data_load_param=data_load_param)
    data_stats['nsub_loaded'] = datasheet_participants.shape[0]

    ### calculate exclusions criteria
    filter_criteria = calc_filter_criteria_fn(datasheet_temp, datasheet_participants, data_stats, data_load_param=data_load_param)

    assert filter_criteria.loc[:, 'subjectId'].unique().shape[0] == filter_criteria.shape[0]
    filter_fn_dict = data_load_param['filter_fn']
    filter_values_dict = {'subjectId': filter_criteria['subjectId'].copy()}

    for criteria, fn in filter_fn_dict.items():
        assert criteria in filter_criteria.columns
        filter_values_dict[criteria] = filter_criteria[criteria].apply(fn)
        assert filter_values_dict[criteria].dtype == bool

    filter_ = pd.DataFrame(filter_values_dict)
    filter_.set_index('subjectId', inplace=True)

    include_series = filter_.all(axis=1)
    include_series_idx = include_series.values
    subjects_included = include_series[include_series].index
    subjects_excluded = include_series[~include_series].index

    filter_['subject_included'] = include_series

    datasheet_participants['subjectIncluded'] = include_series_idx

    ####
    # relabel subjects, count resp
    ####
    n_resp_list = list()
    subject_rename_dict = dict()
    for i_subid, subid in enumerate(subjects_included):
        subject_rename_dict[subid] = f'mturk-{subid}'

        sub_idx = datasheet_temp['subjectId'] == subid
        n_resp_list.append(sub_idx.sum())
        datasheet_temp.loc[sub_idx, 'subjectId'] = subject_rename_dict[subid]

        part_sub_idx = datasheet_participants['subjectId'] == subid
        assert part_sub_idx.sum() == 1
        datasheet_participants.loc[part_sub_idx, 'subjectId'] = subject_rename_dict[subid]
    assert np.unique(n_resp_list).size == 1

    response_selector = datasheet_temp['subjectId'].isin(list(subject_rename_dict.values()))

    if response_selector.sum() == 0:
        warnings.warn(f'No responses found')

    data_included = datasheet_temp.loc[response_selector, :].copy()
    data_excluded = datasheet_temp.loc[~response_selector, :].copy()

    datasheet_participants_included = datasheet_participants.loc[datasheet_participants['subjectIncluded'], :]

    dataout = data_included
    ### filter data
    ### add demographic data to subject list
    ### return data table, list of subject data tables
    ### return full stats of how many people failed what criteria

    ### filter response data by included participants
    # dataout = package_fn(data_included, subjects_included, data_stats, data_load_param=data_load_param)

    return dataout, datasheet_participants_included, filter_


def get_exp5_data(path_data, path_subjecttracker, episode_info):
    import numpy as np
    from pandas.api.types import CategoricalDtype
    # from import_empirical_cuecomb import currency_conversion

    data_load_param = {
        'filter_fn': {  # acceptable values evaluate to True. Criteria not included here are ignored
            'subjectValidation1': lambda x: x,
            'val_notrecognized': lambda x: x == 1,  # reported recognizing the GB gameshow
        },
        'ncond': None,
    }

    import_responses_fn = import_responses_exp5
    import_participants_fn = import_participants_exp5
    calc_filter_criteria_fn = calc_filter_criteria_exp5

    dout_mturk, participants_mturk, filter_df = import_empirical_data_wrapper(path_data, path_subjecttracker, data_load_param, import_responses_fn=import_responses_fn, import_participants_fn=import_participants_fn, calc_filter_criteria_fn=calc_filter_criteria_fn, package_fn=None, followup_fn=None, plot_param=None, bypass_plotting=False, debug=False)

    decision_key = {'1': 'C', '2': 'D'}
    stim_key_dict = dict()
    for stimid in np.unique(dout_mturk['stimulus'].values):
        assert '_' in stimid
        episode_ = stimid.split('_')[0]
        player_ = stimid.split('_')[1]
        episode_row = episode_info.loc[stimid, :]

        stim_key_dict[stimid] = {
            'outcome': episode_row.loc['outcome'],
            'pot': episode_row.loc['pot_usd'],
            'pot_gbp': episode_row.loc['pot_gbp'],
            'gender': episode_row.loc['gender'],
        }

        assert isinstance(stim_key_dict[stimid]['pot'], float)
        assert isinstance(stim_key_dict[stimid]['pot_gbp'], float)

    # BTS_actual_otherDecisionConfidence
    # BTS_actual_thisDecisionConfidence
    # BTS_predicted_otherDecisionConfidence
    # BTS_predicted_thisDecisionConfidence
    bts_actual_key = {
        "0": ['C', 2],
        "1": ['C', 1],
        "2": ['C', 0],
        "3": ['D', 0],
        "4": ['D', 1],
        "5": ['D', 2],
    }
    col_str = np.full((dout_mturk.shape[0], 1), 'NN', dtype=object)
    col_int = np.full((dout_mturk.shape[0], 1), 0, dtype=int)
    col_float = np.full((dout_mturk.shape[0], 1), 0.0, dtype=float)
    new_cols_dict = {
        'veridical_outcome': col_str.copy(),
        'pred_outcome': col_str.copy(),
        'pred_a1': col_str.copy(),
        'predconf_a1': col_int.copy(),
        'pred_a2': col_str.copy(),
        'predconf_a2': col_int.copy(),
        'BTS_a1': col_float.copy(),
        'BTS_a2': col_float.copy(),
        'potGBP': col_float.copy(),
        'player_gender': col_str.copy(),
    }

    # ****** assert randcond ****
    """
    BTS_predicted_*:: [0,100], with 0 -> C, 100 -> D
    """

    for i_row in range(dout_mturk.shape[0]):
        row = dout_mturk.iloc[i_row, :]

        stimid = row['stimulus']
        stim_key = stim_key_dict[stimid]

        veridical_outcome_ = stim_key['outcome']

        pred_a1_, predconf_a1_ = bts_actual_key[str(row['BTS_actual_thisDecisionConfidence'])]
        pred_a2_, predconf_a2_ = bts_actual_key[str(row['BTS_actual_otherDecisionConfidence'])]

        pred_outcome_ = f"{pred_a1_}{pred_a2_}"

        bts_a1_ = int(50 - row['BTS_predicted_thisDecisionConfidence'])
        bts_a2_ = int(50 - row['BTS_predicted_otherDecisionConfidence'])
        potGBP = stim_key['pot']
        player_gender = stim_key['gender']

        new_cols_dict['veridical_outcome'][i_row, 0] = veridical_outcome_
        new_cols_dict['pred_outcome'][i_row, 0] = pred_outcome_
        new_cols_dict['pred_a1'][i_row, 0] = pred_a1_
        new_cols_dict['predconf_a1'][i_row, 0] = predconf_a1_
        new_cols_dict['pred_a2'][i_row, 0] = pred_a2_
        new_cols_dict['predconf_a2'][i_row, 0] = predconf_a2_
        new_cols_dict['BTS_a1'][i_row, 0] = bts_a1_
        new_cols_dict['BTS_a2'][i_row, 0] = bts_a2_
        new_cols_dict['potGBP'][i_row, 0] = potGBP
        new_cols_dict['player_gender'][i_row, 0] = player_gender

    for col in new_cols_dict:
        dout_mturk[col] = new_cols_dict[col]

    dataout_responses = dout_mturk.loc[:, ('subjectId', 'stimulus', 'pot', 'veridical_outcome', 'pred_outcome', 'pred_a1', 'predconf_a1', 'pred_a2', 'predconf_a2', 'BTS_a1', 'BTS_a2', 'player_gender', 'trial_number', 'gender')]

    dataout_participants = participants_mturk.loc[:, ('subjectId', 'dem_gender')].copy()
    dataout_participants.rename(columns={'dem_gender': 'gender'}, inplace=True)

    dataout_participants['score'] = 0.0
    scores_ = np.zeros(dataout_participants.shape[0])
    for irow in list(range(dataout_participants.shape[0])):
        # subrow_ = dataout_participants.iloc[irow, :]
        # subid = subrow_.loc[:, 'subjectId']
        subid = dataout_participants.iloc[irow, :].loc['subjectId']
        sub_idx = dataout_responses['subjectId'] == subid
        subdf_ = dataout_responses.loc[sub_idx, :]
        score_ = np.sum((subdf_.loc[:, 'veridical_outcome'] == subdf_.loc[:, 'pred_outcome']).to_numpy()) / subdf_.shape[0]
        scores_[irow] = score_
        # dataout_participants.loc[:, 'score'].iloc[irow] = score_
        # subrow_.loc['score'] = score_
    dataout_participants.loc[:, 'score'] = scores_

    dataout_participants_sorted = dataout_participants.sort_values(['score'], ascending=False).reset_index(drop=True)

    dataout_responses['stimulus'] = dataout_responses['stimulus'].astype(CategoricalDtype(ordered=False, categories=np.unique(dataout_responses['stimulus'].values)))
    dataout_responses['pot'] = dataout_responses['pot'].astype(float)
    dataout_responses['veridical_outcome'] = dataout_responses['veridical_outcome'].astype(CategoricalDtype(ordered=False, categories=['CC', 'CD', 'DC', 'DD']))
    dataout_responses['pred_outcome'] = dataout_responses['pred_outcome'].astype(CategoricalDtype(ordered=False, categories=['CC', 'CD', 'DC', 'DD']))
    dataout_responses['pred_a1'] = dataout_responses['pred_a1'].astype(CategoricalDtype(ordered=False, categories=['C', 'D']))
    dataout_responses['pred_a2'] = dataout_responses['pred_a2'].astype(CategoricalDtype(ordered=False, categories=['C', 'D']))
    dataout_responses['predconf_a1'] = dataout_responses['predconf_a1'].astype(int)
    dataout_responses['predconf_a2'] = dataout_responses['predconf_a2'].astype(int)
    dataout_responses['BTS_a1'] = dataout_responses['BTS_a1'].astype(int)
    dataout_responses['BTS_a2'] = dataout_responses['BTS_a2'].astype(int)
    dataout_responses['player_gender'] = dataout_responses['player_gender'].astype(CategoricalDtype(ordered=False, categories=['M', 'F']))
    dataout_responses['trial_number'] = dataout_responses['trial_number'].astype(int)
    dataout_responses['subjectId'] = dataout_responses['subjectId'].astype(CategoricalDtype(ordered=False, categories=np.unique(dataout_responses['subjectId'].values)))

    """
    veridical_outcome -- veridical outcome
    pred_outcome -- predicted outcome
    pred_a1 -- predicted target player's decision
    predconf_a1 -- confidence in target player's decision (0 -> not confident, 1 -> slightly  confident, 2 -> very confident)
    pred_a2 -- predicted opponent player's decision
    predconf_a2 -- confidence in opponent player's decision (0 -> not confident, 1 -> slightly  confident, 2 -> very confident)
    BTS_a1
    BTS_a2
    player_gender
    trial_num
    gender	
    """

    dataout_responses_sorted = dataout_responses.sort_values(['subjectId', 'stimulus']).reset_index(drop=True)

    empiricalOutcomeJudgments = dataout_responses_sorted
    participants_info = dataout_participants_sorted
    stim_info = stim_key_dict

    return empiricalOutcomeJudgments, participants_info, stim_info, filter_df
