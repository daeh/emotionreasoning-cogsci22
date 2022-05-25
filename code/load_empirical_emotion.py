#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def make_nobs_df(datasheet_in, outcomes=None, pots=None):
    import numpy as np
    import pandas as pd

    if outcomes is None:
        outcomes = np.unique(datasheet_in['outcome'].values)
    if pots is None:
        pots = np.unique(datasheet_in['pot'].values)

    index_outcome = np.full((len(outcomes), datasheet_in.shape[0]), False, dtype=bool)
    index_pot = np.full((len(pots), datasheet_in.shape[0]), False, dtype=bool)
    for i_outcome, outcome in enumerate(outcomes):
        index_outcome[i_outcome, :] = datasheet_in['outcome'] == outcome
    for i_pot, pot in enumerate(pots):
        index_pot[i_pot, :] = datasheet_in['pot'] == pot
    nobsarray = np.full((len(pots), len(outcomes)), 0, dtype=int)
    for i_outcome, outcome in enumerate(outcomes):
        for i_pot, pot in enumerate(pots):
            nobsarray[i_pot, i_outcome] = (index_outcome[i_outcome, :] & index_pot[i_pot, :]).sum()

    nobsdf = pd.DataFrame(data=nobsarray, index=pots, columns=outcomes, dtype=np.int64)
    nobsdf.index.set_names(['pots'], inplace=True)

    assert nobsdf.sum().sum() == datasheet_in.shape[0]

    return nobsdf


def currency_conversion(value_gbp, season):

    exchange_rate = dict()
    inflation = dict()

    ### http://www.x-rates.com/average/?from=GBP&to=USD&amount=1&year=2008
    exchange_rate['1'] = 1.947389  # GPB -> USD in March 2007
    exchange_rate['2'] = 2.044439  # GPB -> USD in October 2007
    exchange_rate['3'] = 1.982679  # GPB -> USD in April 2008
    exchange_rate['4'] = 1.884860  # GPB -> USD in August 2008
    exchange_rate['5'] = 1.533000  # GPB -> USD in November 2008

    ### http://www.in2013dollars.com/2008-dollars-in-2017?amount=1
    inflation['1'] = 1.16  # 2007 -> 2017
    inflation['2'] = 1.16  # 2007 -> 2017
    inflation['3'] = 1.16  # 2007 -> 2017
    inflation['4'] = 1.12  # 2008 -> 2017
    inflation['5'] = 1.12  # 2008 -> 2017

    return round((value_gbp * exchange_rate[f'{season}'] * inflation[f'{season}']) * 2) / 2


def get_episode_info(episode_key_csv_path):
    import numpy as np
    import pandas as pd

    dtype_dict = {
        "Season": str,
        "P1.Decision": str,
        "P2.Decision": str,
        "P1.Gender": str,
        "P2.Gender": str,
    }
    converters_dict = {
        "No.": lambda x: str(x).zfill(3),
        "Pot": lambda x: float(str(x).replace(',', '')) if len(str(x)) > 0 else np.nan,
    }
    episode_key_temp = pd.read_csv(
        episode_key_csv_path,
        header=0, index_col=None,
        dtype=dtype_dict,
        converters=converters_dict,
    )
    stim_no = episode_key_temp['No.'].fillna(0).astype(int)
    episode_key_temp1 = episode_key_temp.loc[(stim_no > 0) & (stim_no <= 289), list(set(dtype_dict.keys()).union(set(converters_dict.keys())))]

    decision_key = {'1': 'C', '2': 'D'}
    stim_key_list, stim_errors = list(), list()
    for irow in range(episode_key_temp1.shape[0]):
        epi_row = episode_key_temp1.iloc[irow, :]
        try:
            gamenumber_ = epi_row.loc['No.']
            season_ = int(epi_row.loc['Season'])
            p1decision_ = decision_key[epi_row.loc['P1.Decision']]
            p2decision_ = decision_key[epi_row.loc['P2.Decision']]
            pot_gbp_ = epi_row.loc['Pot']
            for player_ in ['1', '2']:
                stim_key_list.append(dict(
                    stimulus=f"{gamenumber_}_{player_}",
                    outcome=f"{p1decision_}{p2decision_}" if player_ == '1' else f"{p2decision_}{p1decision_}",
                    gender=epi_row.loc[f"P{player_}.Gender"],
                    pot_usd=currency_conversion(pot_gbp_, season_),
                    pot_gbp=pot_gbp_,
                ))
        except:
            stim_errors.append(epi_row)

    episode_info_ = pd.DataFrame(stim_key_list).convert_dtypes().sort_values(by='stimulus')
    assert np.unique(episode_info_['stimulus'].to_numpy()).size == episode_info_.shape[0]

    return episode_info_.set_index('stimulus'), stim_errors


def import_responses_expC(path_data, data_stats, data_load_param=None):
    import numpy as np
    import pandas as pd

    if data_load_param is None:
        data_load_param = dict()
    emoLabels = data_load_param.get('emoLabels', None)

    #######
    ### Read in response data
    #######

    def unitscale(x): return int(x) / 48.0

    datasheet_temp = pd.read_csv(
        path_data,
        header=0, index_col=None,
        dtype={
            "experiment": str,
            "subjectId": str,
            "stimulus": str,
            "outcome": str,
            "pot": float,
            "gender": str,
        },
        converters=dict([(emotion, unitscale) for emotion in emoLabels]),
    )

    data_stats['nresp_loaded'] = datasheet_temp.shape[0]
    data_stats['_nobs_unfiltered'] = make_nobs_df(datasheet_temp)

    return datasheet_temp


def import_participants_expC(path_subjecttracker, data_load_param=None):
    import numpy as np
    import pandas as pd

    #################

    datasheet_participants = pd.read_csv(
        path_subjecttracker,
        header=0, index_col=None,
        dtype={
            "subjectId": str,
            "randCondNum": int,
            "validationRadio": str,
            "subjectValidation1": bool,
            "val_notrecognized": bool,
            "techissues": bool,
            "dem_gender": str,
            "dem_language": str,
            "val_recognized": str,
            "val_feedback": str,
        },
    )

    return datasheet_participants


def package_empdata_expC(data_included, subjects_included, data_stats, data_load_param=None):
    """
    data_included, subjects_included, data_stats, data_load_param= datasheet_temp, datasheet_participants, data_stats, data_load_param
    """

    import numpy as np
    import pandas as pd
    from pandas.api.types import CategoricalDtype
    import warnings

    emoLabels = data_load_param.get('emoLabels', None)
    outcomes = data_load_param['outcomes']

    ##############
    ## pull data by subject, repackage minimal variables
    ##############

    ### make categorical after selecting which data to include

    data_included['outcome'] = data_included['outcome'].astype(CategoricalDtype(ordered=False, categories=['CC', 'CD', 'DC', 'DD']))

    for catfield in ["stimulus", "pot", "gender"]:
        data_included[catfield] = data_included[catfield].astype(CategoricalDtype(ordered=False))

    nresp_list = list()
    empemodata_bysubject_list = []
    for i_subid, subid in enumerate(subjects_included):
        subjectData = data_included.loc[data_included['subjectId'] == subid, :]

        tempdict = dict()
        nTrials = subjectData.shape[0]
        # prob = 1/nTrials
        nresp_list.append(nTrials)

        for feature in emoLabels:
            tempdict[('emotionIntensities', feature)] = subjectData[feature]

        tempdict[('stimulus', 'stimid')] = subjectData['stimulus']
        tempdict[('stimulus', 'outcome')] = subjectData['outcome']
        tempdict[('stimulus', 'pot')] = subjectData['pot']
        tempdict[('stimulus', 'exp')] = subjectData['experiment']
        tempdict[('subjectId', 'subjectId')] = f'mturk-{subid}'
        # tempdict[('prob','prob')] = [prob]*nTrials

        empemodata_bysubject_list.append(pd.DataFrame.from_dict(tempdict).reset_index(drop=True))

    assert len(np.unique(nresp_list)) == 1
    data_stats['nresp_per_sub_retained'] = nresp_list[0]
    alldata = pd.concat(empemodata_bysubject_list)

    for catfield in ["subjectId"]:
        alldata[catfield] = alldata[catfield].astype(CategoricalDtype(ordered=False))

    pots = np.unique(alldata[('stimulus', 'pot')])
    assert np.all(pots == data_included['pot'].cat.categories)

    ###
    tempdfdict = dict()
    for outcome in outcomes:
        tempdfdict[outcome] = [None] * len(pots)
    nobsdf = pd.DataFrame(data=np.full((len(pots), len(outcomes)), 0, dtype=int), index=pots, columns=outcomes, dtype=np.int64)
    nobsdf.index.set_names(['pots'], inplace=True)

    empiricalEmotionJudgments = dict()
    for outcome in outcomes:
        for i_pot, pot in enumerate(pots):
            df = alldata.loc[(alldata[('stimulus', 'outcome')] == outcome) & (alldata[('stimulus', 'pot')] == pot), 'emotionIntensities']
            nobsdf.loc[pot, outcome] = df.shape[0]
            if df.shape[0] > 0:
                tempdfdict[outcome][i_pot] = df.reset_index(inplace=False, drop=True)

        empiricalEmotionJudgments[outcome] = pd.concat(tempdfdict[outcome], axis=0, keys=pots, names=['pots', None])
    empiricalEmotionJudgments['nobs'] = nobsdf

    return alldata  # empiricalEmotionJudgments


def calc_filter_criteria_expC(datasheet_temp, datasheet_participants, data_stats, data_load_param=None):
    import numpy as np
    import pandas as pd
    import warnings

    emoLabels = data_load_param.get('emoLabels', None)

    ########
    # Test that all subjects have same number of responses
    ########
    unique_sub_batch2 = np.unique(datasheet_participants['subjectId'].values)
    unique_sub_batch1 = np.unique(datasheet_temp['subjectId'].values)

    nresponses = list()
    for subject in unique_sub_batch2:
        nresponses.append(np.sum(datasheet_temp['subjectId'] == subject))
    # assert len(np.unique(nresponses)) == 1, "subjects have different numbers of responses"

    ########
    # Test that all responses are associated with a batch_2_ subject
    ########
    np.testing.assert_array_equal(unique_sub_batch1, unique_sub_batch2, err_msg=f"Subjects don't match \nbatch_1:\n{datasheet_temp['subjectId']}\nbatch_2:\n{datasheet_participants['subjectId']}")

    ##########
    ### Subject Filter
    ##########

    datasheet_participants['response_filter'] = np.ones((datasheet_participants.shape[0], 1), dtype=bool)

    validation_df = datasheet_participants.loc[:, ('subjectId', 'val_notrecognized', 'techissues')].copy()
    for val_id in ["subjectValidation1", "response_filter"]:
        validation_df[val_id] = datasheet_participants.loc[:, val_id].copy()

    return validation_df


def print_descriptive_stats_c(filter_df_c, plotParam):
    import numpy as np

    for expt in ['CA']:

        exp_filterdf = filter_df_c.loc[filter_df_c['experiment'] == expt, :]

        ### number collected
        plotParam['save_text_var'].write(f"{exp_filterdf.shape[0]}%", f"exp-Eg{expt}_N_prefilter.tex")

        ### number that had tech issues
        notechissues_df = exp_filterdf.loc[(exp_filterdf['techissues']), :]
        n_techissue = exp_filterdf.shape[0] - notechissues_df.shape[0]
        plotParam['save_text_var'].write(f"{n_techissue}%", f"exp-Eg{expt}_failed1_tech.tex")

        ### number that recognized stim
        norecog_notechissues_df = exp_filterdf.loc[(exp_filterdf['val_notrecognized']) & (exp_filterdf['techissues']), :]
        n_additional1_recog = notechissues_df.shape[0] - norecog_notechissues_df.shape[0]
        plotParam['save_text_var'].write(f"{n_additional1_recog}%", f"exp-Eg{expt}_failed2_recog.tex")

        ### additional number that did not pass comprehension questions
        n_additional2_failed_validation = norecog_notechissues_df.loc[~norecog_notechissues_df['subjectValidation1'], :].shape[0]
        plotParam['save_text_var'].write(f"{n_additional2_failed_validation}%", f"exp-Eg{expt}_failed3_validation.tex")

        ### number included in analysis
        exp_filterdf_included = exp_filterdf.loc[exp_filterdf['subject_included'], :]
        n_included = exp_filterdf_included.shape[0]
        plotParam['save_text_var'].write(f"{n_included}%", f"exp-Eg{expt}_N_included.tex")

        assert exp_filterdf.shape[0] - n_techissue - n_additional1_recog - n_additional2_failed_validation == n_included

        n_female = np.sum(exp_filterdf_included['gender'] == 'female')
        n_male = np.sum(exp_filterdf_included['gender'] == 'male')
        n_unreported = exp_filterdf_included.shape[0] - n_male - n_female

        plotParam['save_text_var'].write(f"{n_female}%", f"exp-Eg{expt}_N_included_female.tex")
        plotParam['save_text_var'].write(f"{n_male}%", f"exp-Eg{expt}_N_included_male.tex")
        plotParam['save_text_var'].write(f"{n_unreported}%", f"exp-Eg{expt}_N_included_unreported.tex")


def print_descriptive_stats_o(participants_info, filter_df_o, plotParam):
    import numpy as np

    ### number collected
    plotParam['save_text_var'].write(f"{filter_df_o.shape[0]}%", f"exp-AgXC_N_prefilter.tex")

    ### number that recognized stim and/or had tech issues
    norecog_notechissues_df = filter_df_o.loc[(filter_df_o['val_notrecognized']), :]
    n_recog_or_techissue = filter_df_o.shape[0] - norecog_notechissues_df.shape[0]
    plotParam['save_text_var'].write(f"{n_recog_or_techissue}%", f"exp-AgXC_failed2_recog.tex")

    ### additional number that did not pass comprehension questions
    n_additional_failed_validation = norecog_notechissues_df.loc[~norecog_notechissues_df['subjectValidation1'], :].shape[0]
    plotParam['save_text_var'].write(f"{n_additional_failed_validation}%", f"exp-AgXC_failed3_validation.tex")

    ### number included in analysis
    exp_filterdf_included = filter_df_o.loc[filter_df_o['subject_included'], :]
    n_included = exp_filterdf_included.shape[0]
    plotParam['save_text_var'].write(f"{n_included}%", f"exp-AgXC_N_included.tex")

    assert filter_df_o.shape[0] - n_recog_or_techissue - n_additional_failed_validation == n_included
    assert n_included == participants_info.shape[0]

    plotParam['save_text_var'].write(f"{filter_df_o.sum()['subject_included']}%", f"exp-AgXC_N_included.tex")

    n_female = np.sum(participants_info['gender'] == 'female')
    n_male = np.sum(participants_info['gender'] == 'male')
    n_unreported = participants_info.shape[0] - n_male - n_female

    plotParam['save_text_var'].write(f"{n_female}%", f"exp-AgXC_N_included_female.tex")
    plotParam['save_text_var'].write(f"{n_male}%", f"exp-AgXC_N_included_male.tex")
    plotParam['save_text_var'].write(f"{n_unreported}%", f"exp-AgXC_N_included_unreported.tex")


def load_data_e_xc(summary_csv_path=None, trials_csv_path=None):
    import numpy as np
    import pandas as pd

    trials_df = pd.read_csv(trials_csv_path, header=0)
    summary_df = pd.read_csv(summary_csv_path, header=0)

    v0 = -1 * np.ones((summary_df.shape[0], 3), dtype=int)
    v0[:, 0] = summary_df['val_trainingvideo'].astype(str) == '7510'
    v0[:, 1] = summary_df['val_emomatch_contemptuous'] == 'disdainful'
    v0[:, 2] = summary_df['val_expressionmatch_joyful'] == 'AF25HAS'

    summary_df['val_notrecognized'] = summary_df.loc[:, 'val_notrecognized'] == 1
    vr = -1 * np.ones((summary_df.shape[0], 1), dtype=int)
    vr[:, 0] = summary_df.loc[:, 'val_notrecognized'] == 1

    v1 = -1 * np.ones((summary_df.shape[0], 4), dtype=int)
    v1[:, 0] = summary_df['payoff_comprehension_CD12_p1'] == '0'
    v1[:, 1] = summary_df['payoff_comprehension_CD12_p2'] == '12000'
    v1[:, 2] = summary_df['payoff_comprehension_DD42_p1'] == '0'
    v1[:, 3] = summary_df['payoff_comprehension_CC42_p1'] == '21000'

    # summary_df.loc[:, 'valid_lessrecog'] = np.all(v0, axis=1).astype(int)
    summary_df.loc[:, 'valid'] = np.all(np.hstack([v0, vr]), axis=1).astype(int)
    # summary_df.loc[:, 'valid_1'] = np.all(v1, axis=1).astype(int)
    # summary_df.loc[:, 'valid_t'] = np.all(np.hstack([v0, v1]), axis=1).astype(int)

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
    valid_pid = summary_df.loc[summary_df['valid'] == 1, 'workerIdAnon']
    assert valid_pid.unique().shape[0] == valid_pid.shape[0]

    filtered_summ_df_list = list()
    filtered_trial_df_list = list()
    for i_subid, subid in enumerate(valid_pid):
        sub_summary_df = summary_df.loc[summary_df['workerIdAnon'] == subid, :].copy()
        sub_trials_df = trials_df.loc[trials_df['workerIdAnon'] == subid, :].copy()
        pid_ = f'mturk-{subid}'
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

    return filtered_trials_df, summary_df


def import_empirical_data_wrapper(path_data, path_subjecttracker, data_load_param, import_responses_fn=None, import_participants_fn=None, calc_filter_criteria_fn=None, package_fn=None, followup_fn=None, plot_param=None, bypass_plotting=False, debug=False):
    import numpy as np
    import pandas as pd
    import warnings

    data_stats = {
        'label': data_load_param.get('label', 'none'),
        'nsub_loaded': None,
        'nsub_retained': None,
        'nresp_loaded': None,
        'nresp_retained': None,
        'nresp_per_sub_retained': None,
        '_nobs_unfiltered': None,
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

    # assert filter_criteria.loc[:, 'subjectId'].unique().shape[0] == filter_criteria.shape[0]
    filter_fn_dict = data_load_param['filter_fn']
    filter_values_dict = {'subjectId': filter_criteria['subjectId'].copy()}

    for criteria, fn in filter_fn_dict.items():
        assert criteria in filter_criteria.columns, criteria
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

    response_selector = datasheet_temp['subjectId'].isin(subjects_included)

    if response_selector.sum() == 0:
        warnings.warn(f'No responses found')

    data_included = datasheet_temp.loc[response_selector, :].copy()
    data_excluded = datasheet_temp.loc[~response_selector, :].copy()

    ### filter response data by included participants
    dataout = package_fn(data_included, subjects_included, data_stats, data_load_param=data_load_param)

    return dataout, filter_, datasheet_participants


def load_empirical_data(data_load_param, paths, plotParam):
    import numpy as np
    import pandas as pd
    from pandas.api.types import CategoricalDtype
    from load_empirical_outcome import get_exp5_data

    import_fn_dict = {
        'import_responses_fn': import_responses_expC,
        'import_participants_fn': import_participants_expC,
        'calc_filter_criteria_fn': calc_filter_criteria_expC,
        'package_fn': package_empdata_expC,
        'followup_fn': None,
    }

    outcomes = data_load_param['outcomes']
    emotion_labels_ordered = data_load_param['emoLabels']

    ###

    episode_info, gb_episode_load_errors_ = get_episode_info(paths['episode_key_csv'])

    empiricalOutcomeJudgments, participants_info, stim_info, filter_df_o = get_exp5_data(paths['exp_a_xc_trials'], paths['exp_a_xc_subjects'], episode_info)
    print_descriptive_stats_o(participants_info, filter_df_o, plotParam)

    stimid_by_outcome = dict(zip(outcomes, [list(), list(), list(), list()]))
    for stimid in stim_info:
        outcome = stim_info[stimid]['outcome']
        stimid_by_outcome[outcome].append(stimid)

    ###
    expt = 'CA'
    empiricalEmotionJudgments_temp, filter_df_c, datasheet_participants = import_empirical_data_wrapper(paths['exp_e_ca_trials'], paths['exp_e_ca_subjects'], data_load_param, **import_fn_dict, plot_param=None)
    empiricalEmotionJudgments_temp_CA = empiricalEmotionJudgments_temp.loc[empiricalEmotionJudgments_temp['stimulus']['exp'] == expt, :].copy()

    assert np.array_equal(filter_df_c.index.to_numpy(), datasheet_participants['subjectId'].to_numpy())
    filter_df_c.loc[:, 'gender'] = datasheet_participants['dem_gender'].to_numpy()
    filter_df_c.loc[:, 'experiment'] = datasheet_participants['experiment'].to_numpy()
    print_descriptive_stats_c(filter_df_c, plotParam)

    #########

    expt = 'XC'
    exp12_filtered_trials_df, exp12_summary_df = load_data_e_xc(trials_csv_path=paths['exp_e_xc_trials'], summary_csv_path=paths['exp_e_xc_subjects'])
    plotParam['save_text_var'].write(f"{exp12_summary_df.shape[0]}%", f"exp-Eg{expt}_N_prefilter.tex")

    ### number that recognized stim
    norecog_df = exp12_summary_df.loc[exp12_summary_df['val_notrecognized'], :]
    n_additional1_recog = exp12_summary_df.shape[0] - norecog_df.shape[0]
    plotParam['save_text_var'].write(f"{n_additional1_recog}%", f"exp-Eg{expt}_failed2_recog.tex")

    ### additional number that did not pass comprehension questions
    n_additional2_failed_validation = np.sum(norecog_df['valid'] == 0)
    plotParam['save_text_var'].write(f"{n_additional2_failed_validation}%", f"exp-Eg{expt}_failed3_validation.tex")

    ### number included in analysis
    n_included_exp12 = np.sum(exp12_summary_df['valid'] == 1)
    plotParam['save_text_var'].write(f"{n_included_exp12}%", f"exp-Eg{expt}_N_included.tex")

    assert exp12_summary_df.shape[0] - n_additional1_recog - n_additional2_failed_validation == n_included_exp12

    n_female_exp12 = np.sum(exp12_summary_df.loc[exp12_summary_df['valid'] == 1, 'dem_gender'] == 'female')
    n_male_exp12 = np.sum(exp12_summary_df.loc[exp12_summary_df['valid'] == 1, 'dem_gender'] == 'male')
    n_other_exp12 = n_included_exp12 - n_female_exp12 - n_male_exp12

    plotParam['save_text_var'].write(f"{n_female_exp12}%", f"exp-Eg{expt}_N_included_female.tex")
    plotParam['save_text_var'].write(f"{n_male_exp12}%", f"exp-Eg{expt}_N_included_male.tex")
    plotParam['save_text_var'].write(f"{n_other_exp12}%", f"exp-Eg{expt}_N_included_unreported.tex")

    ncounts_exp12 = dict()
    for stimid in stim_info:
        ncounts_exp12[stimid] = exp12_filtered_trials_df.loc[exp12_filtered_trials_df['stimulus'] == stimid, :].shape[0]
    np.min(list(ncounts_exp12.values()))
    np.max(list(ncounts_exp12.values()))

    exp12_filtered_trials_df2 = exp12_filtered_trials_df.rename(columns={"stimulus": "stimid", "pid": "subjectId"})
    temp_outcomes = [None] * exp12_filtered_trials_df2.shape[0]
    for ii in range(len(temp_outcomes)):
        stimid = exp12_filtered_trials_df2.iloc[ii, :].loc['stimid']
        temp_outcomes[ii] = stim_info[stimid]['outcome']
    exp12_filtered_trials_df2.loc[:, 'outcome'] = temp_outcomes
    exp12_filtered_trials_df2.loc[:, 'exp'] = expt

    exp12_filtered_trials_df3 = exp12_filtered_trials_df2.loc[:, (*emotion_labels_ordered, 'stimid', 'outcome', 'pot', 'exp', 'subjectId')]

    exp12_filtered_trials_df3.columns = empiricalEmotionJudgments_temp_CA.columns.copy()

    empiricalEmotionJudgments_temp_XC = exp12_filtered_trials_df3.copy()
    empiricalEmotionJudgments_temp_XC.loc[:, ('stimulus', 'exp')] = expt

    empiricalEmotionJudgments = pd.concat([empiricalEmotionJudgments_temp_XC, empiricalEmotionJudgments_temp_CA])

    empiricalEmotionJudgments.loc[:, ('stimulus', 'exp')] = empiricalEmotionJudgments['stimulus']['exp'].astype(CategoricalDtype(ordered=False, categories=['CA', 'XC']))
    empiricalEmotionJudgments.loc[:, ('stimulus', 'outcome')] = empiricalEmotionJudgments['stimulus']['outcome'].astype(CategoricalDtype(ordered=False, categories=['CC', 'CD', 'DC', 'DD']))
    for catfield in ["stimid", "pot"]:
        empiricalEmotionJudgments.loc[:, ('stimulus', catfield)] = empiricalEmotionJudgments['stimulus'][catfield].astype(CategoricalDtype(ordered=False))

    ### Rename participants ###

    pid1 = empiricalOutcomeJudgments['subjectId'].unique()
    ### assert every participant has same number of responses
    _, pid_counts1 = np.unique(empiricalOutcomeJudgments['subjectId'].to_list(), return_counts=True)
    assert np.unique(pid_counts1).size == 1

    pid2 = empiricalEmotionJudgments['subjectId']['subjectId'].unique()
    ### assert every participant has same number of responses
    _, pid_counts2 = np.unique(empiricalEmotionJudgments['subjectId']['subjectId'].to_list(), return_counts=True)
    assert np.unique(pid_counts2).size == 1

    pid_all = [*pid1, *pid2]  # all participants across studies
    ### assert no overlapping participants between studies
    assert np.unique(pid_all).size == len(pid_all)

    ### remap participant IDs
    pid_renamed = {subid: f"mturk{i_subid+1:04}" for i_subid, subid in enumerate(pid_all)}

    ### rename
    empiricalOutcomeJudgments.replace({'subjectId': pid_renamed}, inplace=True)
    empiricalEmotionJudgments.replace({('subjectId', 'subjectId'): pid_renamed}, inplace=True)

    data_collection_info = dict(
        exp12_summary_df=exp12_summary_df,
    )
    return empiricalEmotionJudgments, empiricalOutcomeJudgments, episode_info, stim_info, participants_info, stimid_by_outcome, data_collection_info
