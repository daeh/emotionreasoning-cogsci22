#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def calc_emotionjudgment_reliability(empiricalEmotionJudgments, exp_conds=None):
    """
    calculate the correlation between one observer's emotion judgments with the mean of other observers' emotion judgments for the same stimuli.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr

    emotions = empiricalEmotionJudgments['emotionIntensities'].columns.to_list()

    ##############

    ggdata_acrossemostim_list = list()
    for exp_cond_ in exp_conds:

        data_in = empiricalEmotionJudgments.loc[empiricalEmotionJudgments['stimulus']['exp'] == exp_cond_, :]
        pids = np.unique(data_in['subjectId']['subjectId'].to_numpy())  # all participant ids for experiment 'exp_cond_'

        ntrials_ = 12  # number of stimuli that each observer saw
        pres_temp = np.full([pids.size, ntrials_, len(emotions)], np.nan, dtype=float)
        rest_temp = np.full_like(pres_temp, np.nan)

        for i_pid, pid in enumerate(pids):
            pid_df = data_in.loc[data_in['subjectId']['subjectId'] == pid, :]
            notpid_df = data_in.loc[data_in['subjectId']['subjectId'] != pid, :]
            assert pid_df.shape[0] == ntrials_

            pid_emotions = pid_df['emotionIntensities'].to_numpy()
            notpid_emotions = notpid_df['emotionIntensities'].to_numpy()

            for i_stimid, stimid in enumerate(pid_df['stimulus']['stimid']):
                ### the emotion judgments for stimulus 'i_stimid' made by observer 'i_pid' ###
                pres_temp[i_pid, i_stimid, :] = pid_emotions[pid_df['stimulus']['stimid'] == stimid, :]
                ### mean of the emotion judgments for stimulus 'i_stimid' made by all observers except 'i_pid' ###
                rest_temp[i_pid, i_stimid, :] = notpid_emotions[notpid_df['stimulus']['stimid'] == stimid, :].mean(axis=0)

        assert not np.any(np.isnan(pres_temp))
        assert not np.any(np.isnan(rest_temp))

        coors_across_emo = np.full([pids.size], np.nan, dtype=float)
        for i_pid, pid in enumerate(pids):
            ### correlation of one-vs-rest, across stim, emotion ###
            coors_across_emo[i_pid] = pearsonr(pres_temp[i_pid, :, :].flatten(), rest_temp[i_pid, :, :].flatten())[0]
        ggdata_acrossemostim_list.append(pd.DataFrame(dict(correlation=coors_across_emo, experiment=[exp_cond_ for _ in range(coors_across_emo.size)])))

    return dict(ggdata_acrossemostim=pd.concat(ggdata_acrossemostim_list))


def calc_outcomejudgment_reliability(empiricalOutcomeJudgments, stimid_by_outcome=None):
    import numpy as np
    from scipy.stats import pearsonr

    outcomes = list(stimid_by_outcome.keys())

    stimids = list()
    for outcome in stimid_by_outcome:
        for stimid in stimid_by_outcome[outcome]:
            stimids.append(stimid)
    pids = sorted(empiricalOutcomeJudgments['subjectId'].unique().to_list())

    onehotresps = list()
    for pid in pids:
        onehotarray = np.zeros([len(stimids), len(outcomes)], dtype=int)
        df_ = empiricalOutcomeJudgments.loc[empiricalOutcomeJudgments['subjectId'] == pid, :]
        for i_stimid, stimid in enumerate(stimids):
            onehotarray[i_stimid, :] = (np.array(outcomes) == df_.loc[df_['stimulus'] == stimid, 'pred_outcome'].item()).astype(int)
        onehotresps.append(onehotarray)

    onehotresppop = np.dstack(onehotresps)
    popmean_ = onehotresppop.mean(axis=-1)

    aXC_reliability_corrs_ = list()
    for onehotarray in onehotresps:
        aXC_reliability_corrs_.append(pearsonr(popmean_.flatten(), onehotarray.flatten())[0])

    return aXC_reliability_corrs_


def calc_bs_by_exp(empir_reliability_onevrest_res, aXC_reliability_corrs_, save_text_var=None, bootstrap_samples=10000, alpha=0.05, seed=None):
    import numpy as np
    import pandas as pd
    from utils import bootstrap_pe

    rng = np.random.default_rng(seed)

    ### add (x,c) -> a experiemnt data to emotion judgment data
    aXC_data_temp_ = pd.DataFrame(aXC_reliability_corrs_, columns=['correlation'])
    aXC_data_temp_['experiment'] = 'AgXC'
    reli_onevrest_corrs = pd.concat([empir_reliability_onevrest_res['ggdata_acrossemostim'].copy(), aXC_data_temp_])

    ### calculate 1 vs rest reliability and bootstrap ci
    reli_onevrest_plotdata_ = dict()
    for i_exp_cond_, exp_cond_ in enumerate(reli_onevrest_corrs['experiment'].unique()):
        x_ = reli_onevrest_corrs.loc[reli_onevrest_corrs['experiment'] == exp_cond_, 'correlation'].to_numpy()
        x_median_, x_ci_ = bootstrap_pe(x_, alpha=alpha, bootstrap_samples=bootstrap_samples, estimator=np.median, rng=rng)

        reli_onevrest_plotdata_[exp_cond_] = dict(
            coors=x_,
            median=x_median_,
            ci=x_ci_,
        )

        if exp_cond_ == 'AgXC':
            fname = f'aXC_reliability-onehot.tex'
        else:
            fname = f"reliable_across_emostim_Eg{exp_cond_}.tex"
        str_out = f"{x_median_:0.2f} [{x_ci_[0]:0.2f}, {x_ci_[1]:0.2f}]%"
        save_text_var.write(str_out, fname)

    return reli_onevrest_corrs, reli_onevrest_plotdata_


def calc_fbeta_scores(empiricalOutcomeJudgments, stimid_by_outcome=None, rng=None, plotParam=None):
    import numpy as np
    import pandas as pd
    from utils import calc_confusion_matrix, random_argmax

    outcomes = list(stimid_by_outcome.keys())

    alljudgments_f1 = calc_confusion_matrix(empiricalOutcomeJudgments, outcomes)

    confident_judgments_df = empiricalOutcomeJudgments.loc[(empiricalOutcomeJudgments['predconf_a1'] > 1) & (empiricalOutcomeJudgments['predconf_a2'] > 1), :]
    confident_judgments_df['stimulus'].unique().shape
    confident_judgments_df['subjectId'].unique().shape
    str_out = f"{100 * (confident_judgments_df.shape[0] / empiricalOutcomeJudgments.shape[0]):0.1f}\%%"
    fname = f"prop_confident_judgments.tex"
    plotParam['save_text_var'].write(str_out, fname)
    confidentjudments_f1 = calc_confusion_matrix(confident_judgments_df, outcomes)

    woc_resp = list()
    for outcome in stimid_by_outcome:
        for stimid in stimid_by_outcome[outcome]:

            preds_ = empiricalOutcomeJudgments.loc[empiricalOutcomeJudgments['stimulus'] == stimid, 'pred_outcome']
            woc_pred_ = outcomes[random_argmax(np.array([np.sum(preds_ == outcome_pred) for outcome_pred in outcomes]), axis=0, rng=rng, warn=False)]
            woc_resp.append(dict(subjectId='woc', stimulus=stimid, veridical_outcome=outcome, pred_outcome=woc_pred_))
    wocjudments_f1 = calc_confusion_matrix(pd.DataFrame(woc_resp), outcomes)

    #### get f1 per participant
    subids = sorted(empiricalOutcomeJudgments['subjectId'].unique().to_list())
    substats_f1 = dict()
    for subid in subids:
        subdf = empiricalOutcomeJudgments.loc[empiricalOutcomeJudgments['subjectId'] == subid, :]
        substats_f1[subid] = calc_confusion_matrix(subdf, outcomes)

    return dict(substats_f1=substats_f1, alljudgments_f1=alljudgments_f1, confidentjudments_f1=confidentjudments_f1, wocjudments_f1=wocjudments_f1)


def calc_fbeta_stats(substats_f1, outcomes):
    import numpy as np
    from utils import wilcoxon, check_assumptions

    fbeta_uniform_chance = 1 / len(outcomes)

    modified_sub = dict()
    f1_dict, pc_dict = dict(), dict()
    for i_outcome, outcome in enumerate(outcomes):
        f1_temp = list()
        pc_temp = list()
        for subid in substats_f1:
            f1_temp.append(substats_f1[subid]['fbeta_score'][outcome])
            pc_temp.append(substats_f1[subid]['fbeta_chance'][outcome])
            if substats_f1[subid]['domain_errors']:
                modified_sub[subid] = substats_f1[subid].copy()
        f1_dict[outcome] = np.array(f1_temp)
        pc_dict[outcome] = np.array(pc_temp)

    f1_stats_wilcox = dict()
    f1_stats_vs_uniform_chance_wilcox = dict()
    for outcome in outcomes:
        f1_ = f1_dict[outcome]
        pc_ = pc_dict[outcome]

        ### nonparametric

        # paired Wilcoxon signed-rank test (nonparametric version of ttest_rel)
        # The Wilcoxon signed-rank test tests the null hypothesis that two related paired samples come from the same distribution. In particular, it tests whether the distribution of the differences x - y is symmetric about zero. It is a non-parametric version of the paired T-test.
        f1_stats_wilcox[outcome] = wilcoxon(f1_, pc_)

        # One sample Wilcoxon signed rank test (nonparametric version of ttest_1samp)
        f1_stats_vs_uniform_chance_wilcox[outcome] = wilcoxon(f1_, fbeta_uniform_chance)

    f1_macro_temp, pc_macro_temp = list(), list()
    for subid in substats_f1:
        f1_macro_temp.append(substats_f1[subid]['fbeta_score_macro'])
        pc_macro_temp.append(substats_f1[subid]['fbeta_score_macro_chance'])
    f1_macro = np.array(f1_macro_temp)
    pc_macro = np.array(pc_macro_temp)

    f1_macro_stats_wilcox = wilcoxon(f1_macro, pc_macro)

    ####

    return dict(
        f1=f1_dict,
        pc=pc_dict,
        wilcox=f1_stats_wilcox,
        wilcox_uniform=f1_stats_vs_uniform_chance_wilcox,
    ), dict(
        f1=dict(overall=f1_macro),
        pc=dict(overall=fbeta_uniform_chance),
        wilcox=dict(overall=f1_macro_stats_wilcox),
        wilcox_uniform=dict(overall=wilcoxon(f1_macro, np.full_like(f1_macro, fbeta_uniform_chance))),
    )


def player_decision_independence(paths=None, plotParam=None):
    import numpy as np
    from scipy.stats import chisquare
    from load_empirical_emotion import get_episode_info

    '''
    Test if players decisions are independent of their opponent
    '''

    episode_info, _ = get_episode_info(paths['episode_key_csv'])

    outcomes_obs = dict(zip(*np.unique(episode_info['outcome'], return_counts=True)))
    actions_obs = dict(C=2 * outcomes_obs['CC'] + outcomes_obs['CD'] + outcomes_obs['DC'], D=2 * outcomes_obs['DD'] + outcomes_obs['CD'] + outcomes_obs['DC'])
    n_obs = np.array([outcomes_obs['CC'], outcomes_obs['CD'] + outcomes_obs['DC'], outcomes_obs['DD']])
    f_obs = n_obs / np.sum(n_obs)
    actions_f_obs = np.array([actions_obs['C'], actions_obs['D']]) / (actions_obs['C'] + actions_obs['D'])
    # Hardy-Weinberg equilibrium: p^2 + 2pq + q^2 = 1.0
    f_exp = [actions_f_obs[0]**2, 2 * actions_f_obs[0] * actions_f_obs[1], actions_f_obs[1]**2]
    assert np.isclose(np.sum(f_exp), 1.0)

    c2stat, c2p = chisquare(f_obs, f_exp=f_exp, ddof=1)

    plotParam['save_text_var'].write(f"{actions_f_obs[0]*100:0.1f}\%%", f'GBstats-proportC-allepisodes.tex')
    plotParam['save_text_var'].write(f"$\chi^2_1 = {c2stat:0.3f}$, $p = {c2p:0.3f}$%", f'GBstats-chi2-hardyweinberg.tex')
    plotParam['save_text_var'].write(f"{episode_info.shape[0]}%", f'GBstats-nplayers.tex')


def empirical_winvlose_stats(empiricalOutcomeJudgments, nboot=None, rng=None, plotParam=None):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from utils import bootstrap_pe

    '''win vs loss roc auc'''

    wllist = list()
    remap_ = dict(CC='W', CD='L', DC='W', DD='L')
    for irow_, row_ in empiricalOutcomeJudgments.iterrows():
        wllist.append(dict(
            subjectId=row_['subjectId'],
            stimulus=row_['stimulus'],
            pred_outcome=remap_[row_['pred_outcome']],
            veridical_outcome=remap_[row_['veridical_outcome']],
        ))

    wldf_all = pd.DataFrame(wllist)

    pids = sorted(empiricalOutcomeJudgments['subjectId'].unique())
    wlaccuracy = list()
    wlauroc = list()
    for pid in pids:
        wldf = wldf_all.loc[wldf_all['subjectId'] == pid, :]
        wlaccuracy.append(np.sum(wldf['veridical_outcome'] == wldf['pred_outcome']) / wldf.shape[0])
        vvv = np.zeros(wldf.shape[0])
        vvv[wldf['veridical_outcome'] == 'W'] = 1
        ppp = np.zeros(wldf.shape[0])
        ppp[wldf['pred_outcome'] == 'W'] = 1
        wlauroc.append(roc_auc_score(vvv, ppp))

    wl_accuracy_populmean = bootstrap_pe(np.array(wlaccuracy), alpha=0.05, bootstrap_samples=nboot, estimator=np.mean, rng=rng)

    wl_auroc_populmean = bootstrap_pe(np.array(wlauroc), alpha=0.05, bootstrap_samples=nboot, estimator=np.mean, rng=rng)

    text_val_ = f"{wl_accuracy_populmean[0]*100:0.1f}\% [{wl_accuracy_populmean[1][0]*100:0.1f}, {wl_accuracy_populmean[1][1]*100:0.1f}]%"
    plotParam['save_text_var'].write(text_val_, f'aXC_accuracy_population-mean-winlose.tex')

    text_val_ = f"{wl_auroc_populmean[0]:0.2f} [{wl_auroc_populmean[1][0]:0.2f}, {wl_auroc_populmean[1][1]:0.2f}]%"
    plotParam['save_text_var'].write(text_val_, f'aXC_auroc_population-mean-winlose.tex')


def export_empirical_groundtruth_descriptive_stats(empiricalOutcomeJudgments, fbeta_stats_byoutcome, fscore_populmedian, accuracy_bypid, stimid_by_outcome=None, outcomes=None, nboot=None, rng=None, plotParam=None):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from utils import bootstrap_pe

    outcomes = list(stimid_by_outcome.keys())

    #### ROC-AUC

    veridical_outcome_ = list()
    pred_outcome_ = list()
    for i_outcome, outcome in enumerate(outcomes):
        for stimid in stimid_by_outcome[outcome]:
            sss = empiricalOutcomeJudgments.loc[empiricalOutcomeJudgments['stimulus'] == stimid, 'pred_outcome'].value_counts().loc[outcomes].to_numpy()
            pred_outcome_.append(sss / sss.sum())

            verid_ = np.zeros(len(outcomes), dtype=int)
            verid_[i_outcome] = 1
            veridical_outcome_.append(verid_)

    auroc_bypid = dict()
    for pid in sorted(empiricalOutcomeJudgments['subjectId'].unique().to_list()):
        df_ = empiricalOutcomeJudgments.loc[empiricalOutcomeJudgments['subjectId'] == pid, :]

        veridical_outcome_ = list()
        pred_outcome_ = list()
        for i_outcome, outcome in enumerate(outcomes):
            for stimid in stimid_by_outcome[outcome]:
                sss = df_.loc[df_['stimulus'] == stimid, 'pred_outcome'].value_counts().loc[outcomes].to_numpy()

                pred_outcome_.append(sss / sss.sum())

                verid_ = np.zeros(len(outcomes), dtype=int)
                verid_[i_outcome] = 1
                veridical_outcome_.append(verid_)

        auroc_bypid[pid] = auroc_overall = roc_auc_score(np.vstack(veridical_outcome_), np.vstack(pred_outcome_), multi_class="ovr", average="macro")

    auroc_overall = roc_auc_score(np.vstack(veridical_outcome_), np.vstack(pred_outcome_), multi_class="ovr", average="macro")

    auroc_populmean = bootstrap_pe(np.array(list(auroc_bypid.values())), alpha=0.05, bootstrap_samples=nboot, estimator=np.mean, rng=rng)

    ###### AUROC

    text_val_ = f"{auroc_overall:0.2f}%"
    plotParam['save_text_var'].write(text_val_, f'aXC_auroc_allsubjagg-macro.tex')

    text_val_ = f"{auroc_populmean[0]:0.2f} [{auroc_populmean[1][0]:0.2f}, {auroc_populmean[1][1]:0.2f}]%"
    plotParam['save_text_var'].write(text_val_, f'aXC_auroc_population-mean-macro.tex')

    ###### Fscore

    for i_outcome, outcome in enumerate(outcomes):
        text_val_ = f"{fbeta_stats_byoutcome[outcome]['fscore_median']:0.3f} [{fbeta_stats_byoutcome[outcome]['fscore_median_ci'][0]:0.3f}, {fbeta_stats_byoutcome[outcome]['fscore_median_ci'][1]:0.3f}]%"
        plotParam['save_text_var'].write(text_val_, f'aXC_fscore_population-median-{outcome}.tex')

    text_val_ = f"{fscore_populmedian[0]:0.3f} [{fscore_populmedian[1][0]:0.3f}, {fscore_populmedian[1][1]:0.3f}]%"
    plotParam['save_text_var'].write(text_val_, f'aXC_fscore_population-median-macro.tex')

    ####### Accuracy

    for i_outcome, outcome in enumerate(outcomes):
        text_val_ = f"{fbeta_stats_byoutcome[outcome]['accuracy_mean']*100:0.1f}\% [{fbeta_stats_byoutcome[outcome]['accuracy_mean_ci'][0]*100:0.1f}, {fbeta_stats_byoutcome[outcome]['accuracy_mean_ci'][1]*100:0.1f}]%"
        plotParam['save_text_var'].write(text_val_, f'aXC_accuracy100_population-mean-{outcome}.tex')

    text_val_ = f"{fbeta_stats_byoutcome['overall']['accuracy_mean']:0.2f} [{fbeta_stats_byoutcome['overall']['accuracy_mean_ci'][0]:0.2f}, {fbeta_stats_byoutcome['overall']['accuracy_mean_ci'][1]:0.2f}]%"
    plotParam['save_text_var'].write(text_val_, f'aXC_accuracy_population-mean.tex')

    text_val_ = f"{fbeta_stats_byoutcome['overall']['accuracy_mean']*100:0.1f}\% [{fbeta_stats_byoutcome['overall']['accuracy_mean_ci'][0]*100:0.1f}, {fbeta_stats_byoutcome['overall']['accuracy_mean_ci'][1]*100:0.1f}]%"
    plotParam['save_text_var'].write(text_val_, f'aXC_accuracy100_population-mean.tex')

    text_val_ = f"{np.max(list(accuracy_bypid.values()))*100:0.1f}\%%"
    plotParam['save_text_var'].write(text_val_, "aXC_max_accuracy100.tex")


def summarize_empirical_groundtruth_metrics(fbeta_stats_byoutcome, fbeta_stats_res, fbetamacro_stats_res, outcomes=None, paths=None):
    import pandas as pd

    summary_tab = pd.DataFrame(columns=['Accuracy', 'F-score', 'Wilcoxon', 'Wilcoxon (relative to uniform chance)'], index=['Overall', *outcomes])
    summary_tab.loc[:, :] = '-'
    for outcome_ in fbeta_stats_byoutcome:

        outcome_index_ = "Overall" if outcome_ == 'overall' else outcome_
        summary_tab.loc[outcome_index_, 'F-score'] = fr"{fbeta_stats_byoutcome[outcome_]['fscore_median']:0.2f} [{fbeta_stats_byoutcome[outcome_]['fscore_median_ci'][0]:0.2f},{fbeta_stats_byoutcome[outcome_]['fscore_median_ci'][1]:0.2f}]"

        summary_tab.loc[outcome_index_, 'Accuracy'] = fr"{fbeta_stats_byoutcome[outcome_]['accuracy_mean']:0.2f} [{fbeta_stats_byoutcome[outcome_]['accuracy_mean_ci'][0]:0.2f},{fbeta_stats_byoutcome[outcome_]['accuracy_mean_ci'][1]:0.2f}]"

    for outcome in outcomes:
        summary_tab.loc[outcome, 'Wilcoxon'] = f"{fbeta_stats_res['wilcox'][outcome]['z_str']}, {fbeta_stats_res['wilcox'][outcome]['p_str']}"

        summary_tab.loc[outcome, 'Wilcoxon (relative to uniform chance)'] = rf"{fbeta_stats_res['wilcox_uniform'][outcome]['z_str']}, {fbeta_stats_res['wilcox_uniform'][outcome]['p_str']}"

    summary_tab.loc['Overall', 'Wilcoxon'] = f"{fbetamacro_stats_res['wilcox']['overall']['z_str']}, {fbetamacro_stats_res['wilcox']['overall']['p_str']}"

    summary_tab.loc['Overall', 'Wilcoxon (relative to uniform chance)'] = f"{fbetamacro_stats_res['wilcox_uniform']['overall']['z_str']}, {fbetamacro_stats_res['wilcox_uniform']['overall']['p_str']}"

    dfstyled = summary_tab.style.format(precision=3)
    dfstyled.to_latex(
        paths['varsPub'] / "aXC_summary_table.tex",
        hrules=True,
        column_format="llcccc",
        caption='Human Causal Reasoning - Ground Truth Metrics',
        label='summary_performance',
    )


def confidence_analysis(empiricalOutcomeJudgments, outcomes, plotParam):
    import numpy as np
    import pandas as pd

    from scipy.stats import kendalltau as kendalltaub
    from scipy.stats import spearmanr, pearsonr
    from sklearn.metrics import matthews_corrcoef, confusion_matrix
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    from utils import wilcoxon, check_assumptions

    plt = plotParam['plt']

    subids = empiricalOutcomeJudgments['subjectId'].unique().to_list()

    subdata = dict()
    ### overall, when a participant made a confident judgment, was it more likely to be accurate
    for subid in subids:
        subdf = empiricalOutcomeJudgments.loc[empiricalOutcomeJudgments['subjectId'] == subid, :]
        assert subdf.shape[0] == 88
        subdf_conf = subdf.loc[(subdf['predconf_a1'] == 2) & (subdf['predconf_a2'] == 2), :]
        subdf_notconf = subdf.loc[(subdf['predconf_a1'] < 2) | (subdf['predconf_a2'] < 2), :]
        subdf_conf.shape[0] + subdf_notconf.shape[0] == 88

        conf_n_ = subdf_conf.shape[0]
        conf_proport_ = subdf_conf.shape[0] / subdf.shape[0]
        conf_mean_ = np.mean(np.hstack([subdf['predconf_a1'].to_numpy(), subdf['predconf_a2'].to_numpy()]))
        conf_product_sum_ = np.sum((subdf['predconf_a1'].to_numpy() + 1) * (subdf['predconf_a2'].to_numpy() + 1))

        subdata[subid] = dict()
        for label_, df_ in [('alljudgments', subdf), ('confident', subdf_conf), ('notconfident', subdf_notconf)]:
            n_resp_ = df_.shape[0]

            if n_resp_ > 0:

                y_pred = df_['pred_outcome'].to_numpy()
                y_true = df_['veridical_outcome'].to_numpy()

                hits_ = y_pred == y_true
                n_hits_ = np.sum(hits_)

                skl_confusion = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=outcomes), index=outcomes, columns=outcomes)
                np.sum(skl_confusion, axis=0)  # number of judgments
                np.sum(skl_confusion, axis=1)  # number of true stimuli

                tp_cc_ = skl_confusion.loc['CC', 'CC']
                fp_cc_ = skl_confusion.loc[:, 'CC'].sum() - skl_confusion.loc['CC', 'CC']
                fn_cc_ = skl_confusion.loc['CC', :].sum() - skl_confusion.loc['CC', 'CC']

                tp_cd_ = skl_confusion.loc['CD', 'CD']
                fp_cd_ = skl_confusion.loc[:, 'CD'].sum() - skl_confusion.loc['CD', 'CD']
                fn_cd_ = skl_confusion.loc['CD', :].sum() - skl_confusion.loc['CD', 'CD']

                tp_dc_ = skl_confusion.loc['DC', 'DC']
                fp_dc_ = skl_confusion.loc[:, 'DC'].sum() - skl_confusion.loc['DC', 'DC']
                fn_dc_ = skl_confusion.loc['DC', :].sum() - skl_confusion.loc['DC', 'DC']

                tp_dd_ = skl_confusion.loc['DD', 'DD']
                fp_dd_ = skl_confusion.loc[:, 'DD'].sum() - skl_confusion.loc['DD', 'DD']
                fn_dd_ = skl_confusion.loc['DD', :].sum() - skl_confusion.loc['DD', 'DD']

                micro_num = (tp_cc_ + tp_cd_ + tp_dc_ + tp_dd_)
                micro_denom = (tp_cc_ + fp_cc_ + tp_cd_ + fp_cd_ + tp_dc_ + fp_dc_ + tp_dd_ + fp_dd_)

                if micro_denom > 0:
                    f1_micro_average_ = micro_num / micro_denom
                else:
                    f1_micro_average_ = np.nan

                precision, recall, fbeta_score, support = precision_recall_fscore_support(y_true, y_pred, beta=1.0, average=None, labels=outcomes, warn_for=('precision', 'recall', 'f-score'), zero_division=0)
                precision_macro, recall_macro, fbeta_score_macro, support_macro = precision_recall_fscore_support(y_true, y_pred, beta=1.0, average='macro', labels=outcomes, warn_for=('precision', 'recall', 'f-score'), zero_division=0)

                fbeta_scores_dict__ = dict(zip(outcomes, fbeta_score))
                if tp_cc_ + fp_cc_ + fn_cc_ > 0:
                    fbeta_cc_ = (2 * tp_cc_) / (2 * tp_cc_ + fp_cc_ + fn_cc_)
                else:
                    fbeta_cc_ = np.nan

                if tp_cd_ + fp_cd_ + fn_cd_ > 0:
                    fbeta_cd_ = (2 * tp_cd_) / (2 * tp_cd_ + fp_cd_ + fn_cd_)
                else:
                    fbeta_cd_ = np.nan

                if tp_dc_ + fp_dc_ + fn_dc_ > 0:
                    fbeta_dc_ = (2 * tp_dc_) / (2 * tp_dc_ + fp_dc_ + fn_dc_)
                else:
                    fbeta_dc_ = np.nan

                if tp_dd_ + fp_dd_ + fn_dd_ > 0:
                    fbeta_dd_ = (2 * tp_dd_) / (2 * tp_dd_ + fp_dd_ + fn_dd_)
                else:
                    fbeta_dd_ = np.nan

                fbeta_scores_dict = dict(CC=fbeta_cc_, CD=fbeta_cd_, DC=fbeta_dc_, DD=fbeta_dd_)

                mcc = matthews_corrcoef(y_true, y_pred)

            else:
                fbeta_scores_dict = dict(CC=np.nan, CD=np.nan, DC=np.nan, DD=np.nan)
                f1_micro_average_ = np.nan
                f1_macro_average_ = np.nan
                fbeta_score_macro = np.nan
                mcc = np.nan
                skl_confusion = pd.DataFrame(np.full((len(outcomes), len(outcomes)), np.nan), index=outcomes, columns=outcomes)

            subdata[subid][label_] = dict(fbeta_scores=fbeta_scores_dict, fbeta_score_macro=fbeta_score_macro, fbeta_score_micro=f1_micro_average_, mcc=mcc, confusion=skl_confusion)

        subdata[subid]['alljudgments']['num_max_conf'] = conf_n_
        subdata[subid]['alljudgments']['proport_max_conf'] = conf_proport_
        subdata[subid]['alljudgments']['mean_conf'] = conf_mean_
        subdata[subid]['alljudgments']['sumprod_conf'] = conf_product_sum_

    f1_scores_conf_withinsub = dict()
    f1_scores_conf_withinsub_nandropped = dict()
    for outcome in outcomes:
        temp_list = list()
        for subid, subres in subdata.items():
            temp_list.append(dict(
                subid=subid,
                f1_confident=subres['confident']['fbeta_scores'][outcome],
                f1_notconfident=subres['notconfident']['fbeta_scores'][outcome],
                f1_alljudgments=subres['alljudgments']['fbeta_scores'][outcome]
            ))
        f1_scores_conf_withinsub[outcome] = pd.DataFrame(temp_list)
        f1_scores_conf_withinsub_nandropped[outcome] = pd.DataFrame(temp_list).dropna(axis=0)

    temp_list = list()
    for subid, subres in subdata.items():
        temp_list.append(dict(
            subid=subid,
            f1_confident=subres['confident']['fbeta_score_macro'],
            f1_notconfident=subres['notconfident']['fbeta_score_macro'],
            f1_alljudgments=subres['alljudgments']['fbeta_score_macro']
        ))
    f1_scores_conf_withinsub['alloutcomes'] = pd.DataFrame(temp_list)
    f1_scores_conf_withinsub_nandropped['alloutcomes'] = pd.DataFrame(temp_list).dropna(axis=0)

    outcomes_plus = list(f1_scores_conf_withinsub_nandropped.keys())
    for outcomep in outcomes_plus:
        f1_conf_ = f1_scores_conf_withinsub_nandropped[outcomep]['f1_confident'].to_numpy()
        f1_notconf_ = f1_scores_conf_withinsub_nandropped[outcomep]['f1_notconfident'].to_numpy()
        n_sub = f1_scores_conf_withinsub_nandropped[outcomep].shape[0]
        wc_stat_ = wilcoxon(f1_conf_, f1_notconf_)
        fname = f"withinparticipant_wilcox_conf_vs_notconf_{outcomep}.tex"
        str_out = f"{wc_stat_['z_str']}, {wc_stat_['p_str']}%"
        print(f"{outcomep} (n={n_sub}):: {str_out}")
        plotParam['save_text_var'].write(str_out, fname)

    # %%

    temp_list = list()
    for subid, subres in subdata.items():
        temp_list.append(dict(
            pid=subid,
            fbeta_score_macro=subdata[subid]['alljudgments']['fbeta_score_macro'],
            num_max_conf=subdata[subid]['alljudgments']['num_max_conf'],
            proport_max_conf=subdata[subid]['alljudgments']['proport_max_conf'],
            mean_conf=subdata[subid]['alljudgments']['mean_conf'],
            sumprod_conf=subdata[subid]['alljudgments']['sumprod_conf'],
        ))
    f1_scores_betweensub = pd.DataFrame(temp_list)

    assert f1_scores_betweensub.dropna(axis=0).shape[0] == f1_scores_betweensub.shape[0]

    sumprod_conf_spearman_r, sumprod_conf_spearman_p = spearmanr(f1_scores_betweensub['fbeta_score_macro'].to_numpy(), f1_scores_betweensub['sumprod_conf'].to_numpy())
    kendall_taub, kendall_taup = kendalltaub(f1_scores_betweensub['fbeta_score_macro'].to_numpy(), f1_scores_betweensub['sumprod_conf'].to_numpy())

    proport_max_conf_spearman_r, proport_max_conf_spearman_p = spearmanr(f1_scores_betweensub['fbeta_score_macro'].to_numpy(), f1_scores_betweensub['proport_max_conf'].to_numpy())
    kendall_taub, kendall_taup = kendalltaub(f1_scores_betweensub['fbeta_score_macro'].to_numpy(), f1_scores_betweensub['proport_max_conf'].to_numpy())

    fname = f"betweenparticipant_spearman_f1macro_vs_sumprodconf_alljudge.tex"
    str_out = f"$r_s = {proport_max_conf_spearman_r:0.3f}$, $p = {proport_max_conf_spearman_p:0.3f}$%"
    plotParam['save_text_var'].write(str_out, fname)

    res = dict(subdata=subdata, f1_scores_conf_withinsub=f1_scores_conf_withinsub, f1_scores_conf_withinsub_nandropped=f1_scores_conf_withinsub_nandropped)


def empirical_groundtruth_nht_stats(empiricalOutcomeJudgments, stimid_by_outcome=None, nboot=None, seed=None, paths=None, plotParam=None):

    import numpy as np
    from plot_fun import plot_f1_boxplot_wide
    from utils import bootstrap_pe

    outcomes = list(stimid_by_outcome.keys())

    rng = np.random.default_rng(seed)

    """
    F-score Analysis
    """

    fbeta_res = calc_fbeta_scores(empiricalOutcomeJudgments, stimid_by_outcome=stimid_by_outcome, rng=rng, plotParam=plotParam)

    fbeta_stats_res, fbetamacro_stats_res = calc_fbeta_stats(fbeta_res['substats_f1'], outcomes)

    ##########################################################

    for outcome in outcomes:
        fname = f"allparticipants_wilcox_vs_null_{outcome}.tex"
        str_out = f"{fbeta_stats_res['wilcox'][outcome]['z_str']}, {fbeta_stats_res['wilcox'][outcome]['p_str']}%"
        plotParam['save_text_var'].write(str_out, fname)

    for outcome in outcomes:
        fname = f"allparticipants_wilcox_vs_uniformChance_{outcome}.tex"
        str_out = f"{fbeta_stats_res['wilcox_uniform'][outcome]['z_str']}, {fbeta_stats_res['wilcox_uniform'][outcome]['p_str']}%"
        plotParam['save_text_var'].write(str_out, fname)

    fname = f"allparticipants_wilcox_vs_null_macroAverage.tex"
    str_out = f"{fbetamacro_stats_res['wilcox']['overall']['z_str']}, {fbetamacro_stats_res['wilcox']['overall']['p_str']}%"
    plotParam['save_text_var'].write(str_out, fname)

    ##########################################################

    fscore_bypid = dict()
    fscore_bypid_chance = dict()
    accuracy_bypid = dict()
    fscore_bypidoutcome = dict(zip(outcomes, [list() for _ in outcomes]))
    accuracy_bypidoutcome = dict(zip(outcomes, [list() for _ in outcomes]))
    for pid in sorted(empiricalOutcomeJudgments['subjectId'].unique().to_list()):
        fscore_bypid[pid] = fbeta_res['substats_f1'][pid]['fbeta_score_macro']
        fscore_bypid_chance[pid] = fbeta_res['substats_f1'][pid]['fbeta_score_macro_chance']
        accuracy_bypid[pid] = fbeta_res['substats_f1'][pid]['accuracy']

        for outcome_true in outcomes:
            fscore_bypidoutcome[outcome_true].append(fbeta_res['substats_f1'][pid]['fbeta_score'][outcome_true])

            confmat_ = fbeta_res['substats_f1'][pid]['confusion']
            assert confmat_.loc[outcome_true, :].sum() == 22
            accuracy_bypidoutcome[outcome_true].append(confmat_.loc[outcome_true, outcome_true] / confmat_.loc[outcome_true, :].sum())

    fbeta_stats_byoutcome = dict()
    for outcome in outcomes:

        fscores = fbeta_stats_res['f1'][outcome]
        chancescores = fbeta_stats_res['pc'][outcome]
        fscore_median, fscore_median_ci = bootstrap_pe(fscores, alpha=0.05, bootstrap_samples=nboot, estimator=np.median, rng=rng)
        chance_median, chance_median_ci = bootstrap_pe(chancescores, alpha=0.05, bootstrap_samples=nboot, estimator=np.median, rng=rng)

        accuracy_mean, accuracy_mean_ci = bootstrap_pe(np.array(accuracy_bypidoutcome[outcome]), alpha=0.05, bootstrap_samples=nboot, estimator=np.mean, rng=rng)

        fbeta_stats_byoutcome[outcome] = dict(
            fscore_median=fscore_median,
            fscore_median_null=chance_median,
            fscore_median_ci=fscore_median_ci,
            fscore_median_null_ci=chance_median_ci,
            p=fbeta_stats_res['wilcox'][outcome]['p'],
            ###########
            fscores_=fscores,
            fscores_null_=chancescores,
            ###########
            accuracy_mean=accuracy_mean,
            accuracy_mean_ci=accuracy_mean_ci,
        )

    fscore_populmedian = bootstrap_pe(np.array(list(fscore_bypid.values())), alpha=0.05, bootstrap_samples=nboot, estimator=np.median, rng=rng)

    accuracy_populmean = bootstrap_pe(np.array(list(accuracy_bypid.values())), alpha=0.05, bootstrap_samples=nboot, estimator=np.mean, rng=rng)

    fbeta_stats_byoutcome['overall'] = dict(
        fscore_median=fscore_populmedian[0],
        fscore_median_null=list(fscore_bypid_chance.values()),
        fscore_median_ci=fscore_populmedian[1],
        accuracy_mean=accuracy_populmean[0],
        accuracy_mean_ci=accuracy_populmean[1],
        ########
        fscores_=list(fscore_bypid.values()),
        fscores_null_=list(fscore_bypid_chance.values()),
    )

    export_empirical_groundtruth_descriptive_stats(empiricalOutcomeJudgments, fbeta_stats_byoutcome, fscore_populmedian, accuracy_bypid, stimid_by_outcome=stimid_by_outcome, nboot=nboot, rng=rng, plotParam=plotParam)

    # %%

    ###
    ### aXC_summary_table.tex
    ###
    summarize_empirical_groundtruth_metrics(fbeta_stats_byoutcome, fbeta_stats_res, fbetamacro_stats_res, outcomes=outcomes, paths=paths)

    plot_f1_boxplot_wide(fbeta_stats_byoutcome, paths=paths, outcome_colors=plotParam['display_param']['colors']['outcome-human'])

    empirical_winvlose_stats(empiricalOutcomeJudgments, nboot=nboot, rng=rng, plotParam=plotParam)
