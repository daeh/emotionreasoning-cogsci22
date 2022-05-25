#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def gen_bs_stimid(stimid_by_outcome, rng=None):
    import numpy as np
    if rng is None:
        rng = np.random.default_rng()
    outcomes = list(stimid_by_outcome.keys())
    return np.hstack([rng.choice(stimid_by_outcome[outcome], size=len(stimid_by_outcome[outcome]), replace=True) for outcome in outcomes])


def bootstrap_accuracy(sim_resp_freq_, stimid_by_outcome=None, nboot=1000, seed=None):
    import numpy as np
    rng = np.random.default_rng(seed)
    accuracies = list()
    for _ in range(nboot):
        bs_stimids = gen_bs_stimid(stimid_by_outcome, rng=rng)
        accuracies.append(np.mean([row_[row_['veridical']] for ii, row_ in sim_resp_freq_.loc[bs_stimids, :].iterrows()]))
    return accuracies


def calc_bs_ci(bootstrap_pointests, alpha=0.05):
    import numpy as np
    percentiles = (alpha * 100. / 2., 100. - alpha * 100. / 2.)
    return np.percentile(bootstrap_pointests, percentiles, axis=0)


def resample_stimid_unbalanced(stim_info, rng=None):
    import pandas as pd
    return pd.DataFrame(stim_info).T.sample(n=len(stim_info), replace=True, random_state=rng).index.to_list()


def calc_bs_summarystats(df_, outcomes, stim_info, alpha=0.05, n_boot=1000, seed=None):
    import numpy as np
    from scipy.stats import pearsonr
    from utils import concordance_correlation_coefficient, coefficient_of_determination, calc_bootstrap_ci

    rng = np.random.default_rng(seed)

    empir_original_ = df_['empir_resp_freq'].loc[list(stim_info.keys()), outcomes].to_numpy().flatten()
    model_original_ = df_['simulated_resp_freq'].loc[list(stim_info.keys()), outcomes].to_numpy().flatten()

    bs_pe_array_ = np.full((n_boot,), np.nan, dtype=float)
    bs_pearsonr, bs_concordance, bs_coeffdet, bs_mse = bs_pe_array_.copy(), bs_pe_array_.copy(), bs_pe_array_.copy(), bs_pe_array_.copy()
    for ii in range(n_boot):
        resampled_stimid = resample_stimid_unbalanced(stim_info, rng=rng)

        empir_ = df_['empir_resp_freq'].loc[resampled_stimid, outcomes].to_numpy().flatten()

        model_ = df_['simulated_resp_freq'].loc[resampled_stimid, outcomes].to_numpy().flatten()

        bs_pearsonr[ii] = pearsonr(empir_, model_)[0]
        bs_concordance[ii] = concordance_correlation_coefficient(empir_, model_)
        bs_coeffdet[ii] = coefficient_of_determination(y=empir_, yhat=model_)
        bs_mse[ii] = np.mean(np.square(empir_ - model_))

    return dict(
        pearsonr_ci=calc_bootstrap_ci(pearsonr(empir_original_, model_original_)[0], np.array(bs_pearsonr), alpha=alpha, flavor='basic'),
        concordance_ci=calc_bootstrap_ci(concordance_correlation_coefficient(empir_original_, model_original_), np.array(bs_concordance), alpha=alpha, flavor='basic'),
        coeff_determination_ci=calc_bootstrap_ci(coefficient_of_determination(y=empir_original_, yhat=model_original_), np.array(bs_coeffdet), alpha=alpha, flavor='basic'),
        rmse_ci=calc_bootstrap_ci(np.sqrt(np.mean(np.square(empir_original_ - model_original_))), np.sqrt(bs_mse), alpha=alpha, flavor='basic'),
    )


def simulate_outcome_judgments(
    empiricalEmotionPredictions=None,
    empiricalEmotionAttributions=None,
    empiricalOutcomeJudgments=None,
    stim_info=None,
    stimid_by_outcome=None,
    bw_list=None,
    empir_resp_freq=None,
    nboot_=10000,
    save_text_var=None,
    paths=None,
    display_param_=None,
    model_name=None,
    seed=None,
):
    import numpy as np
    import pandas as pd
    from emotion_reasoning_model import multivariate_outcome_recovery
    from utils import bootstrap_pe
    from plot_fun import plot_bor_bars_fig, plot_bor_scatter_fig, plot_bor_scatter_grid_fig

    outcomes = list(stimid_by_outcome.keys())

    rng = np.random.default_rng(seed)

    bor_res = multivariate_outcome_recovery(
        empiricalEmotionPredictions=empiricalEmotionPredictions,
        empiricalEmotionAttributions=empiricalEmotionAttributions,
        empiricalOutcomeJudgments=empiricalOutcomeJudgments,
        stim_info=stim_info,
        stimid_by_outcome=stimid_by_outcome,
        bw_list=bw_list)

    ### bootstrap ci ###
    bor_res.update(calc_bs_summarystats(bor_res, outcomes, stim_info, alpha=0.05, n_boot=nboot_, seed=seed))
    for stat_ in ['concordance', 'pearsonr', 'coeff_determination', 'rmse']:
        ### ensure bootstrap percision is sufficient
        assert round(bor_res[stat_] * 10000) >= round(bor_res[f'{stat_}_ci'][0] * 10000), f"{stat_} :: {bor_res[stat_]} vs {bor_res[f'{stat_}_ci']}"
        assert round(bor_res[stat_] * 10000) <= round(bor_res[f'{stat_}_ci'][1] * 10000), f"{stat_} :: {bor_res[stat_]} vs {bor_res[f'{stat_}_ci']}"
        ### save bootstrap estimates
        text_val_ = f"{bor_res[stat_]:.3f} [{bor_res[f'{stat_}_ci'][0]:.3f}, {bor_res[f'{stat_}_ci'][1]:.3f}]%"
        save_text_var.write(text_val_, f'bor-{model_name}_{stat_}.tex')

    ### accuracy with respect to ground truth ###
    bor_acc = np.mean([row_[row_['veridical']] for ii, row_ in bor_res['simulated_resp_freq'].iterrows()])

    ### bootstrap ci of accuracy with respect to ground truth ###
    bor_bsci = calc_bs_ci(bootstrap_accuracy(bor_res['simulated_resp_freq'], stimid_by_outcome=stimid_by_outcome, nboot=nboot_, seed=seed))

    ### save accuracy with respect to ground truth ###
    save_text_var.write(f"{bor_acc*100:0.1f}\% [{bor_bsci[0]*100:0.1f}, {bor_bsci[1]*100:0.1f}]%", f"bor_{model_name}_gt-acc.tex")

    ### bootstrap ci for inferred outcome vs veridical outcome pairs ###

    ci_list_ = list()
    for i_outcome, outcome in enumerate(outcomes):
        for i_outcomepred, outcomepred in enumerate(outcomes):
            ci_temp_ = bootstrap_pe(empir_resp_freq.loc[empir_resp_freq['veridical'] == outcome, outcomepred].to_numpy(), alpha=0.05, bootstrap_samples=nboot_, estimator=np.mean, rng=rng)[1]
            ci_list_.append(dict(true=outcome, pred=outcomepred, cilower=ci_temp_[0], ciupper=ci_temp_[1]))
    empir_ci = pd.DataFrame(ci_list_)

    ci_list_ = list()
    for i_outcome, outcome in enumerate(outcomes):
        for i_outcomepred, outcomepred in enumerate(outcomes):
            ci_temp_ = bootstrap_pe(bor_res['simulated_resp_freq'].loc[bor_res['simulated_resp_freq']['veridical'] == outcome, outcomepred].to_numpy(), alpha=0.05, bootstrap_samples=nboot_, estimator=np.mean, rng=rng)[1]
            ci_list_.append(dict(true=outcome, pred=outcomepred, cilower=ci_temp_[0], ciupper=ci_temp_[1]))
    model_ci = pd.DataFrame(ci_list_)

    ### plot inferred outcome vs veridical outcome -- grid ###
    plot_bor_scatter_grid_fig(
        empir_resp_freq,
        bor_res['simulated_resp_freq'],
        stimid_by_outcome=stimid_by_outcome,
        modelname=model_name,
        paths=paths, display_param_=display_param_)

    ### plot inferred outcome vs veridical outcome -- combined ###
    plot_bor_scatter_fig(
        empir_resp_freq,
        bor_res['simulated_resp_freq'],
        modelname=model_name,
        save_text_var=save_text_var, paths=paths)

    ### plot inferred outcome vs veridical outcome -- bars ###
    if model_name == 'emoreasoning':
        plot_bor_bars_fig(
            empir_resp_freq,
            bor_res['simulated_resp_freq'],
            empir_ci=empir_ci,
            model_ci=model_ci,
            modelname=model_name,
            save_text_var=save_text_var, paths=paths, display_param_=display_param_)

    return bor_res


def emotion_understanding_simulations(empiricalOutcomeJudgments=None, empiricalEmotionJudgments=None, nboot=None, seed=None, stim_info=None, stimid_by_outcome=None, plotParam=None):
    import numpy as np
    import pandas as pd

    display_param_ = plotParam['display_param']
    paths = {'figsOut': plotParam['figsOut'],
             'figsPub': plotParam['figsPub'],
             'varsPub': plotParam['varsPub'], }
    outcomes = list(stimid_by_outcome.keys())

    ##########################################################
    #### Empirical descriptive stats
    ######## x -> (a_1, a_2) accuracy
    ##########################################################

    temp_empr_resp_per_stim = dict()
    for outcome in outcomes:
        for stimid in stimid_by_outcome[outcome]:
            stimdf_ = empiricalOutcomeJudgments.loc[empiricalOutcomeJudgments['stimulus'] == stimid, :]
            temp_empr_resp_per_stim[stimid] = dict()
            for outcome_resp in outcomes:
                temp_empr_resp_per_stim[stimid][outcome_resp] = np.sum(stimdf_['pred_outcome'] == outcome_resp) / stimdf_.shape[0]
            temp_empr_resp_per_stim[stimid]['veridical'] = outcome

    empir_resp_freq = pd.DataFrame(temp_empr_resp_per_stim).T.astype(dict(zip(outcomes, [float] * len(outcomes))))

    del stimdf_
    del temp_empr_resp_per_stim
    assert empir_resp_freq.shape[0] == len(list(stim_info.keys()))

    ### accuracy with respect to ground truth ###
    empir_acc = np.mean([row_[row_['veridical']] for ii, row_ in empir_resp_freq.iterrows()])

    ### bootstrap ci of accuracy with respect to ground truth ###
    empir_bsci = calc_bs_ci(bootstrap_accuracy(empir_resp_freq, stimid_by_outcome=stimid_by_outcome, nboot=nboot, seed=seed))

    ### save accuracy with respect to ground truth ###
    plotParam['save_text_var'].write(f"{empir_acc*100:0.1f}\% [{empir_bsci[0]*100:0.1f}, {empir_bsci[1]*100:0.1f}]%", f"bor_empir_gt-acc.tex")

    ##########################################################
    #### Simulated outcome judgments
    ######## Emotion Reasoning simulation
    ##########################################################

    bor_data_reasoning = dict(
        empiricalEmotionPredictions=empiricalEmotionJudgments.loc[empiricalEmotionJudgments['stimulus']['exp'] == 'CA', :],  # dataset 2
        empiricalEmotionAttributions=empiricalEmotionJudgments.loc[empiricalEmotionJudgments['stimulus']['exp'] == 'XC', :],  # dataset 3
        empiricalOutcomeJudgments=empiricalOutcomeJudgments,
        stim_info=stim_info,
        stimid_by_outcome=stimid_by_outcome,
        bw_list=[0.0],  # bandwith == 0 -> use Scott's rule
    )
    res_reasoning = simulate_outcome_judgments(**bor_data_reasoning, empir_resp_freq=empir_resp_freq, nboot_=nboot, save_text_var=plotParam['save_text_var'], paths=paths, display_param_=display_param_, model_name='emoreasoning', seed=seed)

    ##########################################################
    #### Simulated outcome judgments
    ######## Emotion Recognition simulation
    ##########################################################

    ### gridsearch for bandwidth that produces best fit to empirical data ###
    bor_data_recognition = dict(
        empiricalEmotionPredictions=empiricalEmotionJudgments.loc[empiricalEmotionJudgments['stimulus']['exp'] == 'XC', :],  # dataset 3
        empiricalEmotionAttributions=empiricalEmotionJudgments.loc[empiricalEmotionJudgments['stimulus']['exp'] == 'XC', :],  # dataset 3
        empiricalOutcomeJudgments=empiricalOutcomeJudgments,
        stim_info=stim_info,
        stimid_by_outcome=stimid_by_outcome,
        bw_list=[0.0, *np.arange(start=1.2, stop=1.6, step=0.05)],  # bandwidth list
    )
    res_recognition = simulate_outcome_judgments(**bor_data_recognition, empir_resp_freq=empir_resp_freq, nboot_=nboot, save_text_var=plotParam['save_text_var'], paths=paths, display_param_=display_param_, model_name='emorecognition', seed=seed)

    #### Estimate Bayes Factor ###
    ### Emotion Recognition model is not penalized for the free parameter (bandwidth), which is fit directly to the test data.

    ll_reasoning_list, ll_recognition_list = list(), list()
    for i_row, row_ in empiricalOutcomeJudgments.iterrows():
        ll_reasoning_list.append(np.log(res_reasoning['simulated_resp_freq'].loc[row_['stimulus'], row_['pred_outcome']]))
        ll_recognition_list.append(np.log(res_recognition['simulated_resp_freq'].loc[row_['stimulus'], row_['pred_outcome']]))

    ### log(P(D|M_{reasoning})) ###
    ll_m1 = np.sum(ll_reasoning_list)
    ### log(P(D|M_{recognition})) ###
    ll_m2 = np.sum(ll_recognition_list)

    ### posterior odds ratio (base 10) of the reasoning/recognition models ###
    ### log10( P(D|M_{reasoning}) / P(D|M_{recognition}) ) ###
    l10lr = (ll_m1 - ll_m2) / np.log(10.)

    plotParam['save_text_var'].write("$10^{" + f"{round(l10lr)}" + "}$%", f"bor_bf.tex")
