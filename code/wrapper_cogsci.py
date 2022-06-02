#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""wrapper_cogsci.py
"""

import sys
import argparse


def init_config(base_path_str=None, isInteractive=False):
    from pathlib import Path
    from plot_fun import SaveTextVar
    from plot_fun import get_plt

    if base_path_str is None:
        base_path_str = '~/coding/-GitRepos/itegb_cuecomb_cogsci2022/'
        # script_path = os.path.dirname(os.path.abspath( __file__ ))

    expDir = Path(base_path_str).expanduser().resolve(strict=True)

    paths = dict(
        expDir=expDir,
        ### dataset 1 : (x, c) -> (a_1, a_2) ###
        exp_a_xc_trials=expDir / 'datain/exp_a_xc/dataComposite_trials.csv',
        exp_a_xc_subjects=expDir / 'datain/exp_a_xc/dataComposite_subjects.csv',
        ### dataset 2 : (c, a_1, a_2) -> e ###
        exp_e_ca_trials=expDir / 'datain/exp_e_ca/dataComposite_trials.csv',
        exp_e_ca_subjects=expDir / 'datain/exp_e_ca/dataComposite_subjects.csv',
        ### dataset 3 : (x, c) -> e ###
        exp_e_xc_trials=expDir / 'datain/exp_e_xc/dataComposite_trials.csv',
        exp_e_xc_subjects=expDir / 'datain/exp_e_xc/dataComposite_subjects.csv',
        ####
        episode_key_csv=expDir / 'datain/GoldenBallsEpisodeKey.csv',
    )
    for name_, path_ in paths.items():
        assert path_.exists(), f'Path to {name_} not found: {path_}'

    paths['figsOut'] = paths['expDir'] / 'figs'
    paths['figsPub'] = paths['expDir'] / 'manuscript' / 'figs_autogen'
    paths['varsPub'] = paths['expDir'] / 'manuscript' / 'textvars'

    plt, display_param_ = get_plt(isInteractive=isInteractive)

    plotParam = {
        'plt': plt,
        'display_param': display_param_,
        'save_text_var': SaveTextVar(paths['varsPub']),
        'isInteractive': isInteractive,
        'figsOut': paths['figsOut'],
        'figsPub': paths['figsPub'],
        'varsPub': paths['varsPub'],
    }

    outcomes = ['CC', 'CD', 'DC', 'DD']
    emotion_labels_ordered = ("Disappointed", "Annoyed", "Devastated", "Contemptuous", "Embarrassed", "Disgusted", "Apprehensive", "Furious", "Guilty", "Terrified", "Jealous", "Surprised", "Excited", "Impressed", "Hopeful", "Proud", "Grateful", "Joyful", "Relieved", "Content")

    data_load_param = {
        'label': 'exp-generic',
        'description': '',
        'filter_fn': {  # acceptable values evaluate to True. Criteria not included here are ignored
            'val_notrecognized': lambda x: x,
            'techissues': lambda x: not x,
            'subjectValidation1': lambda x: x,
        },
        'pots': None,
        'outcomes': outcomes,
        'emoLabels': emotion_labels_ordered,
        'ncond': 300,
        'print_responses': False,
    }

    return plotParam, paths, data_load_param


def plot_test(plotParam, paths):
    from plot_fun import printFigList
    print("Starting plot_test")
    plt = plotParam['plt']
    nrows, ncols = 1, 1
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 2 * nrows), constrained_layout=True)
    ax.plot([0, 1, 2], [0, 1, 0.5])
    ax.set_title('$\mathbb{E}[\mathbf{e} \,{;}\, c,a]$ typing')
    fname = f"test-texplot.pdf"
    figsout = [(paths['figsOut'] / fname, fig, True)]
    plt.close(fig)
    figsout = printFigList(figsout, {'plt': plt, 'isInteractive': True, 'showAllFigs': True})
    print("Finished plot_test")


def main(projectdir=None):

    import numpy as np
    from load_empirical_emotion import load_empirical_data
    from outcome_classification import calc_outcomejudgment_reliability, calc_emotionjudgment_reliability, confidence_analysis, calc_bs_by_exp, empirical_groundtruth_nht_stats, player_decision_independence
    from bor import emotion_understanding_simulations

    isInteractive = False
    try:
        if __IPYTHON__:  # type: ignore
            get_ipython().run_line_magic('matplotlib', 'inline')  # type: ignore
            get_ipython().run_line_magic('load_ext', 'autoreload')  # type: ignore
            get_ipython().run_line_magic('autoreload', '2')  # type: ignore
            isInteractive = True
    except NameError:
        isInteractive = False

    plotParam, paths, data_load_param = init_config(base_path_str=projectdir, isInteractive=isInteractive)
    outcomes = data_load_param['outcomes']

    dbuglevel = 'production'
    nboot = {'ultra': 100, 'rapid': 1000, 'demo': 10000, 'production': 50000}[dbuglevel]
    seed = 1

    print(f"\nBeginning run in {paths['expDir']}\n")

    # %%

    plot_test(plotParam, paths)

    # %%
    """
    Load empirical data
    """
    empiricalEmotionJudgments, empiricalOutcomeJudgments, episode_info, stim_info, participants_info, stimid_by_outcome, data_collection_info = load_empirical_data(data_load_param, paths, plotParam)

    # %%
    """
    Empirical reliability
    """
    ### calculate 1 vs rest reliability and bootstrap ci for all empirical data ###

    ## estimate emotion judgment reliability: (a, c) -> (e), (x, c) -> (e) ##
    empir_reliability_onevrest_res = calc_emotionjudgment_reliability(empiricalEmotionJudgments, exp_conds=['CA', 'XC'])

    ## estimate (x, c) -> (a_1, a_2) reliability based on one hot encoding of outcomes ##
    aXC_reliability_corrs_ = calc_outcomejudgment_reliability(empiricalOutcomeJudgments, stimid_by_outcome=stimid_by_outcome)

    ## bootstrap ci ##
    reli_onevrest_corrs, reli_onevrest_plotdata_ = calc_bs_by_exp(empir_reliability_onevrest_res, aXC_reliability_corrs_, bootstrap_samples=nboot, seed=seed, save_text_var=plotParam['save_text_var'])

    ## mean pot size ##
    pots = [stim_info[stimid]['pot'] for stimlist in stimid_by_outcome.values() for stimid in stimlist]
    plotParam['save_text_var'].write(f"{np.mean(pots):,.2f}%", f"mean_pot_usd.tex")

    # %%
    """
    F-score analysis
    """
    empirical_groundtruth_nht_stats(empiricalOutcomeJudgments, stimid_by_outcome=stimid_by_outcome, nboot=nboot, seed=seed, paths=paths, plotParam=plotParam)

    # %%
    """
    Confidence analysis
    """
    confidence_analysis(empiricalOutcomeJudgments, outcomes, plotParam)

    # %%
    """
    Statistics of actual gameplay
    """
    player_decision_independence(paths=paths, plotParam=plotParam)

    # %%
    """
    Bayesian outcome recovery models
    """
    emotion_understanding_simulations(empiricalOutcomeJudgments=empiricalOutcomeJudgments, empiricalEmotionJudgments=empiricalEmotionJudgments, stim_info=stim_info, stimid_by_outcome=stimid_by_outcome, nboot=nboot, seed=seed, plotParam=plotParam)

    # %%

    return 0


def _cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS)
    parser.add_argument('-p', '--projectdir', type=str, help="Path to the project directory")
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    print(f'\n-- Received {sys.argv} from shell --\n')

    exit_status = 1
    try:
        print('STARTING')
        exit_status = main(**_cli())
    except Exception as e:
        print(f'Got exception of type {type(e)}: {e}')
        print("Not sure what happened, so it's not safe to continue -- crashing the script!")
        sys.exit(1)
    finally:
        print(f"-- main() from wrapper_cogsci.py ended with exit code {exit_status} --")

    if exit_status == 0:
        print("-- SCRIPT COMPLETED SUCCESSFULLY --")
    else:
        print(f"-- SOME ISSUE, EXITING:: {exit_status} --")

    sys.exit(exit_status)
