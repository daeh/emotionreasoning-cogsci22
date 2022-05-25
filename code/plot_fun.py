#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class SaveTextVar():
    def __init__(self, save_path):

        if save_path.is_file():
            raise Exception(f'There is something here {save_path}')

        if not save_path.is_dir():
            save_path.mkdir(parents=True)

        self.save_path = save_path
        self.overwritten = list()

    def write(self, txt, fname):
        fpath = self.save_path / fname
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)
        if fpath.is_file():
            self.overwritten.append(fpath)
            fpath.unlink()
        fpath.write_text(txt)


def printFigList(figlist, plot_param):
    """
    takes printFigList(figslist, plt) or printFigList(figslist, dict(plt=plt))
    (fpath,fig,showFig,dupPath_)
    """
    from collections.abc import Iterable
    import itertools
    import shutil

    def flatten(l):
        from collections.abc import Iterable
        import itertools
        for el in l:
            if isinstance(el, itertools.chain):
                yield from (flatten(list(el)))
            elif isinstance(el, (Iterable, itertools.chain)) and not isinstance(el, (str, bytes, tuple, type(None))):
                yield from flatten(el)
            else:
                yield el

    def show_figure(fig, plt):
        """
        create a dummy figure and use its
        manager to display 'fig'
        """
        dummy = plt.figure()
        new_manager = dummy.canvas.manager
        new_manager.canvas.figure = fig
        fig.set_canvas(new_manager.canvas)

    if isinstance(plot_param, dict):
        plt = plot_param['plt']
    else:
        plt = plot_param

    if isinstance(figlist, (Iterable, itertools.chain)):
        flatlist = list(flatten(figlist))
    else:
        flatlist = list()

    listidx = list(range(len(flatlist)))

    print(f'Printing {len(flatlist)} figures')

    plt.close('all')

    printed, duplicates, overwritten, unprintable = list(), list(), list(), list()
    for ii, item in enumerate(flatlist):
        if isinstance(item, tuple) and len(item) in [2, 3, 4] and item is not None:
            showFig = False
            dupPath_ = None
            if len(item) == 2:
                fpath, fig = item
            elif len(item) == 3:
                fpath, fig, showFig = item
            elif len(item) == 4:
                fpath, fig, showFig, dupPath_ = item
            else:
                fpath = None
                raise TypeError

            if isinstance(fpath, str):
                from pathlib import Path
                print('OSPATH ERROR')
                print(fpath)
                fpath = Path(fpath)
            directory = fpath.resolve().parent
            filename = fpath.name
            extension = fpath.suffix

            if not directory.exists():
                print('Creating directory: {}'.format(str(directory)))
                directory.mkdir(parents=True, exist_ok=True)
            if fpath.exists():
                overwritten.append(str(fpath))
            show_figure(fig, plt)

            try:
                plt.savefig(fpath, format=extension[1::], bbox_inches='tight')
            except RuntimeError:
                print(f"\n\nERROR -- PRINT FAILED for :: {fpath}\n\n")
                unprintable.append(item)
                plt.close('all')
            if not showFig:
                plt.close(fig)

            if not dupPath_ is None:
                dupPath = dupPath_.resolve()

                if not dupPath.suffix:
                    ### if directory
                    dubDirectory = dupPath
                    dupFile = dubDirectory / filename
                else:
                    ### if file
                    dubDirectory = dupPath.parent
                    dupFile = dupPath.name

                if not fpath.exists():
                    from warnings import warn
                    warn(f'Cannot copy non-existant file {str(fpath)}')
                else:
                    if not dubDirectory.exists():
                        dubDirectory.mkdir(parents=True)

                    shutil.copy(fpath, dubDirectory / dupFile)

            print(f'({ii+1}/{len(flatlist)})\t{fpath.name}\tsaved to\t{str(fpath)}')
            duplicates.append(str(fpath)) if str(fpath) in printed else printed.append(str(fpath))
        else:
            print('ERROR{')
            print(type(item))
            print(item)
            print('}')

    if len(overwritten) > 0:
        print(f'\n{len(overwritten)} files overwritten:')
        _ = [print(item) for item in overwritten if item not in duplicates]
    if len(duplicates) > 0:
        import warnings
        msg = f'\n{len(duplicates)} duplicate figures printed:'
        print(msg)
        for item in duplicates:
            print(item)
        warnings.warn(msg)

    if len(unprintable) == 0:
        print('Figure Printing Complete')
    else:
        import warnings
        warnings.warn(f'printFigList Errors :: {len(unprintable)} figures returned')
    return unprintable


def get_plt(isInteractive=None):
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    if isInteractive is None:
        isInteractive = False
        try:
            if __IPYTHON__:  # type: ignore
                isInteractive = True
        except NameError:
            isInteractive = False

    colors_new = dict(
        allhum_judg='#9FC2DF',
        allhum_judg_line='#1764AF',
        allhum_judg_box='#D4E4F2',
        confhum_judg='#CA9FDF',
        confhum_judg_line='#A140D0',
        woc="#9FDFA5",
        woc_line='#119B1E',
        bayeskernel='#FCD095',
    )

    mplParam = {
        'axes.grid': False,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        # 'savefig.format': 'pdf',
        ####
        'axes.edgecolor': np.array([51, 51, 51]) / 255,
        'axes.labelsize': 14,
        'axes.labelcolor': 'black',
        'axes.labelweight': 540,
        ###
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'xtick.bottom': True,
        'ytick.left': True,
        'xtick.color': np.array([77, 77, 77]) / 255,
        'ytick.color': np.array([77, 77, 77]) / 255,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        ###
        'errorbar.capsize': 4.0,
        ###
        'legend.frameon': False,
        'legend.fontsize': 13,
        'legend.borderpad': 0.1,
        'legend.columnspacing': 0.5,  # distance between text of col 1 and patch of col 2
        'legend.handletextpad': 0.2,  # distance between patch and text label
        'legend.handleheight': 1,
        'legend.handlelength': 1,
        'legend.labelspacing': 0.5,
        ###
        'font.size': 8,
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts}',
    }

    display_param_ = {
        'colors': {
            'outcome-human': dict(CC='#8AE08D', CD='#8BCCE1', DC='#E08BAA', DD='#B5B5B4'),
            'outcome-model': dict(CC='#DDF6DD', CD='#DCF0F6', DC='#F6DCE5', DD='#E8E9E9'),
            'model': colors_new,
        },
        'mplParam': mplParam,
    }

    matplotlib.rcParams.update(mplParam)

    if not isInteractive:
        matplotlib.use('pdf')

    return plt, display_param_


def plot_f1_boxplot_wide(bar1data=None, paths=None, outcome_colors=None):
    import numpy as np

    outcomes = ['CC', 'CD', 'DC', 'DD']
    plt, display_param_ = get_plt()

    barWidth = 0.7
    bar_space = 0.0

    r1 = np.arange(len(outcomes))

    plt.close('all')

    fig, ax = plt.subplots(figsize=(4.5, 3))

    for i_outcome, outcome in enumerate(outcomes):
        xcoord = r1[i_outcome]

        color_ = outcome_colors[outcome]
        bplot = ax.boxplot(bar1data[outcome]['fscores_'],
                           vert=True,
                           usermedians=[bar1data[outcome]['fscore_median']],
                           conf_intervals=[bar1data[outcome]['fscore_median_ci']],
                           positions=[i_outcome],
                           widths=barWidth,
                           notch=True,
                           whiskerprops=dict(linewidth=1.5, color='#8E8E8E', zorder=1),
                           boxprops=dict(linewidth=1.2, color='#3d3d3d', zorder=3),
                           medianprops=dict(linewidth=2, color='#3d3d3d', solid_capstyle='butt', zorder=4),
                           showfliers=True,
                           flierprops=dict(
            marker=(5, 2),
            markersize=6,
            linestyle='none',
            markeredgecolor='#3d3d3d',
            markeredgewidth=0.5,
            zorder=5,
            clip_on=False),
            showcaps=False,
            patch_artist=True,
        )
        for patch in bplot['boxes']:
            patch.set_facecolor(color_)

        line_width = np.array([-1, 1]) * 0.45
        ax.plot(np.full(2, xcoord) + line_width, np.ones(2) * bar1data[outcome]['fscore_median_null'], color='w', linewidth=1.5, linestyle='-', zorder=6)
        ax.plot(np.full(2, xcoord) + line_width, np.ones(2) * bar1data[outcome]['fscore_median_null'], color='k', linewidth=1.8, linestyle=(0, (1, 1)), zorder=6)
        ax.plot(np.full(2, xcoord) + line_width, np.ones(2) * bar1data[outcome]['fscore_median_null'], color='k', linewidth=0.8, linestyle='-', zorder=6)

    ax.set_xticks(r1)
    ax.set_xticklabels(outcomes)
    ax.set_ylabel('F-score')
    ax.set_xlabel('True Outcome')

    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(visible=True, which='major', axis='y')

    ax.set_ylim([0, 1])

    fname = f"outcomePred_f1_boxplot_wide.pdf"
    figsout = [(paths['figsOut'] / fname, fig, True)]
    plt.close(fig)

    figsout = printFigList(figsout, {'plt': plt, 'isInteractive': True, 'showAllFigs': True})


def plot_bor_bars_fig(empir_resp_freq, simulated_resp_freq, empir_ci=None, model_ci=None, save_text_var=None, paths=None, display_param_=None, modelname=None):
    import numpy as np
    import matplotlib.gridspec as gridspec
    from scipy.stats import pearsonr, sem
    from copy import deepcopy
    from utils import concordance_correlation_coefficient

    plt, display_param2_ = get_plt()
    mplParam = display_param2_['mplParam']

    mplParamThis = deepcopy(mplParam)
    mplParamThis['axes.labelsize'] = 15
    outcomes = ['CC', 'CD', 'DC', 'DD']

    np.testing.assert_array_equal(empir_resp_freq.index, simulated_resp_freq.index)
    np.testing.assert_array_equal(empir_resp_freq['veridical'].to_numpy(), simulated_resp_freq['veridical'].to_numpy())

    ###################

    bayes_corr_pearsonr = pearsonr(simulated_resp_freq.loc[:, outcomes].to_numpy().flatten(), empir_resp_freq.loc[:, outcomes].to_numpy().flatten())[0]
    fname = f"bor_predicted_respbyoutcome_{modelname}_pearsonr.tex"
    str_out = f"{bayes_corr_pearsonr:#.3f}%"
    save_text_var.write(str_out, fname)

    bayes_corr_concordance = concordance_correlation_coefficient(simulated_resp_freq.loc[:, outcomes].to_numpy().flatten(), empir_resp_freq.loc[:, outcomes].to_numpy().flatten())
    fname = f"bor_predicted_respbyoutcome_{modelname}_concordance.tex"
    str_out = f"{bayes_corr_concordance:#.3f}%"
    save_text_var.write(str_out, fname)

    for outcome in outcomes:
        bayes_corr_concordance_byoutcome = concordance_correlation_coefficient(simulated_resp_freq.loc[:, outcome].to_numpy().flatten(), empir_resp_freq.loc[:, outcome].to_numpy().flatten())
        fname = f"bor_predicted_respbyoutcome_{modelname}-{outcome}_concordance.tex"
        str_out = f"{bayes_corr_concordance_byoutcome:#.3f}%"
        save_text_var.write(str_out, fname)

    ###################

    plt.close('all')

    ### row heights in inches
    r_margin = 1
    x_tick_height = 0.3
    x_label_height = 0.3
    r_height_legend = 0.7
    r_height_gridcell = 1.3
    r_height_bartitle = 0.3

    ### column widths in inches
    c_margin = 1
    c_width_ = 3
    y_tick_width = 0.5
    y_label_width = 0.3

    no_debug = True
    if no_debug:
        static_graphic_height = 1.
    else:
        static_graphic_height = 100.

    gs_row_heights = {
        'mt': r_margin,
        'legend': r_height_legend,
        'bartitleC': r_height_bartitle,
        'gridC': r_height_gridcell,
        'xtick2_1': x_tick_height / 2,
        'rb0': y_tick_width / 3 + y_tick_width / 6,
        'bartitleD': r_height_bartitle,
        'gridD': r_height_gridcell,
        'xtick2_2': x_tick_height,
        'xlabel2': x_label_height,
        'mb': static_graphic_height,
    }

    gs_col_widths = {
        'ml': c_margin,
        'ylabel1': y_label_width,
        'ytick1': y_tick_width,
        'c1': c_width_,
        'ytick2_1': y_tick_width / 3,
        'ytick2_2': y_tick_width / 3 + y_tick_width / 6,
        'ytick2_3': y_tick_width / 3,
        'c2': c_width_,
        'mr': c_margin,
    }

    gs_row, gs_col = dict(), dict()
    for i_key, key in enumerate(gs_row_heights):
        gs_row[key] = i_key
    for i_key, key in enumerate(gs_col_widths):
        gs_col[key] = i_key

    ### constrain width ###
    heights = np.array(list(gs_row_heights.values()))
    width_temps = np.array(list(gs_col_widths.values()))
    width_max = np.sum(width_temps)
    widths = width_max * (width_temps / width_temps.sum())

    fig = plt.figure(figsize=(np.sum(widths), np.sum(heights)), dpi=100)

    gridspec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0, top=1, bottom=0, left=0, right=1)

    axd = dict()

    ########## First Plot ##########

    axid = f"ylabel_grid"
    axd[axid] = fig.add_subplot(gridspec[gs_row[f"bartitleC"]:gs_row[f"xtick2_2"], gs_col[f"ylabel1"]])
    axd[axid].text(0.5, 0.5, f"Proportion of Responses", rotation=90., ha='center', va='center', size=mplParamThis['axes.labelsize'], weight=mplParamThis['axes.labelweight'])
    axd[axid].axis('off')

    axid = f"xlabel_grid"
    axd[axid] = fig.add_subplot(gridspec[gs_row[f"xlabel2"], gs_col[f"c1"]:gs_col[f"mr"]])
    axd[axid].text(0.5, 0.5, f"Judgment", rotation=0., ha='center', va='center', size=mplParamThis['axes.labelsize'], weight=mplParamThis['axes.labelweight'])
    axd[axid].axis('off')

    axid = f"legend1"

    axd[axid] = fig.add_subplot(gridspec[gs_row[f"legend"], gs_col[f"c1"]:gs_col[f"ytick2_2"]])

    hcc = axd[axid].bar([0.5], [-1], label='CC', color=display_param_['colors']['outcome-human']['CC'], edgecolor='dimgrey', linewidth=1.2)
    hcd = axd[axid].bar([0.5], [-1], label='CD', color=display_param_['colors']['outcome-human']['CD'], edgecolor='dimgrey', linewidth=1.2)
    hdc = axd[axid].bar([0.5], [-1], label='DC', color=display_param_['colors']['outcome-human']['DC'], edgecolor='dimgrey', linewidth=1.2)
    hdd = axd[axid].bar([0.5], [-1], label='DD', color=display_param_['colors']['outcome-human']['DD'], edgecolor='dimgrey', linewidth=1.2)

    mcc = axd[axid].bar([0.5], [-1], label='CC', color=display_param_['colors']['outcome-model']['CC'], edgecolor='dimgrey', linewidth=1.2)
    mcd = axd[axid].bar([0.5], [-1], label='CD', color=display_param_['colors']['outcome-model']['CD'], edgecolor='dimgrey', linewidth=1.2)
    mdc = axd[axid].bar([0.5], [-1], label='DC', color=display_param_['colors']['outcome-model']['DC'], edgecolor='dimgrey', linewidth=1.2)
    mdd = axd[axid].bar([0.5], [-1], label='DD', color=display_param_['colors']['outcome-model']['DD'], edgecolor='dimgrey', linewidth=1.2)

    plt.legend((hcc, mcc, hcd, mcd, hdc, mdc, hdd, mdd), ('CC', 'CC', 'CD', 'CD', 'DC', 'DC', 'DD', 'DD'), ncol=4, loc='right', columnspacing=0.5, fontsize=14)

    axd[axid].text(0.04, 0.95, f"Human:", rotation=0., ha='left', va='top', size=18, weight=mplParamThis['axes.labelweight'])
    axd[axid].text(0.04, 0.73, f"Model:", rotation=0., ha='left', va='top', size=18, weight=mplParamThis['axes.labelweight'])

    axd[axid].set_ylim([0.5, 1])
    axd[axid].set_xticks([])
    axd[axid].set_yticks([])
    axd[axid].axis('off')

    ######

    axid = f"legend2"
    axd[axid] = fig.add_subplot(gridspec[gs_row[f"legend"], gs_col[f"ytick2_3"]:gs_col[f"mr"]])

    correctjudge = axd[axid].bar([0.5], [-1], label='Correct', color="white", edgecolor='dimgrey', linewidth=1.2)
    incorrectjudge = axd[axid].bar([0.5], [-1], label='Incorrect', color="white", edgecolor='dimgrey', hatch='xxx', linewidth=1.2)

    plt.legend((correctjudge, incorrectjudge), ('Correct', 'Incorrect'), ncol=2, loc='right', columnspacing=0.5, fontsize=14)

    axd[axid].text(0.1, 0.83, f"Judgment:", rotation=0., ha='left', va='top', size=18, weight=mplParamThis['axes.labelweight'])

    axd[axid].set_ylim([0.5, 1])
    axd[axid].set_xticks([])
    axd[axid].set_yticks([])
    axd[axid].axis('off')

    ######

    axid = f"CC"
    axd[axid] = fig.add_subplot(gridspec[gs_row[f"gridC"], gs_col[f"c1"]:gs_col[f"ytick2_2"]])
    axid = f"CC-title"
    axd[axid] = fig.add_subplot(gridspec[gs_row[f"bartitleC"], gs_col[f"c1"]:gs_col[f"ytick2_2"]])
    axd[axid].text(0.5, 0.4, f"True Outcome: CC", rotation=0., ha='center', va='center', size=mplParamThis['axes.labelsize'], weight=mplParamThis['axes.labelweight'])
    axd[axid].set_xticks([])
    axd[axid].set_yticks([])

    axid = f"CD"
    axd[axid] = fig.add_subplot(gridspec[gs_row[f"gridC"], gs_col[f"ytick2_3"]:gs_col[f"mr"]])
    axid = f"CC-title"
    axd[axid] = fig.add_subplot(gridspec[gs_row[f"bartitleC"], gs_col[f"ytick2_3"]:gs_col[f"mr"]])
    axd[axid].text(0.5, 0.4, f"True Outcome: CD", rotation=0., ha='center', va='center', size=mplParamThis['axes.labelsize'], weight=mplParamThis['axes.labelweight'])
    axd[axid].set_xticks([])
    axd[axid].set_yticks([])

    axid = f"DC"
    axd[axid] = fig.add_subplot(gridspec[gs_row[f"gridD"], gs_col[f"c1"]:gs_col[f"ytick2_2"]])
    axid = f"CC-title"
    axd[axid] = fig.add_subplot(gridspec[gs_row[f"bartitleD"], gs_col[f"c1"]:gs_col[f"ytick2_2"]])
    axd[axid].text(0.5, 0.4, f"True Outcome: DC", rotation=0., ha='center', va='center', size=mplParamThis['axes.labelsize'], weight=mplParamThis['axes.labelweight'])
    axd[axid].set_xticks([])
    axd[axid].set_yticks([])

    axid = f"DD"
    axd[axid] = fig.add_subplot(gridspec[gs_row[f"gridD"], gs_col[f"ytick2_3"]:gs_col[f"mr"]])
    axid = f"CC-title"
    axd[axid] = fig.add_subplot(gridspec[gs_row[f"bartitleD"], gs_col[f"ytick2_3"]:gs_col[f"mr"]])
    axd[axid].text(0.5, 0.4, f"True Outcome: DD", rotation=0., ha='center', va='center', size=mplParamThis['axes.labelsize'], weight=mplParamThis['axes.labelweight'])
    axd[axid].set_xticks([])
    axd[axid].set_yticks([])

    for outcome in outcomes:
        axid = outcome

        axd[axid].set_xticks(list(range(len(outcomes))))
        axd[axid].set_xticklabels([f"{o_}*" if o_ == outcome else f"{o_}" for o_ in outcomes], fontsize=13)
        axd[axid].set_ylim((0, 0.75))
        axd[axid].set_yticks([0, 0.25, 0.5, 0.75])

        if axid in ['CD', 'DD']:
            axd[axid].set_yticklabels([])
        axd[axid].grid(visible=False, which='major', axis='x')
        axd[axid].grid(visible=True, which='major', axis='y')

        vals_temp = empir_resp_freq.loc[empir_resp_freq['veridical'] == outcome, :].drop(columns=['veridical']).astype(float)
        empir_freq = vals_temp.mean()

        vals_temp = simulated_resp_freq.loc[simulated_resp_freq['veridical'] == outcome, :].drop(columns=['veridical']).astype(float)
        model_freq = vals_temp.mean()

        assert np.isclose(np.sum(empir_freq), 1.0)
        assert np.isclose(np.sum(model_freq), 1.0)

        wid = 0.3
        sp = 0.00

        loc0 = np.array(range(len(outcomes))) - sp / 2 - wid / 2
        loc1 = np.array(range(len(outcomes))) + sp / 2 + wid / 2

        for loc, val, ci_df, outcome_colrs, falpha in [
            (loc0, empir_freq, empir_ci, display_param_['colors']['outcome-human'], 1.0),
            (loc1, model_freq, model_ci, display_param_['colors']['outcome-model'], 1.0)
        ]:
            colr_true = outcome_colrs[outcome]
            for i_outcomepred, outcomepred in enumerate(outcomes):
                err = ci_df.loc[(ci_df['true'] == outcome) & (ci_df['pred'] == outcomepred), :]
                if outcomepred == outcome:
                    loc_ = loc[i_outcomepred]
                    val_ = val[outcomepred]
                    facecolor_ = colr_true
                    edgecolor_ = 'dimgrey'
                    width_ = wid
                    edgewidth_ = 1.2
                    hatch_ = None
                    facealpha = falpha
                else:
                    loc_ = loc[i_outcomepred]
                    val_ = val[outcomepred]
                    facecolor_ = colr_true
                    edgecolor_ = 'dimgrey'
                    width_ = wid
                    edgewidth_ = 1.2
                    hatch_ = 'xx'
                    facealpha = falpha
                ### background mask
                b_ = axd[axid].bar(loc_, val_, width=width_, color='white', edgecolor='k', linewidth=edgewidth_, hatch=None, zorder=89)
                ### face color
                b_ = axd[axid].bar(loc_, val_, width=width_, color=facecolor_, edgecolor=edgecolor_, linewidth=edgewidth_, hatch=None, zorder=90, alpha=facealpha)
                ### hatch
                b_ = axd[axid].bar(loc_, val_, width=width_, color='none', edgecolor=edgecolor_, linewidth=edgewidth_, hatch=hatch_, zorder=91)
                axd[axid].plot([loc[i_outcomepred], loc[i_outcomepred]], [err['cilower'], err['ciupper']], color='k', linewidth=2, zorder=92)

    fname = f"bor-{modelname}_composit_bars.pdf"
    figsout = [(paths['figsOut'] / fname, fig, True)]
    plt.close(fig)

    figsout = printFigList(figsout, {'plt': plt, 'isInteractive': True, 'showAllFigs': True})


def plot_bor_scatter_fig(empir_resp_freq, simulated_resp_freq, save_text_var=None, paths=None, modelname=None):
    import numpy as np
    import matplotlib.gridspec as gridspec
    from scipy.stats import pearsonr, sem
    from utils import concordance_correlation_coefficient

    plt, display_param_ = get_plt()
    mplParam = display_param_['mplParam']
    colors_new = display_param_['colors']['model']

    outcomes = ['CC', 'CD', 'DC', 'DD']

    np.testing.assert_array_equal(empir_resp_freq.index, simulated_resp_freq.index)
    np.testing.assert_array_equal(empir_resp_freq['veridical'].to_numpy(), simulated_resp_freq['veridical'].to_numpy())

    bayes_corr_pearsonr = pearsonr(simulated_resp_freq.loc[:, outcomes].to_numpy().flatten(), empir_resp_freq.loc[:, outcomes].to_numpy().flatten())[0]

    bayes_corr_concordance = concordance_correlation_coefficient(simulated_resp_freq.loc[:, outcomes].to_numpy().flatten(), empir_resp_freq.loc[:, outcomes].to_numpy().flatten())

    ###################

    plt.close('all')

    ### row heights in inches
    r_margin = 1
    r_height_scatter = 2.3
    x_tick_height = 0.3
    x_label_height = 0.3
    r_height_legend = 0.5

    ### column widths in inches
    c_margin = 1
    c_width_ = 3
    y_tick_width = 0.5
    y_label_width = 0.3

    no_debug = True
    if no_debug:
        static_graphic_height = 1.
    else:
        static_graphic_height = 100.

    gs_row_heights = {
        'mt': r_margin,
        'title': r_height_legend,
        'scatter': r_height_scatter,
        'xtick1': x_tick_height,
        'xlabel1': x_label_height,
        'mb': static_graphic_height,
    }

    gs_col_widths = {
        'ml': c_margin,
        'ylabel1': y_label_width,
        'ytick1': y_tick_width,
        'c1': c_width_,
        'mr': c_margin,
    }

    gs_row, gs_col = dict(), dict()
    for i_key, key in enumerate(gs_row_heights):
        gs_row[key] = i_key
    for i_key, key in enumerate(gs_col_widths):
        gs_col[key] = i_key

    ### constrain width ###
    heights = np.array(list(gs_row_heights.values()))
    width_temps = np.array(list(gs_col_widths.values()))
    width_max = np.sum(width_temps)
    widths = width_max * (width_temps / width_temps.sum())

    fig = plt.figure(figsize=(np.sum(widths), np.sum(heights)), dpi=100)

    gridspec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0, top=1, bottom=0, left=0, right=1)

    axd = dict()

    ########## First Plot ##########

    axid = f"bayes"
    axd[axid] = fig.add_subplot(gridspec[gs_row[f"scatter"], gs_col[f"c1"]])

    axd[axid].tick_params(axis='both', pad=3)
    axd[axid].text(0.05, 0.95, f"$ccc = {bayes_corr_concordance:#.2}$", size=12, ha='left', va='top')
    axd[axid].text(0.05, 0.85, f"$r ~=~  {bayes_corr_pearsonr:#.2}$", size=12, ha='left', va='top')
    for outcome in outcomes:
        empir_vals_ = empir_resp_freq.loc[empir_resp_freq['veridical'] == outcome, :].drop(columns=['veridical']).to_numpy().flatten()
        model_vals_ = simulated_resp_freq.loc[simulated_resp_freq['veridical'] == outcome, :].drop(columns=['veridical']).to_numpy().flatten()
        axd[axid].scatter(model_vals_, empir_vals_, s=23, linewidths=0.8, alpha=0.6, marker='o', color='k', facecolors='none')

    axd[axid].plot((0, 1), (0, 1), linewidth=2, color='grey', linestyle='--')
    axd[axid].set_xlim([0, 1])
    axd[axid].set_ylim([0, 1])
    axd[axid].set_xticks([0, 0.25, 0.5, 0.75, 1])
    axd[axid].set_yticks([0, 0.25, 0.5, 0.75, 1])

    axd[axid].grid(visible=False, which='major', axis='x')
    axd[axid].grid(visible=False, which='major', axis='y')

    ######################

    axid = f"title"
    axd[axid] = fig.add_subplot(gridspec[gs_row[f"title"], gs_col[f"c1"]])
    axd[axid].text(0.5, 0.5, f"$p(a|x)$", rotation=0., ha='center', va='center', size=18, weight=mplParam['axes.labelweight'])
    axd[axid].axis('off')

    axid = f"ylabel_scatter"
    axd[axid] = fig.add_subplot(gridspec[gs_row[f"scatter"]:gs_row[f"xtick1"], gs_col[f"ylabel1"]])
    axd[axid].text(0.5, 0.5, f"Human", rotation=90., ha='center', va='center', size=mplParam['axes.labelsize'], weight=mplParam['axes.labelweight'])
    axd[axid].axis('off')

    axid = f"xlabel_scatter1"
    axd[axid] = fig.add_subplot(gridspec[gs_row[f"xlabel1"], gs_col[f"c1"]])
    axd[axid].text(0.5, 0.5, f"Model", rotation=0., ha='center', va='center', size=mplParam['axes.labelsize'], weight=mplParam['axes.labelweight'])
    axd[axid].axis('off')

    figsout = [(paths['figsOut'] / f"bor-{modelname}_composit_scatter.pdf", fig, False)]
    plt.close(fig)

    figsout = printFigList(figsout, {'plt': plt, 'isInteractive': True, 'showAllFigs': True})


def plot_bor_scatter_grid_fig(empir_resp_freq, simulated_resp_freq, stimid_by_outcome=None, paths=None, display_param_=None, modelname=None, figtitle=None):
    import matplotlib.gridspec as gridspec
    from scipy.stats import pearsonr, sem

    plt, display_param2_ = get_plt()

    outcomes = ['CC', 'CD', 'DC', 'DD']

    plt.close('all')

    nrows = 4
    ncols = 4
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=((5 / 1.2) / 1.5 * ncols, (4 / 1.2) / 1.5 * nrows), constrained_layout=True, sharex=True, sharey=True)

    for i_outcomepred, outcomepred in enumerate(outcomes):
        for i_outcome, outcome in enumerate(outcomes):
            for stimid in stimid_by_outcome[outcome]:
                axs[i_outcome, i_outcomepred].scatter(simulated_resp_freq.loc[stimid, outcomepred].flatten(), empir_resp_freq.loc[stimid, outcomepred].flatten(), color=display_param_['colors']['outcome-human'][outcome], alpha=1, marker='o', facecolors='none', zorder=5)

            axs[i_outcome, i_outcomepred].set_ylim([0, 1])
            axs[i_outcome, i_outcomepred].set_xlim([0, 1])
            axs[i_outcome, i_outcomepred].plot((0, 1), (0, 1), linewidth=1, color='grey', linestyle='--', zorder=1)
            axs[i_outcome, i_outcomepred].set_xticks([0, 0.25, 0.5, 0.75, 1])
            axs[i_outcome, i_outcomepred].set_yticks([0, 0.25, 0.5, 0.75, 1])

            axs[i_outcome, i_outcomepred].text(0.03, 0.88, r"True:$~~~~~~~~$" + f"{outcome}\nJudgment: {outcomepred}", horizontalalignment='left', verticalalignment='center', transform=axs[i_outcome, i_outcomepred].transAxes)

    for i_outcome, outcome in enumerate(outcomes):
        axs[i_outcome, 0].set_ylabel(f'True Outcome: {outcome}\n\nHuman', fontdict={'fontsize': 15})
    for i_outcomepred, outcomepred in enumerate(outcomes):
        axs[0, i_outcomepred].set_title(f"Inferred Outcome: {outcomepred}\n", fontdict={'fontsize': 15})
        axs[-1, i_outcomepred].set_xlabel(f"Model", fontdict={'fontsize': 15})

    if figtitle is not None:
        plt.suptitle(figtitle, fontsize=22, y=1.04, fontweight='bold')

    fname = f"bor-{modelname}_gridscatter.pdf"
    figsout = [(paths['figsOut'] / fname, fig, False)]
    plt.close(fig)

    figsout = printFigList(figsout, {'plt': plt, 'isInteractive': True, 'showAllFigs': True})
