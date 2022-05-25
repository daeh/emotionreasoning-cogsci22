#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def weighted_descriptive_stats_byoutcome(emodf):
    """
    Calculates SD of each emotion (across outcomes), weighted by the number of observations of each stimulus.
    """
    import numpy as np
    import pandas as pd

    stimids = sorted(emodf['stimulus']['stimid'].unique())

    prob_stim_ = len(stimids)**-1

    resp_values_list_ = list()
    resp_probs_list_ = list()
    for stimid in stimids:
        data_by_stim_ = emodf.loc[emodf['stimulus']['stimid'] == stimid, 'emotionIntensities']
        # n_obs_ = np.sum(emodf['stimulus']['stimid'] == stimid)
        n_obs_ = data_by_stim_.shape[0]
        assert n_obs_ > 0
        prob_resp_ = n_obs_**-1
        resp_values_list_.append(data_by_stim_.to_numpy())
        resp_probs_list_.append(np.full([n_obs_, 1], prob_stim_ * prob_resp_, dtype=np.float64))
    x_ = np.vstack(resp_values_list_)
    p_ = np.vstack(resp_probs_list_)
    assert x_.shape[0] == p_.shape[0]
    assert np.isclose(np.sum(p_), 1.0)

    p_sum_to_nobs_ = p_ * emodf.shape[0]

    mean_weighted_ = dict()
    ev_unweighted_ = dict()
    sd_weighted_ = dict()
    sd_weighted_sample_ = dict()
    sd_unweighted_ = dict()
    sd_unweighted_sample_ = dict()
    for i_emotion, emotion in enumerate(emodf['emotionIntensities'].columns.to_list()):
        mean_weighted_[emotion] = np.dot(x_[:, i_emotion], p_)[0]
        sd_weighted_[emotion] = np.sqrt(np.dot(np.square(x_[:, i_emotion] - mean_weighted_[emotion]), p_))[0]

        sd_weighted_sample_[emotion] = np.sqrt(np.dot(np.square(x_[:, i_emotion] - mean_weighted_[emotion]), p_sum_to_nobs_) / (p_sum_to_nobs_.sum() - 1))[0]

        ev_unweighted_[emotion] = emodf.loc[:, ('emotionIntensities', emotion)].mean()
        sd_unweighted_[emotion] = emodf.loc[:, ('emotionIntensities', emotion)].std()
        sd_unweighted_sample_[emotion] = emodf.loc[:, ('emotionIntensities', emotion)].std(ddof=1)

    return dict(mean_weighted=pd.Series(mean_weighted_), sd_weighted=pd.Series(sd_weighted_), sd_weighted_sample=pd.Series(sd_weighted_sample_), mean_unweighted=pd.Series(ev_unweighted_), sd_unweighted=pd.Series(sd_unweighted_), sd_unweighted_sample=pd.Series(sd_unweighted_sample_))


def scotts_rule(emodf):
    """
    Scott DW. Multivariate density estimation: theory, practice, and visualization. New York: Wiley; 1992. 317 p. (Wiley series in probability and mathematical statistics). 
    Eq. 6.42
    """
    import numpy as np
    sd_ = weighted_descriptive_stats_byoutcome(emodf)['sd_unweighted']
    nobs_, ndims_ = emodf['emotionIntensities'].shape
    return sd_ * np.power(nobs_, -1.0 / (ndims_ + 4))


def calc_bootstrap_ci(theta_hat, theta_tilde_samples, alpha=0.05, flavor='percentile'):
    '''
    theta_hat : statistic estimate from the original sample
    theta_tilde : statistic estimate from a bootstrap sample
    '''
    import numpy as np

    percentiles = (alpha * 100. / 2., 100. - alpha * 100. / 2.)

    if flavor == 'percentile':
        q_low, q_high = np.percentile(theta_tilde_samples, percentiles, axis=0)
        ci_ = np.array([q_low, q_high])

    elif flavor == 'basic':
        '''
        1 - alpha = P(q_{alpha/2} - theta_hat <= theta_tilde - theta_hat <= q_{1-alpha/2} - theta_hat)
                  = P(2*theta_hat - q_{1-alpha/2} <= theta <= 2*theta_hat - q_{alpha/2})
        '''
        q_low, q_high = np.percentile(theta_tilde_samples, percentiles, axis=0)
        bias = np.mean(theta_tilde_samples) - theta_hat
        ci_ = np.array([2 * theta_hat - q_high, 2 * theta_hat - q_low])

    return ci_


def coefficient_of_determination(y=None, yhat=None):
    import numpy as np

    if isinstance(y, np.ndarray):
        yemp_ = y
    elif isinstance(y, list):
        yemp_ = np.array(y)
    else:
        raise TypeError

    if isinstance(yhat, np.ndarray):
        yhat_ = yhat
    elif isinstance(yhat, list):
        yhat_ = np.array(yhat)
    else:
        raise TypeError

    for v_ in [yemp_, yhat_]:
        assert v_.dtype in [float, int], v_.dtype

    ssres = np.sum(np.square(yemp_ - yhat_))
    sstot = np.sum(np.square(yemp_ - np.mean(yemp_)))
    return 1.0 - (ssres / sstot)


def concordance_correlation_coefficient(x_in_, y_in_):
    import numpy as np

    if isinstance(x_in_, np.ndarray):
        x_ = x_in_
    elif isinstance(x_in_, list):
        x_ = np.array(x_in_)
    else:
        raise TypeError

    if isinstance(y_in_, np.ndarray):
        y_ = y_in_
    elif isinstance(y_in_, list):
        y_ = np.array(y_in_)
    else:
        raise TypeError

    for v_ in [x_, y_]:
        assert v_.dtype in [float, int, np.float32], v_.dtype

    x_mean = np.mean(x_)
    y_mean = np.mean(y_)
    x_var = np.sum(np.square(x_ - x_mean)) / (x_.size)
    y_var = np.sum(np.square(y_ - y_mean)) / (y_.size)
    xy_covar = np.dot((x_ - x_mean), (y_ - y_mean)) / x_.size
    ccc = (2 * xy_covar) / (x_var + y_var + np.square(x_mean - y_mean))

    return ccc


def calc_bootstrap_ci(theta_hat, theta_tilde_samples, alpha=0.05, flavor='percentile'):
    '''
    theta_hat : statistic estimate from the original sample
    theta_tilde : statistic estimate from a bootstrap sample
    '''
    import numpy as np

    percentiles = (alpha * 100. / 2., 100. - alpha * 100. / 2.)

    if flavor == 'percentile':
        q_low, q_high = np.percentile(theta_tilde_samples, percentiles, axis=0)
        ci_ = np.array([q_low, q_high])

    elif flavor == 'basic':
        '''
        1 - alpha = P(q_{alpha/2} - theta_hat <= theta_tilde - theta_hat <= q_{1-alpha/2} - theta_hat)
                  = P(2*theta_hat - q_{1-alpha/2} <= theta <= 2*theta_hat - q_{alpha/2})
        '''
        q_low, q_high = np.percentile(theta_tilde_samples, percentiles, axis=0)
        bias = np.mean(theta_tilde_samples) - theta_hat
        ci_ = np.array([2 * theta_hat - q_high, 2 * theta_hat - q_low])

    return ci_


def bootstrap_pe(x, alpha=0.05, bootstrap_samples=1000, estimator=None, flavor='percentile', rng=None):
    from sklearn.utils import resample
    import numpy as np
    # https://blog.methodsconsultants.com/posts/understanding-bootstrap-confidence-interval-output-from-the-r-boot-package/
    # https://www.erikdrysdale.com/bca_python/

    if rng is None:
        rng = np.random.default_rng()

    seeds = rng.integers(low=1, high=np.iinfo(np.int32).max, size=bootstrap_samples)

    if flavor == 'percentile':
        '''
        theta_hat : statistic estimate from the original sample
        theta_tilde : statistic estimate from a bootstrap sample
        '''
        assert x.shape[0] > 1
        assert x.size == x.shape[0]

        theta_hat = estimator(x)

        theta_tilde_samples = np.full((bootstrap_samples,), np.nan)
        for ii in np.arange(bootstrap_samples):
            theta_tilde_samples[ii] = estimator(resample(x, replace=True, n_samples=x.shape[0], random_state=seeds[ii]))

        pe_ = theta_hat
        ci_ = calc_bootstrap_ci(theta_hat, theta_tilde_samples, alpha=alpha, flavor=flavor)

    elif flavor == 'basic':
        '''
        theta_hat : statistic estimate from the original sample
        theta_tilde : statistic estimate from a bootstrap sample
        bias : E[theta_tilde] - theta_hat
        1 - alpha = P(q_{alpha/2} - theta_hat <= theta_tilde - theta_hat <= q_{1-alpha/2} - theta_hat)
                  = P(2*theta_hat - q_{1-alpha/2} <= theta <= 2*theta_hat - q_{alpha/2})
        '''
        assert x.shape[0] > 1
        assert x.size == x.shape[0]

        theta_hat = estimator(x)

        theta_tilde_samples = np.full((bootstrap_samples,), np.nan)
        for ii in np.arange(bootstrap_samples):
            theta_tilde_samples[ii] = estimator(resample(x, replace=True, n_samples=x.shape[0], random_state=seeds[ii]))

        bias = np.mean(theta_tilde_samples) - theta_hat

        pe_ = theta_hat
        ci_ = calc_bootstrap_ci(theta_hat, theta_tilde_samples, alpha=alpha, flavor=flavor)

    return pe_, ci_


def random_argmax(ar, axis=0, rng=None, warn=True):
    """  
    """
    import numpy as np

    if rng is None:
        rng = np.random.default_rng()

    arout = np.argmax(ar, axis=axis)
    multimax_ = np.sum((ar == np.max(ar, axis=axis, keepdims=True)), axis=axis) > 1
    multimaxidx = np.argwhere(multimax_)

    if np.sum(multimax_) > 0:
        if warn:
            print(f"MultiMax: {np.sum(multimax_)} / {arout.size}")
        full_shape = list(ar.shape)
        full_shape_reduced = full_shape.copy()
        reduced_shape = arout.shape
        for redax in np.array(axis).flatten():
            full_shape_reduced[redax] = None
        full_shape_reduced_check = [xx for xx in full_shape_reduced if xx is not None]
        assert np.array_equal(reduced_shape, full_shape_reduced_check)
        dim_mapping = np.zeros(len(reduced_shape), dtype=int)
        ii_red = -1
        for ii_full, red_dim_size in enumerate(full_shape_reduced):
            if red_dim_size is not None:
                ii_red += 1
                dim_mapping[ii_red] = ii_full

        idxer_template = [slice(None) for _ in range(len(full_shape))]
        for ii in range(multimaxidx.shape[0]):
            idxer = idxer_template.copy()
            for ii_multi in range(multimaxidx[ii].size):
                idxer[dim_mapping[ii_multi]] = multimaxidx[ii][ii_multi]

            rand_arg = rng.choice(np.argwhere(ar[tuple(idxer)] == np.max(ar[tuple(idxer)])).flatten())
            if np.sum(arout.shape) > 0:
                arout[tuple(multimaxidx[ii].tolist())] = rand_arg
            else:
                arout = rand_arg

    return arout


def calc_confusion_matrix(dfin, categories, bypass_assertions=False):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score, precision_recall_fscore_support, multilabel_confusion_matrix, confusion_matrix
    from sklearn.metrics import accuracy_score

    y_true = dfin.loc[:, 'veridical_outcome'].to_numpy()
    y_pred = dfin.loc[:, 'pred_outcome'].to_numpy()

    confusion_n = pd.DataFrame(-1 * np.ones((len(categories), len(categories))), index=categories, columns=categories, dtype=int)  # veridical is row index, judgment is columns
    confusion_f = pd.DataFrame(-1 * np.ones((len(categories), len(categories))), index=categories, columns=categories)  # veridical is row index, judgment is columns
    confusion_n.loc[:, :] = confusion_matrix(y_true, y_pred, labels=categories)
    confusion_f.loc[:, :] = confusion_matrix(y_true, y_pred, labels=categories, normalize='all')

    assert np.sum(confusion_n.to_numpy().flatten()) == dfin.shape[0], np.sum(confusion_n.to_numpy().flatten())
    if not bypass_assertions:
        assert np.isclose(np.sum(confusion_f.to_numpy().flatten()), 1), np.sum(confusion_f.to_numpy().flatten())

    precision, recall, fbeta_score, support = precision_recall_fscore_support(y_true, y_pred, beta=1.0, average=None, labels=categories, warn_for=('precision', 'recall', 'f-score'), zero_division=0)

    precision_macro, recall_macro, fbeta_score_macro, support_macro = precision_recall_fscore_support(y_true, y_pred, beta=1.0, average='macro', labels=categories, warn_for=('precision', 'recall', 'f-score'), zero_division=0)

    c_R = 0.25
    c_Q = 0.75
    c_r = 0.25
    c_q = 0.75
    c_tp_ = c_R * c_r
    c_tn_ = c_Q * c_q
    c_fp_ = c_r * c_Q
    c_fn_ = c_q * c_R
    uniform_chance_f1 = f1_score(np.repeat(categories, len(categories)), np.tile(categories, len(categories)), average=None, labels=categories, zero_division='warn')

    def calc_f1_manual_(con_mat_f, con_mat_n, outcomes, bypass_checks=False):

        fbeta_score_, matthews_corr_coef_, sensitivity_, precision_, domain_errors_ = dict(), dict(), dict(), dict(), dict()

        fbeta_score_micro_tp = list()
        fbeta_score_micro_tpfp = list()

        for i_outcome, outcome in enumerate(outcomes):

            tp_ = con_mat_n.loc[outcome, outcome]
            fp_ = con_mat_n.loc[:, outcome].sum() - con_mat_n.loc[outcome, outcome]
            tn_ = con_mat_n.sum().sum() - con_mat_n.loc[:, outcome].sum() - con_mat_n.loc[outcome, :].sum() + con_mat_n.loc[outcome, outcome]
            fn_ = con_mat_n.loc[outcome, :].sum() - con_mat_n.loc[outcome, outcome]

            fbeta_score_micro_tp.append(tp_)
            fbeta_score_micro_tpfp.append(tp_ + fp_)

            sensitivity_[outcome] = tp_ / (tp_ + fn_)
            if con_mat_n.loc[:, outcome].sum() > 0:
                precision_[outcome] = tp_ / (tp_ + fp_)
            else:
                precision_[outcome] = np.nan
            fbeta_score_[outcome] = (2 * tp_) / (2 * tp_ + fp_ + fn_)

            if tp_ + fp_ + fn_ == 0:
                domain_errors_[outcome] = dict(precision=precision_[outcome], sensitivity=sensitivity_[outcome])
                print(f"domain error: {outcome}")
                fbeta_score_[outcome] = 0.0

            sensitivity_tmp = con_mat_f.loc[outcome, outcome] / np.sum(con_mat_f.loc[outcome, :]) if np.sum(con_mat_f.loc[outcome, :]) > 0 else np.nan
            precision_tmp = con_mat_f.loc[outcome, outcome] / np.sum(con_mat_f.loc[:, outcome]) if np.sum(con_mat_f.loc[:, outcome]) > 0 else np.nan
            if not bypass_checks:
                assert np.isclose(sensitivity_[outcome], sensitivity_tmp), f"sensitivity_[outcome]: {sensitivity_[outcome]} vs sensitivity_tmp: {sensitivity_tmp}"
                if (not np.isnan(precision_tmp)) or (not np.isnan(precision_[outcome])):
                    assert np.isclose(precision_[outcome], precision_tmp), f"{precision_tmp}, {precision_}"
                assert tp_ + fp_ + fn_ > 0
            tp_f_ = con_mat_f.loc[outcome, outcome]
            fp_f_ = con_mat_f.loc[:, outcome].sum() - con_mat_f.loc[outcome, outcome]
            tn_f_ = con_mat_f.sum().sum() - con_mat_f.loc[:, outcome].sum() - con_mat_f.loc[outcome, :].sum() + con_mat_f.loc[outcome, outcome]
            fn_f_ = con_mat_f.loc[outcome, :].sum() - con_mat_f.loc[outcome, outcome]
            if not bypass_checks:
                assert np.isclose(fbeta_score_[outcome], (2 * tp_f_) / (2 * tp_f_ + fp_f_ + fn_f_))

        fbeta_score_micro_ = np.sum(fbeta_score_micro_tp) / np.sum(fbeta_score_micro_tpfp)
        fbeta_score_macro_ = np.mean(list(fbeta_score_.values()))

        return fbeta_score_, fbeta_score_macro_, fbeta_score_micro_, None, domain_errors_

    fbeta_score_manual, fbeta_score_macro_manual, fbeta_score_micro_manual, mcc_manual, domain_errors_full = calc_f1_manual_(confusion_f, confusion_n, categories, bypass_checks=bypass_assertions)

    rc_true = np.sum(confusion_f, axis=1)
    rc_pred = np.sum(confusion_f, axis=0)
    chance_confusion_f = confusion_f.copy()
    chance_confusion_f.loc[:, :] = np.outer(rc_true, rc_pred)

    chance_f1, chance_f1_macro, chance_f1_micro, chance_mcc, domain_errors_chance = calc_f1_manual_(chance_confusion_f, chance_confusion_f, categories, bypass_checks=bypass_assertions)

    domain_errors = dict()
    if domain_errors_full:
        domain_errors['f'] = domain_errors_full
    if domain_errors_chance:
        domain_errors['chance'] = domain_errors_chance

    def calc_se_(p, n): return np.sqrt(p * (1 - p) / n)
    n_stim_per_outcome = 22
    unbiased_hitrate_se = dict()
    for i_outcome, outcome in enumerate(categories):
        unbiased_hitrate_se[outcome] = calc_se_(fbeta_score[i_outcome], n_stim_per_outcome)

    return dict(
        fbeta_score=dict(zip(categories, fbeta_score)),
        _fbeta_score_=fbeta_score_manual,
        precision=dict(zip(categories, precision)),
        recall=dict(zip(categories, recall)),  # sensitivity
        support=dict(zip(categories, support)),
        ###
        fbeta_chance=chance_f1,
        fbeta_uniform_chance=uniform_chance_f1[0],
        #####
        accuracy=accuracy_score(y_true, y_pred, normalize=True),
        ###
        fbeta_score_macro=fbeta_score_macro,
        _fbeta_score_macro_=fbeta_score_macro_manual,
        fbeta_score_macro_chance=chance_f1_macro,
        #####
        confusion=confusion_n,
        confusion_frequency=confusion_f,
        #####
        se=unbiased_hitrate_se,
        #####
        hits=dict([(outcome, confusion_n.loc[outcome, outcome]) for outcome in categories]),
        domain_errors=domain_errors,
    )


def check_assumptions(deltas_):
    import numpy as np
    import scipy.stats as stats

    assumptions = dict()

    ### no significant outliers in the differences

    ### distribution of the differences in the dependent variable between the two related groups should be approximately normally distributed

    ### differences are symmetrically distributed

    #################
    alpha = 0.05

    ### Shapiro-Wilk normality test
    stat, p = stats.shapiro(deltas_)

    assumptions['shapiro'] = dict(
        test='Shapiro-Wilk',
        stat=stat,
        p=p,
        result=f'Sample looks Gaussian (fail to reject H0 at {alpha})' if p > alpha else f'Sample does not look Gaussian (reject H0 at {alpha})'
    )

    ###################

    # The D’Agostino’s K^2 test calculates summary statistics from the data, namely kurtosis and skewness, to determine if the data distribution departs from the normal distribution, named for Ralph D’Agostino.

    # Skew is a quantification of how much a distribution is pushed left or right, a measure of asymmetry in the distribution.
    # Kurtosis quantifies how much of the distribution is in the tail. It is a simple and commonly used statistical test for normality.

    stat, p = stats.normaltest(deltas_)

    assumptions['normaltest'] = dict(
        test="D'Agostino's K^2",
        stat=stat,
        p=p,
        result=f'Sample looks Gaussian (fail to reject H0 at {alpha})' if p > alpha else f'Sample does not look Gaussian (reject H0 at {alpha})'
    )

    ##################

    stat = stats.anderson(deltas_)

    res_ = list()
    for i in range(len(stat.critical_values)):
        sl, cv = stat.significance_level[i], stat.critical_values[i]
        if stat.statistic < stat.critical_values[i]:
            res_.append('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
        else:
            res_.append('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))

    assumptions['anderson'] = dict(
        test='Anderson-Darling',
        stat=stat,
        result=res_
    )

    ############

    w_sum, p_val = stats.wilcoxon(deltas_ - np.mean(deltas_))

    assumptions['wilcoxon'] = dict(
        test="Wilcoxon Signed-Ranks",
        stat=w_sum,
        p=p_val,
        result=f'Sample looks symmetric (fail to reject H0 at {alpha})' if p > alpha else f'Sample does not look symmetric (reject H0 at {alpha})'
    )

    return assumptions


def wilcoxon(data1, data2):
    import numpy as np
    import scipy.stats as stats

    # for "two-sided"
    # prob = 2. * distributions.norm.sf(abs(z))
    # ->
    # +/- distributions.norm.isf(prob/2.)

    if not isinstance(data2, np.ndarray):
        data2 = np.array(data2)

    nans_ = np.isnan(data1) + np.isnan(data2) if data2.size > 1 else np.isnan(data1)
    n_nan = np.sum(nans_)
    if n_nan > 0:
        print(f"dropping {n_nan} NAN from data, at {np.where(nans_)}")
        data1 = data1[np.logical_not(nans_)]

        if data2.size > 1:
            data2 = data2[np.logical_not(nans_)]

    deltas_ = data1 - data2

    assumptions_ = check_assumptions(deltas_)

    if assumptions_['wilcoxon']['p'] < 0.05:
        print("\n\nWARNING:: Wilcoxon Signed-Ranks test failed symmetry assumption!\n\n")

    median1_ = np.median(data1)
    median2_ = np.median(data2)
    deltas_median_ = np.median(deltas_)

    ### stats.wilcoxon(A - B) is same as stats.wilcoxon(A, y=B)
    w_sum, p_val = stats.wilcoxon(
        deltas_,
        zero_method='wilcox',
        correction=False,
        alternative='two-sided',
        mode='auto'
    )

    z_statistic = np.sign(deltas_median_) * stats.norm.isf(p_val / 2.)

    effect_size = np.abs(z_statistic) / np.sqrt(deltas_.size)  # assuming only 1 group. If data1 and data2 are from different subjects, N should be 2 * deltas_.size

    #################

    # format_precision = 2 if p_val >= 0.01 else 3
    format_precision = 3

    z_stat_round = round(z_statistic, format_precision)
    z_str = f"$z = {z_stat_round:0.2f}$" if format_precision == 2 else f"$z = {z_stat_round:0.3f}$"

    p_str = f"$p = {p_val:0.2f}$" if format_precision == 2 else f"$p = {p_val:0.3f}$"
    if p_val < 0.001:
        p_str = f"$p < 0.001$"

    effect_size_round = round(effect_size, format_precision)
    r_str = f"$r = {effect_size_round:0.2f}$" if format_precision == 2 else f"$r = {effect_size_round:0.3f}$"

    n_str = f"$n = {deltas_.size}$"

    desc_long = f"two-sided Wilcoxon signed-rank test, {z_str}, {p_str}, {r_str}, {n_str}"
    if n_nan > 0:
        desc_long = desc_long + f". {n_nan} excluded due to missing data."
    #################

    # "Wilcoxon signed-rank test, z = 1.035, P = 0.301, n = 9"
    # small size	medium size	large size
    # abs(r)	0.1	0.3	0.5
    return dict(
        median1=median1_,
        median2=median2_,
        median_delta=deltas_median_,
        z=z_statistic,
        p=p_val,
        r=effect_size,  # effect size
        n=deltas_.size,
        w=w_sum,
        z_str=z_str,
        p_str=p_str,
        r_str=r_str,
        desc_long=desc_long
    )
