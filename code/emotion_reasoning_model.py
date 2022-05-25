#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class MultivariateKDE():

    def __init__(self, bandwidth):
        self.bandwidth = bandwidth
        self.sigma = None
        self.emopredictions_df = None
        self.stimid_by_outcome = None
        self.dists = dict()
        self.support = None

    def make_kde(self, empir_emopredictions_df, stimid_by_outcome):
        import numpy as np
        from copy import deepcopy
        import torch
        from torch.distributions import Normal

        self.emopredictions_df = empir_emopredictions_df.copy()
        self.stimid_by_outcome = deepcopy(stimid_by_outcome)

        n_emotions = self.emopredictions_df.loc[:, 'emotionIntensities'].shape[1]
        if np.isscalar(self.bandwidth):
            self.sigma = torch.ones(n_emotions) * self.bandwidth
        else:
            assert len(self.bandwidth) == n_emotions
            self.sigma = torch.Tensor(self.bandwidth)

        emotion_judgments = torch.Tensor(self.emopredictions_df.loc[:, 'emotionIntensities'].to_numpy())
        stimid_idx = self.emopredictions_df.loc[:, ('stimulus', 'stimid')].to_numpy()

        for outcome in self.stimid_by_outcome:
            self.dists[outcome] = dict()
            for stimid in self.stimid_by_outcome[outcome]:
                self.dists[outcome][stimid] = Normal(emotion_judgments[stimid_idx == stimid, :], self.sigma[None, :])

    def score(self, emp_resp):
        import numpy as np
        import torch

        ### score takes a single observation
        assert emp_resp.shape[0] == emp_resp.size

        emp_resp_tensor = torch.Tensor(emp_resp.to_numpy())
        score = dict()
        for outcome, stimid_list in self.stimid_by_outcome.items():
            scores = torch.full([len(stimid_list)], np.nan, dtype=torch.double)
            for i_stimid, stimid in enumerate(stimid_list):
                score_ = self.dists[outcome][stimid].log_prob(emp_resp_tensor[None, :]).double()  # dims: [n_obs of e|C=stimid x n_emotions]
                score_1 = score_.sum(dim=1)  # product over emotion dimension

                ### Average over component distributions of the KDE (i.e. score the e|X=stimid,resp=i vector in each e|C=stimid and average the scores to yield a mean score for this e|X=stimid,resp=i in P(e|C=stimid,resp=j). Loop over stimid to calculate the probability of the single observation e|X=stimid,resp=i in the P(e|C) for every stimid.
                scores[i_stimid] = score_1.logsumexp(dim=0) - np.log(score_1.size(0), dtype=np.float64)

            ### Average over the stimid within each outcome. Yields the score of this single observation, the emotion vector e|X=stimid,resp=i, for an outcome. I.e. score of e|X=stimid,resp=i in P(e|C=a_1,a_2)
            score[outcome] = (scores.logsumexp(dim=0) - np.log(scores.size(0), dtype=np.float64)).item()

        return score


class BayesianBeliefUpdating():

    def __init__(self, norm_percentile=None):
        if norm_percentile is None:
            norm_percentile = ['percentile-norm-X', 'percentile-norm-C'][0]
        self.norm_percentile = norm_percentile

    def prep_data(self, emoPredictionsContext=None, emoAttributionsExpressions=None, empiricalOutcomeJudgments=None, stimid_by_outcome=None, stim_info=None):

        import numpy as np
        import pandas as pd

        outcomes = list(stimid_by_outcome.keys())

        emotions = emoPredictionsContext['emotionIntensities'].columns.to_list()
        assert np.array_equal(emotions, emoAttributionsExpressions['emotionIntensities'].columns.to_list())

        stimidorder = np.concatenate(list(stimid_by_outcome.values()))

        stimids_present = sorted(emoPredictionsContext['stimulus']['stimid'].unique().to_list())
        assert np.array_equal(stimids_present, sorted(emoAttributionsExpressions['stimulus']['stimid'].unique().to_list()))
        assert np.array_equal(stimids_present, sorted(stimidorder.tolist()))

        ############
        ## calculate empirical p(a|x) (the data to be explained)
        ############
        temp_empr_resp_per_stim = dict()
        for stimid in stimidorder:
            stimdf_ = empiricalOutcomeJudgments.loc[empiricalOutcomeJudgments['stimulus'] == stimid, :]
            temp_empr_resp_per_stim[stimid] = dict()
            for i_outcome_resp, outcome_resp in enumerate(outcomes):
                temp_empr_resp_per_stim[stimid][outcome_resp] = np.sum(stimdf_['pred_outcome'] == outcome_resp) / stimdf_.shape[0]
            temp_empr_resp_per_stim[stimid]['veridical'] = stim_info[stimid]['outcome']
        empir_resp_freq = pd.DataFrame(temp_empr_resp_per_stim).T.astype(dict(zip(outcomes, [float] * len(outcomes))))

        ############
        ## estimate hyperprior p(a) from the BTS judgments
        ############
        pr = dict(zip(outcomes, [0, 0, 0, 0]))
        for ridx_, row_ in empiricalOutcomeJudgments.iterrows():
            a1 = 'D' if row_['BTS_a1'] < 0 else 'C'
            a2 = 'D' if row_['BTS_a2'] < 0 else 'C'
            pr[f'{a1}{a2}'] += 1
        prob_outcome_hyperprior = pd.Series(np.array(list(pr.values())) / empiricalOutcomeJudgments.shape[0], list(pr.keys()))

        self.emo_predictions = emoPredictionsContext.copy()
        self.emo_attributions = emoAttributionsExpressions.copy()
        self.prob_outcome_hyperprior = prob_outcome_hyperprior
        self.empir_outcome_resp_freq = empir_resp_freq
        ####
        self.stim_info = stim_info
        self.stimdid_by_outcome = stimid_by_outcome
        self.stimid_order = stimidorder
        self.emotions = emotions

    def fit_multivariate_kde(self, bw=None):

        import numpy as np
        import pandas as pd
        from utils import scotts_rule, weighted_descriptive_stats_byoutcome

        assert bw >= 0

        stimid_order = self.stimid_order
        outcomes = list(self.stimdid_by_outcome.keys())

        if bw == 0:  # Use Scott's rule

            torch_bandwidth_C = scotts_rule(self.emo_predictions).loc[self.emotions].to_numpy()
            torch_bandwidth_X = scotts_rule(self.emo_attributions).loc[self.emotions].to_numpy()

            scipy_bandwidth_C = ['scott' for emotion in self.emotions]
            scipy_bandwidth_X = ['scott' for emotion in self.emotions]

        elif bw > 0:

            torch_bandwidth_C = weighted_descriptive_stats_byoutcome(self.emo_predictions)['sd_unweighted'].loc[self.emotions].to_numpy() * bw
            torch_bandwidth_X = weighted_descriptive_stats_byoutcome(self.emo_attributions)['sd_unweighted'].loc[self.emotions].to_numpy() * bw

            ### scipy gaussian_kde scales the bandwidth by the standard deviation of the data, so bandwidth is divided by the SD here ###
            scipy_bandwidth_C = np.zeros_like(torch_bandwidth_C)
            scipy_bandwidth_X = np.zeros_like(torch_bandwidth_X)
            for i_emotion, emotion in enumerate(self.emotions):
                scipy_bandwidth_C[i_emotion] = torch_bandwidth_C[i_emotion] / np.std(self.emo_predictions.loc[:, ('emotionIntensities', emotion)].to_numpy())
                scipy_bandwidth_X[i_emotion] = torch_bandwidth_X[i_emotion] / np.std(self.emo_attributions.loc[:, ('emotionIntensities', emotion)].to_numpy())

        else:
            raise Exception("bandwidth not understood")

        emp_emo_C_df, emp_emo_X_df, support_ = normalize_emotion_ratings_across_conditions(self.emo_predictions, self.emo_attributions, emotion_subset=self.emotions, sigma_C=scipy_bandwidth_C, sigma_X=scipy_bandwidth_X, norm_percentile=self.norm_percentile)

        if self.norm_percentile == 'percentile-norm-X':
            ### don't recalculate sigma_C if self.emo_predictions is unchanged ###
            sigma_C = torch_bandwidth_C.copy()
        else:
            ### if the C dist has changed, recalculate sigma_C ###
            sigma_C = weighted_descriptive_stats_byoutcome(emp_emo_C_df)['sd_unweighted'].loc[self.emotions].to_numpy() * bw

        mkde = MultivariateKDE(sigma_C)
        mkde.make_kde(emp_emo_C_df, self.stimdid_by_outcome)

        p_a1a2_give_x, lp_ex_a1a2 = calc_probability_outcome_given_emotions(emp_emo_X_df, self.prob_outcome_hyperprior, mkde.score, outcomes)
        p_a1a2_give_x['veridical'] = pd.Series({stimid: self.stim_info[stimid]['outcome'] for stimid in p_a1a2_give_x.index.to_list()})

        simulated_resp_freq = p_a1a2_give_x.loc[stimid_order, :].copy()

        ####

        np.testing.assert_array_equal(self.empir_outcome_resp_freq.index, simulated_resp_freq.index)
        np.testing.assert_array_equal(self.empir_outcome_resp_freq['veridical'].to_numpy(), simulated_resp_freq['veridical'].to_numpy())

        return simulated_resp_freq

    def score(self, bw=None):
        from utils import concordance_correlation_coefficient

        outcomes = list(self.stimdid_by_outcome.keys())

        simulated_resp_freq = self.fit_multivariate_kde(bw=bw)

        y_empir_ = self.empir_outcome_resp_freq.loc[:, outcomes].to_numpy().flatten()
        y_model_ = simulated_resp_freq.loc[:, outcomes].to_numpy().flatten()

        concorr = concordance_correlation_coefficient(y_empir_, y_model_)

        return concorr

    def full_results(self, bw=None):
        import numpy as np
        from scipy.stats import pearsonr
        from utils import concordance_correlation_coefficient, coefficient_of_determination

        outcomes = list(self.stimdid_by_outcome.keys())

        simulated_resp_freq = self.fit_multivariate_kde(bw=bw)

        y_empir_ = self.empir_outcome_resp_freq.loc[:, outcomes].to_numpy().flatten()
        y_model_ = simulated_resp_freq.loc[:, outcomes].to_numpy().flatten()

        pcorr = pearsonr(y_empir_, y_model_)[0]
        concorr = concordance_correlation_coefficient(y_empir_, y_model_)
        coefdet = coefficient_of_determination(y=y_empir_, yhat=y_model_)
        mse = np.mean(np.square(y_empir_ - y_model_))

        print(f"bw {bw:0.4} :: mse={mse:0.4}, r={pcorr:0.4}, ccc={concorr:0.4}")

        return dict(empir_resp_freq=self.empir_outcome_resp_freq, simulated_resp_freq=simulated_resp_freq, emotion_subset=self.emotions, P_c_acrossStim=self.prob_outcome_hyperprior, norm_percentile=self.norm_percentile, bandwidth=bw, pearsonr=pcorr, concordance=concorr, coeff_determination=coefdet, mse=mse, rmse=np.sqrt(mse))


def calc_probability_outcome_given_emotions(empir_emoattributions_df, prob_outcome_hyperprior, emopredictions_dist_score_fn, outcomes):
    import numpy as np
    import pandas as pd
    from scipy.special import logsumexp

    p_a1a2_give_x_ = {outcome: dict() for outcome in outcomes}
    lp_ex_a1a2_dict = dict()

    lp_a1a2_acrossStim = np.array([np.log(prob_outcome_hyperprior[outcome], dtype=np.float64) for outcome in outcomes], dtype=np.float64)

    for stimid in sorted(empir_emoattributions_df['stimulus']['stimid'].unique()):

        ### all empirical responses for a given video in the X condition -- J(e|X=stimid)
        empirical_attributions_X = empir_emoattributions_df.loc[empir_emoattributions_df['stimulus']['stimid'] == stimid, 'emotionIntensities']

        n_resp = empirical_attributions_X.shape[0]
        lp_ex_give_x = -1 * np.log(n_resp, dtype=np.float64)

        lp_a1a2_ex_give_x = {outcome: np.full(n_resp, np.nan, dtype=np.float64) for outcome in outcomes}

        lp_ex_a1a2_dict[stimid] = np.full([n_resp, len(outcomes)], np.nan, dtype=np.float64)
        for emp_resp_idx in range(n_resp):

            ### \vec{e} -- \vec{e}_i | X = stimid
            emp_resp = empirical_attributions_X.iloc[emp_resp_idx, :]

            ### P( \vec{e} | a1, a2 )
            ### returns the score the a single observation, e|X=stimid,response=i, in each of the 4 outcomes.
            ### The score is calculated as the probability of observing the \vec{e}|X=stimid,response=i vector in each \vec{e}|C=stimid,resp=j.
            lp_ex_give_a1a2_dict = emopredictions_dist_score_fn(emp_resp)
            lp_ex_give_a1a2 = np.array([lp_ex_give_a1a2_dict[outcome] for outcome in outcomes], dtype=np.float64)

            ### P( \vec{e} | a1, a2 ) * P( a1, a2 )
            lp_ex_a1a2 = lp_ex_give_a1a2 + lp_a1a2_acrossStim
            lp_ex_a1a2_dict[stimid][emp_resp_idx, :] = lp_ex_a1a2

            ### P( \vec{e} ) = \sum_{a_1, a_2}[ P( a1, a2 ) * P( \vec{e} | a1, a2 ) ]
            lp_ex = logsumexp(lp_ex_a1a2)

            ### P(ex|a1a2) * P(a1a2) * P(ex|x) / p(ex)
            for i_outcome, outcome in enumerate(outcomes):
                lp_a1a2_ex_give_x[outcome][emp_resp_idx] = lp_ex_a1a2[i_outcome] + lp_ex_give_x - lp_ex

        for outcome in outcomes:
            ### P( a1, a2 | x) = \sum_{ \vec{e} }[ P( \vec{e} | a1, a2 ) * P( a1, a2 ) * P( \vec{e} | x ) / P( \vec{e} ) ]
            p_a1a2_give_x_[outcome][stimid] = np.exp(logsumexp(lp_a1a2_ex_give_x[outcome]), dtype=np.float64)

    p_a1a2_give_x = pd.DataFrame(p_a1a2_give_x_, dtype=np.float64)
    assert np.all(np.isclose(np.sum(p_a1a2_give_x.to_numpy(), axis=1, dtype=np.float64), 1.0))

    p_a1a2_give_x['veridical'] = 'XX'

    ### p_a1a2_give_x # multiplied by p(e|x) --> normalized within stimulus
    return p_a1a2_give_x, lp_ex_a1a2_dict


def percentile_normalisation(resp_series_source, mapping_source_to_dest_, support__):
    import numpy as np
    resp_series_source_transformed = resp_series_source.astype(float).copy()
    resp_series_source_transformed.loc[:] = 0.0
    for emotion in resp_series_source.index.to_list():
        xval_source = resp_series_source.loc[emotion]
        assert np.min(np.absolute(support__ - xval_source)) < np.absolute(support__[-3] - support__[-4]), f"xval_source = {xval_source}, np.min(np.absolute(support__ - xval_source)) = {np.min(np.absolute(support__ - xval_source))}, np.absolute(support__[-3] - support__[-4]) = {np.absolute(support__[-3] - support__[-4])}"
        xval_source_idx = np.argmin(np.absolute(support__ - xval_source))
        xval_dest_idx = mapping_source_to_dest_[emotion][xval_source_idx]
        xval_dest = support__[xval_dest_idx]
        resp_series_source_transformed.loc[emotion] = xval_dest
    return resp_series_source_transformed


def normalize_emotion_ratings_across_conditions(emp_emo_C_df_in, emp_emo_X_df_in, emotion_subset=None, sigma_C=None, sigma_X=None, norm_percentile=None):
    import numpy as np
    from scipy.stats import gaussian_kde

    emp_emo_C_df = emp_emo_C_df_in.copy()
    emp_emo_X_df = emp_emo_X_df_in.copy()
    support_ = np.linspace(-1, 2, (49 * 12 - 11))

    if emotion_subset is None:
        emotion_subset = emp_emo_C_df['emotionIntensities'].columns.to_list()
        assert len(emotion_subset) == 20

    if norm_percentile == 'percentile-norm-X':

        mapping_X_to_C = dict()
        emp_emo_Xuntransformed_df = emp_emo_X_df.copy()

        weights_ = None
        for i_emotion, emotion in enumerate(emotion_subset):
            obs_C_ = emp_emo_C_df.loc[:, ('emotionIntensities', emotion)]
            obs_X_ = emp_emo_Xuntransformed_df.loc[:, ('emotionIntensities', emotion)]

            kde_C_ = gaussian_kde(obs_C_, weights=weights_, bw_method=sigma_C[i_emotion])
            kde_X_ = gaussian_kde(obs_X_, weights=weights_, bw_method=sigma_X[i_emotion])

            cdf_C_ = np.array([kde_C_.integrate_box_1d(-1000, xx) for xx in support_])
            cdf_X_ = np.array([kde_X_.integrate_box_1d(-1000, xx) for xx in support_])

            ### when e|X value is xx (where xx = support_[ii]), that corresponds to a e|C value of mapping_X_to_C[emotion][ii].
            mapping_X_to_C[emotion] = np.full((support_.shape[0]), 0, dtype=int)
            for xval_X_idx in range(support_.shape[0]):
                xval_X = support_[xval_X_idx]
                cdf_X_at_xval = cdf_X_[xval_X_idx]

                mapping_X_to_C[emotion][xval_X_idx] = np.argmin(np.absolute(cdf_C_ - cdf_X_at_xval))

        emp_X_mat = emp_emo_Xuntransformed_df.loc[:, 'emotionIntensities'].astype(float).to_numpy()
        emp_X_transmormed_mat = np.full_like(emp_X_mat, np.nan, dtype=float)
        for i_resp in range(emp_X_mat.shape[0]):
            resp_series_X_ = emp_emo_Xuntransformed_df.iloc[i_resp].loc['emotionIntensities'].astype(float)
            assert np.allclose(resp_series_X_.to_numpy(), emp_X_mat[i_resp, :])
            emp_X_transmormed_mat[i_resp, :] = percentile_normalisation(resp_series_X_, mapping_X_to_C, support_)
        emp_emo_X_df.loc[:, 'emotionIntensities'] = emp_X_transmormed_mat

    elif norm_percentile == 'percentile-norm-C':

        mapping_C_to_X = dict()
        mapping_C_to_X_debug = dict()
        emp_emo_Cuntransformed_df = emp_emo_C_df.copy()

        weights_ = None
        for i_emotion, emotion in enumerate(emotion_subset):
            obs_C_ = emp_emo_Cuntransformed_df.loc[:, ('emotionIntensities', emotion)]
            obs_X_ = emp_emo_X_df.loc[:, ('emotionIntensities', emotion)]
            kde_C_ = gaussian_kde(obs_C_, weights=weights_, bw_method=sigma_C[i_emotion])
            kde_X_ = gaussian_kde(obs_X_, weights=weights_, bw_method=sigma_X[i_emotion])

            cdf_C_ = np.array([kde_C_.integrate_box_1d(-1000, xx) for xx in support_])
            cdf_X_ = np.array([kde_X_.integrate_box_1d(-1000, xx) for xx in support_])

            ### when e|C value is xx (where xx = support_[xval_C_idx]), that corresponds to a e|X value of mapping_C_to_X[emotion][xval_C_idx].
            mapping_C_to_X[emotion] = np.full((support_.shape[0]), 0, dtype=int)
            mapping_C_to_X_debug[emotion] = list()
            for xval_C_idx in range(support_.shape[0]):
                xval_C = support_[xval_C_idx]
                cdf_C_at_xval = cdf_C_[xval_C_idx]

                ### xval_C -> xval_X; xval_C = support_[xval_C_idx], xval_X_idx = mapping_C_to_X[emotion][xval_C_idx]
                mapping_C_to_X[emotion][xval_C_idx] = np.argmin(np.absolute(cdf_X_ - cdf_C_at_xval))

                xval_X_idx = mapping_C_to_X[emotion][xval_C_idx]
                xval_X = support_[xval_X_idx]
                cdf_X_at_xval = cdf_X_[xval_X_idx]
                mapping_C_to_X_debug[emotion].append(dict(xval_C_idx=xval_C_idx, xval_C=xval_C, cdf_C_at_xval=cdf_C_at_xval, xval_X_idx=xval_X_idx, xval_X=xval_X, cdf_X_at_xval=cdf_X_at_xval))

        emp_C_mat = emp_emo_Cuntransformed_df.loc[:, 'emotionIntensities'].astype(float).to_numpy()
        emp_C_transmormed_mat = np.full_like(emp_C_mat, np.nan, dtype=float)
        for i_resp in range(emp_C_mat.shape[0]):
            resp_series_C_ = emp_emo_Cuntransformed_df.iloc[i_resp].loc['emotionIntensities'].astype(float)
            assert np.allclose(resp_series_C_.to_numpy(), emp_C_mat[i_resp, :])
            emp_C_transmormed_mat[i_resp, :] = percentile_normalisation(resp_series_C_, mapping_C_to_X, support_)
        emp_emo_C_df.loc[:, 'emotionIntensities'] = emp_C_transmormed_mat

    else:
        raise NameError('norm_percentile not recognized:: >>{norm_percentile}<<')

    return emp_emo_C_df, emp_emo_X_df, support_


def multivariate_outcome_recovery(empiricalEmotionPredictions=None, empiricalEmotionAttributions=None, empiricalOutcomeJudgments=None, stimid_by_outcome=None, stim_info=None, bw_list=None):

    import numpy as np

    bbu_model = BayesianBeliefUpdating()

    bbu_model.prep_data(emoPredictionsContext=empiricalEmotionPredictions, emoAttributionsExpressions=empiricalEmotionAttributions, empiricalOutcomeJudgments=empiricalOutcomeJudgments, stimid_by_outcome=stimid_by_outcome, stim_info=stim_info)

    scores = list()
    for bw in bw_list:
        scores.append(bbu_model.score(bw))

    bw_best = bw_list[np.argmax(scores)]
    ### make sure bandwidth range covers top performance value
    if len(bw_list) > 1 and bw_best > 0:
        assert bw_best < np.max(bw_list) and bw_best > np.min([bw for bw in bw_list if bw > 0])

    return bbu_model.full_results(bw_best)
