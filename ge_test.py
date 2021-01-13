import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import sys
import csv
import os
import matplotlib.pyplot as plt
import tqdm
import numpy as np


import ge_data
import ge_nn
import ge_loss


def ge_test_fun(data, n_device, batchsize, n_targets, model_path):
    #データロード
    test_set = ge_data.ge_test_dataset(data)
    test_loader = DataLoader(test_set, batch_size = batchsize, shuffle=True, num_workers=16)
    device_str = "cuda:{}".format(n_device)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print("used device : ", device)
    #損失関数
    loss_fun = nn.PoissonNLLLoss()
    #モデルの読み込み
    test_model = ge_nn.Net(n_targets=n_targets)
    test_model.to(device)
    test_model.load_state_dict(torch.load(model_path))
    test_model.eval()
    #損失の記録
    test_loss = []
    test_score = []
    with torch.no_grad():
        for test_in, test_out in test_loader:
            #モデル入力
            test_in,  test_out = test_in.to(device), test_out.to(device)
            out = test_model(test_in)
            #損失計算
            loss = loss_fun(out, test_out)
            test_loss.append(loss.item())
            #score計算
            score = ge_loss.log_r2_score(out, test_out)
            test_score.append(score)
        avr_test_loss = np.average(test_loss)
        avr_test_score = np.average(test_score)
    print('test data loss:{}, test r2 score:'.format(avr_test_loss))
    with open('train_log.txt', 'a') as f:
        f.write('test data loss:{}, test r2 score:{}'.format(avr_test_loss, avr_test_score))


def r2_score(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
    y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)

    if _num_samples(y_pred) < 2:
        msg = "R^2 score is not well-defined with less than two samples."
        warnings.warn(msg, UndefinedMetricWarning)
        return float('nan')

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
        weight = sample_weight[:, np.newaxis]
    else:
        weight = 1.

    numerator = (weight * (y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    denominator = (weight * (y_true - np.average(
        y_true, axis=0, weights=sample_weight)) ** 2).sum(axis=0, dtype=np.float64)
    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = np.ones([y_true.shape[1]])
    output_scores[valid_score] = 1 - (numerator[valid_score] / denominator[valid_score])
    # arbitrary set to zero to avoid -inf scores, having a constant
    # y_true is not interesting for scoring a regression anyway
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.
    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            # return scores individually
            return output_scores
        elif multioutput == 'uniform_average':
            # passing None as weights results is uniform mean
            avg_weights = None
        elif multioutput == 'variance_weighted':
            avg_weights = denominator
            # avoid fail on constant y or one-element arrays
            if not np.any(nonzero_denominator):
                if not np.any(nonzero_numerator):
                    return 1.0
                else:
                    return 0.0
    else:
        avg_weights = multioutput

    return np.average(output_scores, weights=avg_weights)


