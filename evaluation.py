import argparse
import collections

import numpy as np
import pandas as pd
from sklearn import metrics


def get_y_true(task_name):
    true_data_file = "data/semeval2014/bert-single/service/test.csv"  # change the test_data file

    df = pd.read_csv(true_data_file, sep='\t', header=None).values
    y_true = []

    for i in range(len(df)):
        label = df[i][1]
        assert label in ['positive', 'neutral', 'negative', 'conflict', 'none'], "error!"
        if label == 'positive':
            n = 0
        elif label == 'neutral':
            n = 1
        elif label == 'negative':
            n = 2
        elif label == 'conflict':
            n = 3
        elif label == 'none':
            n = 4
        y_true.append(n)
    return y_true


def get_y_pred(pred_data_dir, task_name):
    pred = []
    score = []
    with open(pred_data_dir + "/test.txt", "r", encoding="utf-8") as f:
        s = f.readline().strip().split()
        while s:
            pred.append(int(s[0]))
            score.append([float(s[1]), float(s[2]), float(s[3]), float(s[4]), float(s[5])])
            s = f.readline().strip().split()
    return pred, score


def semeval_PRF(y_true, y_pred):
    """
    Calculate "Micro P R F" of aspect detection task of SemEval-2014.
    """
    s_all = 0
    g_all = 0
    s_g_all = 0
    for i in range(len(y_pred)//5):
        s = set()
        g = set()
        for j in range(5):
            if y_pred[i*5+j] != 4:
                s.add(j)
            if y_true[i*5+j] != 4:
                g.add(j)
        if len(g) == 0:
            continue
        s_g = s.intersection(g)
        s_all += len(s)
        g_all +=len(g)
        s_g_all += len(s_g)

    p = s_g_all/s_all
    r = s_g_all/g_all
    f = 2*p*r/(p+r)

    return p, r, f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        choices=["semeval_single", "travel_experience"
                                 "semeval_QA_EXPT", "semeval_QA_T"],
                        help="The name of the task to evalution.")
    parser.add_argument("--pred_data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The pred data dir.")
    args = parser.parse_args()
    result = collections.OrderedDict()

    y_true = get_y_true(args.task_name)
    y_pred, score = get_y_pred(args.pred_data_dir, args.task_name)
    aspect_P, aspect_R, aspect_F = semeval_PRF(y_true, y_pred)

    result = {'aspect_P': aspect_P,
              'aspect_R': aspect_R,
              'aspect_F': aspect_F}

    for key in result.keys():
        print(key, "=", str(result[key]))


if __name__ == "__main__":
    main()
