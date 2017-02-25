import numpy as np
import pandas as pd
from score import pinball_score
from sklearn.model_selection import KFold
import model_gboost
import pickle
#from drybulb import DryBulbModel
import os

if __name__ == '__main__':
    region = 'ME'
    data_filename = '../out/data/%s.csv' % region
    df = pd.read_csv(data_filename)

    quantiles = np.arange(0.1, 1, 0.1)
    for quantile in quantiles:
        print('Region: %s - quantile: %.1f' % (region, quantile))
        models = [
            model_gboost.Model(quantile,
                n_estimators=500,  # 500
                learning_rate=0.3,
                max_depth = 10,
                min_samples_leaf=8
            )
        ]

        for model in models:
            nmodels = len(os.listdir('../out/models'))
            model_filename = '../out/models/%s-q%d-%d.csv' % (
                region, quantile*10, nmodels)

            print('Run model...')
            X = model.get_features(df)
            y = df.as_matrix(['DEMAND'])[:, 0]

            yps = np.zeros(len(y))
            scores = np.zeros(3)
            for k, (tr, ts) in enumerate(KFold(3, True).split(X)):
                model.fit(X[tr], y[tr])
                yp = model.predict(X[ts])
                score = pinball_score(quantile, y[ts], yp)
                scores[k] = score
                print('fold %d - score: %.3f' % (k, score))
                yps[ts] = yp
            header = '%.3f (%.3f)' % (np.mean(scores), np.std(scores))
            np.savetxt(model_filename, yps, header=header)
