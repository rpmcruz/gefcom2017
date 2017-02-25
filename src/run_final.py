import numpy as np
import pandas as pd
import model_gboost
from features import extract_date_features
import datetime
import sys
from sklearn.isotonic import IsotonicRegression
import platform

N = 24*30
START = datetime.datetime(2017, 4, 1)


def create_future():
    dates = [START + datetime.timedelta(hours=h) for h in range(N)]
    return extract_date_features(dates, 2017)


def post_processing(yps):
    yps = np.asarray(yps)
    m = IsotonicRegression()
    for i in range(yps.shape[1]):
        yps[:, i] = m.fit_transform(np.arange(9), yps[:, i])
    return yps


def predict_region(region):
    sys.stdout.write(region + '\n')
    sys.stdout.flush()
    df = pd.read_csv('../out/data/%s.csv' % region)

    quantiles = np.arange(0.1, 1, 0.1)
    yps = [[] for _ in quantiles]
    ys = [[] for _ in quantiles]

    future_df = create_future()

    for i, quantile in enumerate(quantiles):
        sys.stdout.write('model %.1f\n' % quantile)
        sys.stdout.flush()

        NMODELS = 3
        models = [None] * NMODELS
        for j in range(NMODELS):
            model = model_gboost.Model(
                quantile, n_estimators=500, max_depth=10,
                learning_rate=0.3,
                min_samples_leaf=8)
            X = model.get_features(df)
            y = df.as_matrix(['DEMAND'])[:, 0]
            model.fit(X, y)
            models[j] = model

        Xp = model.get_features(future_df)
        yp = np.zeros(len(Xp))
        for model in models:
            yp += model.predict(Xp) / NMODELS
        print('yp:', yp[:10])
        yps[i] = yp
        ys[i] = y

    yps = post_processing(yps)

    out = pd.DataFrame()
    out['Date'] = [d.strftime('%m/%d/%Y') for d in future_df['date']]
    out['Hour'] = future_df['hour']+1
    for i, quantile in enumerate(quantiles):
        out['Q%d' % (quantile*100)] = yps[i]
    return region, out


if __name__ == '__main__':
    computer = platform.uname()[1]
    out_filename = '../out/yp/vilab-%s-%s.xls' % (
        computer, START.strftime('%Y-%m-%d'))

    regions = ['CT', 'ME', 'NEMASSBOST', 'NH', 'RI', 'SEMASS','VT', 
               'WCMASS', 'MASS', 'TOTAL']

    import multiprocessing
    p = multiprocessing.Pool(4)
    res = p.map(predict_region, regions)

    writer = pd.ExcelWriter(out_filename)
    for region, out in res:
        out.to_excel(writer, region, index=False)
    writer.save()

# draw
'''
plt.style.use('ggplot')
plt.plot(range(len(ys)), ys, '.', label='real', alpha=0.8)
for quantile, yp in zip(quantiles, yps):
    plt.plot(
        range(len(yp)), yp, label='quantile %.2f' % quantile, linewidth=2)
plt.xlabel('time')
plt.ylabel('Demand')
legend = plt.legend(loc='lower left', ncol=2)
legend.get_frame().set_facecolor('#ffffff')
plt.savefig('../out/%s-%s.png' % (out_filename[:-4], region))
plt.show()
'''
