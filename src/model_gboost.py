from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
import numpy as np    

PRINT_RESULTS = False
USE_PCA = False
#['day_solstice', 'day',
#['hour_%d' % i for i in range(23)] + \
#['month_%d' % i for i in range(11)] + \
#'workday'

FIXED_FEATURES = \
    ['sin.hour', 'cos.hour'] + \
    ['monday', 'tuesday', 'wednesday', 'thursday', 
    'friday', 'saturday', 'sunday'] + \
    ['holiday']

RANDOM_FEATURES = \
    ['dist_25_04', 'dist_20_07', 'dist_31_10', 'distance_Thanksgiven',
     'day', 'day_solstice', 'day_solstice_2'] + \
    ['dist.day15.%d' % month for month in range(1, 12+1)] + \
    ['is-dst', 'dist.holiday', 'workday', 'weekend', 'night_hours', 
    'holiday_neighbor' ] + \
    ['special_holiday.%d' % i for i in range(4)]


class Model:
    def __init__(self, quantile, **args):
        self.args = args
        self.model = GradientBoostingRegressor(
            'quantile', alpha=quantile, **args)
        self.features = FIXED_FEATURES[:] + RANDOM_FEATURES
        #n = int(len(RANDOM_FEATURES) * 0.6)
        #self.features += list(np.random.choice(RANDOM_FEATURES, n, False))

    def get_features(self, df):
        return df.as_matrix(self.features)

    def fit(self, X, y):
        if USE_PCA:
            self.Xmu = np.mean(X, 0)
            self.Xstd = np.std(X, 0)+1e-12
            X = (X-self.Xmu) / self.Xstd
            self.pca = PCA()
            X = self.pca.fit_transform(X)

        self.model.fit(X, y)

        if PRINT_RESULTS:
            print('Feature importance:')
            print('features:', self.features)
            importance = self.model.feature_importances_
            order = np.argsort(importance)[::-1]
            print('order:', order)
            for i in order:
                print('%20s: %.4f' % (self.features[i], importance[i]))

        return self

    def predict(self, X):
        if USE_PCA:
            X = (X-self.Xmu) / self.Xstd
            X = self.pca.transform(X)

        yp = self.model.predict(X)
        return yp

Model.features = []
