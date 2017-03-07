# GEFCOM 2017 Report

by Ivo Silva, Ricardo Cruz and Carla Gonçalves

## Competition

The goal was to predict **electricity demand** for several **regions** in New England (US), including the aggregated regions (hierarhical predictions), and demand needs to be predicted for the 9 quantiles (probabilistic predictions).

The metric is [the pinball function](https://www.lokad.com/pinball-loss-function-definition) divided by a baseline prediction.

Rules: [http://blog.drhongtao.com/2016/10/instructions-for-gefcom2017-qualifying-match.html](http://blog.drhongtao.com/2016/10/instructions-for-gefcom2017-qualifying-match.html)

## Data

* the granularity of the observations was in hours
* several variables were provided but only **demand** and **temperature** variables could be used for the defined-data, plus information on US federal holidays

## Feature Engineering

Several experiments were performed. In the end, the following features were used:

* $sin(\frac{2\pi}{24}h)$ and $cos(\frac{2\pi}{24}h)$ to make hour rotation invariant
* one-hot encoded weekday
* the absolute distance in days to several days of the year: 04/25, 07/20, 10/31, to thanksgiven, and to every day 15 of each month
* distance to the hottest and the colest days of the year (selected by analazing Dry-Bulb timeseries)
* the absolute distance to the closest holiday
* whether the hour is in daylight-saving time
* whether it is a working day or weekend, or if a "night" hour (between midnight and 6am), and whether it is adjacent to an holiday
* whether it is one of four selected "special" holidays, such as Christmas
* whether it is a holiday neighbour (day before or after)

Features were selected according to feature importance computed from the gradient boosting trees model. Some cross validation was also used for feature selection.

## Model

The model used was Gradient Boosting Trees, using the pinball function as the loss function. Each of which contained 500 estimators. Hyperparameter selection was performed by human-guided grid search. Each model was trained independently for each quantile.

The final prediction was the average of three such models, using a isotonic regression for cosmetic reasons: to avoid quantile overlap since quantile-models were trained independently.

No special hierarchical treatment was performed. The running time was over 6 hours in an i7 with 8 hyperthreads running 8 processes. Memory contrains was not an issue for this machine.

This model was used for the latter competitions. The model was improved across the competition. This report was only prepared afterwards.

## Thoughts

What was learned from the competition:

* PCA was not found to improve predictable efficiency
* quantile regression was first used in conjunction with the gradient boosting trees, as the first model in the ensemble, but did not seem to improve predictions signifiticavely
* likewise, an attempt at [model stacking](http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/) did not seem to improve predictability, and was not pursued
* we did not use the provided historical temperature because we would need to make our own predictions, and our attempt at doing so did not improve the base model.
