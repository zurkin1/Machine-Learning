# line plot of time series
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
import numpy

series = Series.from_csv('daily-minimum-temperatures-in-me.csv', header=0)
print(series.head(20))
#series.plot()
#pyplot.show()

split_point = len(series) - 7
dataset, validation = series[0:split_point], series[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('arima_dataset.csv')
validation.to_csv('arima_validation.csv')

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return numpy.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# seasonal difference
X = series.values
days_in_year = 365
differenced = difference(X, days_in_year)
# fit model
model = ARIMA(differenced, order=(7,0,1))
model_fit = model.fit(disp=0)
# print summary of fit model
print(model_fit.summary())
# one-step out-of sample forecast
forecast = model_fit.forecast()[0]
# invert the differenced forecast to something usable
forecast = inverse_difference(X, forecast, days_in_year)
print('25.12.1990 forecast: %f' % forecast)

# Predict: one-step out of sample forecast. Predict arbitrary in-sample and out-of-sample time steps
start_index = len(differenced)
end_index = len(differenced)
forecast = model_fit.predict(start=start_index, end=end_index)
forecast = inverse_difference(X, forecast, days_in_year)
print('25.12.1990 forecast: %f' % forecast)

# multi-step out-of-sample forecast. steps=the number of time steps to forecast
forecast = model_fit.forecast(steps=7)[0]
# invert the differenced forecast to something usable
history = [x for x in X]
day = 1
for yhat in forecast:
	inverted = inverse_difference(history, yhat, days_in_year)
	print('Day %d: %f' % (day, inverted))
	history.append(inverted)
	day += 1