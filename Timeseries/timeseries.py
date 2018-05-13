import pandas as pd
import sklearn.metrics as skm


# plot
#pyplot.plot(test)
#pyplot.plot(predictions, color='red')
#pyplot.show()

series = [3,10,12,13,12,10,12]
def average(series):
    return float(sum(series))/len(series)
average(series)
# 10.285714285714286

#moving average using n last points
def moving_average(series, n):
    return average(series[-n:])

moving_average(series, 3)
# 11.333333333333334
moving_average(series, 4)
# 11.75

def average2(series, n=None):
    if n is None:
        return average(series, len(series))
    return float(sum(series[-n:]))/n

# >>> average(series, 3)
# 11.333333333333334
# >>> average(series)
# 10.285714285714286

# weighted average, weights is a list of weights
def weighted_average(series, weights):
    result = 0.0
    weights.reverse()
    for n in range(len(weights)):
        result += series[-n-1] * weights[n]
    return result

weights = [0.1, 0.2, 0.3, 0.4]
weighted_average(series, weights)
# 11.5

# Level: given a series and alpha, return series of smoothed points (no prediction)
def exponential_smoothing(series, alpha):
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result

exponential_smoothing(series, 0.1)
# [3, 3.7, 4.53, 5.377, 6.0393, 6.43537, 6.991833]
exponential_smoothing(series, 0.9)
# [3, 9.3, 11.73, 12.873000000000001, 12.0873, 10.20873, 11.820873]

v_alpha=0.3
v_gamma=0.1
v_beta = 0.05 #0.029
v_slen = 1

# Trend: given a series and alpha, beta, return series of smoothed points (with one extra prediction)
def double_exponential_smoothing(series, alpha=v_alpha, beta=v_beta):
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # we are forecasting
          value = result[-1]
        else:
          value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    return result


double_exponential_smoothing(series, alpha=0.9, beta=0.9)
# [3, 17.0, 15.45, 14.210500000000001, 11.396044999999999, 8.183803049999998, 12.753698384500002, 13.889016464000003]

series = [30,21,29,31,40,48,53,47,37,39,31,29,17,9,20,24,27,35,41,38,27,31,27,26,21,13,21,18,33,35,40,36,22,24,21,20,17,14,17,19,26,29,40,31,20,24,18,26,17,9,17,21,28,32,46,33,23,28,22,27,18,8,17,21,31,34,44,38,31,30,26,32]

def initial_trend(series, slen):
    sum = 0.0
    for i in range(slen):
        sum += float(series[i+slen] - series[i]) / slen
    return sum / slen

initial_trend(series, 12)
# -0.7847222222222222

def initial_seasonal_components(series, slen):
    seasonals = {}
    season_averages = []
    n_seasons = int(len(series)/slen)
    # compute season averages
    for j in range(n_seasons):
        season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen))
    # compute initial values
    for i in range(slen):
        sum_of_vals_over_avg = 0.0
        for j in range(n_seasons):
            sum_of_vals_over_avg += series[slen*j+i]-season_averages[j]
        seasonals[i] = sum_of_vals_over_avg/n_seasons
    return seasonals

initial_seasonal_components(series, 12)
# {0: -7.4305555555555545, 1: -15.097222222222221, 2: -7.263888888888888, 3: -5.097222222222222, 4: 3.402777777777778, 5: 8.069444444444445, 6: 16.569444444444446, 7: 9.736111111111112, 8: -0.7638888888888887, 9: 1.902777777777778, 10: -3.263888888888889, 11: -0.7638888888888887}

def triple_exponential_smoothing(series,  slen=v_slen, alpha=v_alpha, beta=v_beta, gamma=v_gamma, n_preds=1):
    result = []
    seasonals = initial_seasonal_components(series, slen)
    for i in range(len(series)+n_preds):
        if i == 0: # initial values
            smooth = series[0]
            trend = initial_trend(series, slen)
            result.append(series[0])
            continue
        if i >= len(series): # we are forecasting
            m = i - len(series) + 1
            result.append((smooth + m*trend) + seasonals[i%slen])
        else:
            val = series[i]
            last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)
            trend = beta * (smooth-last_smooth) + (1-beta)*trend
            seasonals[i%slen] = gamma*(val-smooth) + (1-gamma)*seasonals[i%slen]
            result.append(smooth+trend+seasonals[i%slen])
    return result

# forecast 24 points (i.e. two seasons)
print(triple_exponential_smoothing(series, 12, 0.716, 0.029, 0.993, 24))
# [30, 20.34449316666667, 28.410051892109554, 30.438122252647577, 39.466817731253066, ...

def f(group):
   #print('f handling deal ', group.DEAL_CODE.min(), ' coupon ', group.COUPON_CODE.min())
   if(len(group.index)<9):
       return(group.SPENT.iloc[0])
   group = group.reset_index(drop=True)
   df = pd.merge(left=group, right=data2, on=['WEEKIND', 'YEARIND'], how='outer')
   #df['SPENT2'] = df.SPENT_x + df.SPENT_y
   df.SPENT = df.SPENT.fillna(0)
   df['DATE'] = 100*df.YEARIND + df.WEEKIND
   df.sort_values(['DATE'], inplace=True)
   df = df.fillna(method='bfill')
   df = df.fillna(method='ffill')
   df = df.dropna()
   df = df.reset_index(drop=True)
   #if(df.ACCOUNT_CODE.min()==75756):
   #    df.to_csv('account75756.csv')
   hws = triple_exponential_smoothing(df.SPENT)
   return hws[-1] # return the predicted value

data2 = pd.read_csv('CSV/User1.csv', index_col=None)
data2['SPENT'] = 0
data = pd.read_csv('c:/Users/dani/PycharmProjects/WOW/CSV/TCUSTCOUPONITEMSSPENT.csv', index_col=None).reset_index(drop=True)
grouped = data.groupby(['DEAL_CODE', 'ACCOUNT_CODE', 'COUPON_CODE']).apply(f)
grouped.to_csv('CSV\TCUST_predicted.csv')

#mse = skm.mean_squared_error(data_merge.COUPON_NETO_SUB, data_merge.BUY)
#print('alpha=[', v_alpha ,'], beta=[', v_beta, '] gamma=[', v_gamma, '] v_slen=[', v_slen, '] mse=[', mse, ']')
