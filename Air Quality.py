## Import packages
from numpy import unique
from numpy import nan
from numpy import array
from numpy import savetxt
from pandas import read_csv
from numpy import loadtxt
from numpy import isnan
from numpy import count_nonzero
from numpy import array
from numpy import nanmedian
from numpy import save
from numpy import load
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

## ----Data Preparation by Group----

# split the dataset by 'chunkID', return a dict of id to rows
def to_chunks_before(values, chunk_ix=1):
    chunks = dict()
    # get the unique chunk ids
    chunk_ids = unique(values[:, chunk_ix])
    # group rows by chunk id
    for chunk_id in chunk_ids:
        selection = values[:, chunk_ix] == chunk_id
        chunks[chunk_id] = values[selection, :]
    return chunks

# split each chunk into train/test sets
def split_train_test(chunks, row_in_chunk_ix=2):
    train, test = list(), list()
    # first 5 days of hourly observations for train
    cut_point = 5 * 24
    # enumerate chunks
    for k, rows in chunks.items():
        # split chunk rows by 'position_within_chunk'
        train_rows = rows[rows[:, row_in_chunk_ix] <= cut_point, :]
        test_rows = rows[rows[:, row_in_chunk_ix] > cut_point, :]
        if len(train_rows) == 0 or len(test_rows) == 0:
            print('>dropping chunk=%d: train=%s, test=%s' % (k, train_rows.shape, test_rows.shape))
            continue
        # store with chunk id, position in chunk, hour and all targets
        indices = [1, 2, 5] + [x for x in range(56, train_rows.shape[1])]
        train.append(train_rows[:, indices])
        test.append(test_rows[:, indices])
    return train, test

# return a list of relative forecast lead times
def get_lead_times():
    return [1, 2, 3, 4, 5, 10, 17, 24, 48, 72]

# convert the rows in a test chunk to forecasts
def to_forecasts(test_chunks, row_in_chunk_ix=1):
    # get lead times
    lead_times = get_lead_times()
    # first 5 days of hourly observations for train
    cut_point = 5 * 24
    forecasts = list()
    # enumerate each chunk
    for rows in test_chunks:
        chunk_id = rows[0, 0]
        # enumerate each lead time
        for tau in lead_times:
            # determine the row in chunk we want for the lead time
            offset = cut_point + tau
            # retrieve data for the lead time using row number in chunk
            row_for_tau = rows[rows[:, row_in_chunk_ix] == offset, :]
            # check if we have data
            if len(row_for_tau) == 0:
                # create a mock row [chunk, position, hour] + [nan...]
                row = [chunk_id, offset, nan] + [nan for _ in range(39)]
                forecasts.append(row)
            else:
                # store the forecast row
                forecasts.append(row_for_tau[0])
    return array(forecasts)

## ----Missing Data Imputation, Split The Data into Training & Test Sets for Supervised Learning ----

# split the dataset by 'chunkID', return a list of chunks
def to_chunks_after(values, chunk_ix=0):
    chunks = list()
    # get the unique chunk ids
    chunk_ids = unique(values[:, chunk_ix])
    # group rows by chunk id
    for chunk_id in chunk_ids:
        selection = values[:, chunk_ix] == chunk_id
        chunks.append(values[selection, :])
    return chunks

# interpolate series of hours (in place) in 24 hour time
def interpolate_hours(hours):
    # find the first hour
    ix = -1
    for i in range(len(hours)):
        if not isnan(hours[i]):
            ix = i
            break
    # fill-forward
    hour = hours[ix]
    for i in range(ix + 1, len(hours)):
        # increment hour
        hour += 1
        # check for a fill
        if isnan(hours[i]):
            hours[i] = hour % 24
    # fill-backward
    hour = hours[ix]
    for i in range(ix - 1, -1, -1):
        # decrement hour
        hour -= 1
        # check for a fill
        if isnan(hours[i]):
            hours[i] = hour % 24

# return true if the array has any non-nan values
def has_data(data):
    return count_nonzero(isnan(data)) < len(data)

# impute missing data
def impute_missing(train_chunks, rows, hours, series, col_ix):
    # impute missing using the median value for hour in all series
    imputed = list()
    for i in range(len(series)):
        if isnan(series[i]):
            # collect all rows across all chunks for the hour
            all_rows = list()
            for rows in train_chunks:
                [all_rows.append(row) for row in rows[rows[:, 2] == hours[i]]]
            # calculate the central tendency for target
            all_rows = array(all_rows)
            # fill with median value
            value = nanmedian(all_rows[:, col_ix])
            if isnan(value):
                value = 0.0
            imputed.append(value)
        else:
            imputed.append(series[i])
    return imputed

# layout a variable with breaks in the data for missing positions
def variable_to_series(chunk_train, col_ix, n_steps=5 * 24):
    # lay out whole series
    data = [nan for _ in range(n_steps)]
    # mark all available data
    for i in range(len(chunk_train)):
        # get position in chunk
        position = int(chunk_train[i, 1] - 1)
        # store data
        data[position] = chunk_train[i, col_ix]
    return data

# created input/output patterns from a sequence
def supervised_for_lead_time(series, n_lag, lead_time):
    samples = list()
    # enumerate observations and create input/output patterns
    for i in range(n_lag, len(series)):
        end_ix = i + (lead_time - 1)
        # check if can create a pattern
        if end_ix >= len(series):
            break
        # retrieve input and output
        start_ix = i - n_lag
        row = series[start_ix:i] + [series[end_ix]]
        samples.append(row)
    return samples

# create supervised learning data for each lead time for this target
def target_to_supervised(chunks, rows, hours, col_ix, n_lag):
    train_lead_times = list()
    # get series
    series = variable_to_series(rows, col_ix)
    if not has_data(series):
        return None, [nan for _ in range(n_lag)]
    # impute
    imputed = impute_missing(chunks, rows, hours, series, col_ix)
    # prepare test sample for chunk-variable
    test_sample = array(imputed[-n_lag:])
    # enumerate lead times
    lead_times = get_lead_times()
    for lead_time in lead_times:
        # make input/output data from series
        train_samples = supervised_for_lead_time(imputed, n_lag, lead_time)
        train_lead_times.append(train_samples)
    return train_lead_times, test_sample

# prepare training [var][lead time][sample] and test [chunk][var][sample]
def data_prep(chunks, n_lag, n_vars=39):
    lead_times = get_lead_times()
    train_data = [[list() for _ in range(len(lead_times))] for _ in range(n_vars)]
    test_data = [[list() for _ in range(n_vars)] for _ in range(len(chunks))]
    # enumerate targets for chunk
    for var in range(n_vars):
        # convert target number into column number
        col_ix = 3 + var
        # enumerate chunks to forecast
        for c_id in range(len(chunks)):
            rows = chunks[c_id]
            # prepare sequence of hours for the chunk
            hours = variable_to_series(rows, 2)
            # interpolate hours
            interpolate_hours(hours)
            # check for no data
            if not has_data(rows[:, col_ix]):
                continue
            # convert series into training data for each lead time
            train, test_sample = target_to_supervised(chunks, rows, hours, col_ix, n_lag)
            # store test sample for this var-chunk
            test_data[c_id][var] = test_sample
            if train is not None:
                # store samples per lead time
                for lead_time in range(len(lead_times)):
                    # add all rows to the existing list of rows
                    train_data[var][lead_time].extend(train[lead_time])
        # convert all rows for each var-lead time to a numpy array
        for lead_time in range(len(lead_times)):
            train_data[var][lead_time] = array(train_data[var][lead_time])
    return array(train_data), array(test_data)

# fit a single model
def fit_model(model, X, y):
    # clone the model configuration
    local_model = clone(model)
    # fit the model
    local_model.fit(X, y)
    return local_model

# fit one model for each variable and each forecast lead time [var][time][model]
def fit_models(model, train):
    # prepare structure for saving models
    models = [[list() for _ in range(train.shape[1])] for _ in range(train.shape[0])]
    R_score = [list() for _ in range(train.shape[0])]
    # R_score = list()
    # enumerate vars
    for i in range(train.shape[0]):
        # enumerate lead times
        for j in range(train.shape[1]):
            # get data
            data = train[i, j]
            X, y = data[:, :-1], data[:, -1]
            # fit model
            local_model = fit_model(model, X, y)
            models[i][j].append(local_model)
            R_score[i].append(local_model.score(X, y))
            #print(local_model.score(X, y))
    return models, R_score

# return forecasts as [chunks][var][time]
def make_predictions(models, test):
    lead_times = get_lead_times()
    predictions = list()
    # enumerate chunks
    for i in range(test.shape[0]):
        # enumerate variables
        chunk_predictions = list()
        for j in range(test.shape[1]):
            # get the input pattern for this chunk and target
            pattern = test[i, j]
            # assume a nan forecast
            forecasts = array([nan for _ in range(len(lead_times))])
            # check we can make a forecast
            if has_data(pattern):
                pattern = pattern.reshape((1, len(pattern)))
                # forecast each lead time
                forecasts = list()
                for k in range(len(lead_times)):
                    yhat = models[j][k][0].predict(pattern)
                    forecasts.append(yhat[0])
                forecasts = array(forecasts)
            # save forecasts for each lead time for this variable
            chunk_predictions.append(forecasts)
        # save forecasts for this chunk
        chunk_predictions = array(chunk_predictions)
        predictions.append(chunk_predictions)
    return array(predictions)

# convert the test dataset in chunks to [chunk][variable][time] format
def prepare_test_forecasts(test_chunks):
    predictions = list()
    # enumerate chunks to forecast
    for rows in test_chunks:
        # enumerate targets for chunk
        chunk_predictions = list()
        for j in range(3, rows.shape[1]):
            yhat = rows[:, j]
            chunk_predictions.append(yhat)
        chunk_predictions = array(chunk_predictions)
        predictions.append(chunk_predictions)
    return array(predictions)

## ---- Evaluate Models Using Mean Absolute Error (MAE) ----

# calculate the error between an actual and predicted value
def calculate_error(actual, predicted):
    # give the full actual value if predicted is nan
    if isnan(predicted):
        return abs(actual)
    # calculate abs difference
    return abs(actual - predicted)

# evaluate a forecast in the format [chunk][variable][time]
def evaluate_forecasts(predictions, testset):
    lead_times = get_lead_times()
    total_mae, times_mae = 0.0, [0.0 for _ in range(len(lead_times))]
    total_c, times_c = 0, [0 for _ in range(len(lead_times))]
    # enumerate test chunks
    for i in range(len(test_chunks)):
        # convert to forecasts
        actual = testset[i]
        predicted = predictions[i]
        # enumerate target variables
        for j in range(predicted.shape[0]):
            # enumerate lead times
            for k in range(len(lead_times)):
                # skip if actual in nan
                if isnan(actual[j, k]):
                    continue
                # calculate error
                error = calculate_error(actual[j, k], predicted[j, k])
                # update statistics
                total_mae += error
                times_mae[k] += error
                total_c += 1
                times_c[k] += 1
    # normalize summed absolute errors
    total_mae /= total_c
    times_mae = [times_mae[i] / times_c[i] for i in range(len(times_mae))]
    return total_mae, times_mae

# summarize scores
def summarize_error(name, total_mae):
    print('%s: %.3f MAE' % (name, total_mae))

## ---- Chosen Regression Models ----

# a list of regression models
def get_models(models=dict()):
    # linear models
    models['Multiple Linear'] = LinearRegression()
    models['Ridge'] = Ridge()
    print('Defined %d models' % len(models))
    return models

# evaluate a suite of models
def evaluate_models(models, train, test, actual):
    Total_MAE = dict()
    R = dict()
    for name, model in models.items():
        # fit models
        fits, R_score = fit_models(model, train)
        # make predictions
        predictions = make_predictions(fits, test)
        # evaluate forecast
        total_mae, _ = evaluate_forecasts(predictions, actual)
        Total_MAE[name] = total_mae
        R[name] = R_score
        # summarize forecast
        #summarize_error(name, total_mae)
    return Total_MAE, R #total_mae, R_score

# load dataset
dataset = read_csv('/Users/mengkaixu/Desktop/Time Series Dataset/TrainingData.csv', header=0)
# group data by chunks (Group)
values = dataset.values
chunks = to_chunks_before(values)
# split into train/test
train, test = split_train_test(chunks)
# flatten training chunks (Group) to rows
train_rows = array([row for rows in train for row in rows])
# print(train_rows.shape)
print('Train Rows: %s' % str(train_rows.shape))
# reduce train to forecast lead times only
test_rows = to_forecasts(test)
print('Test Rows: %s' % str(test_rows.shape))
# save reshaped datasets (without missing data imputation)
savetxt('/Users/mengkaixu/Desktop/Time Series Dataset/naive_train.csv', train_rows, delimiter=',')
savetxt('/Users/mengkaixu/Desktop/Time Series Dataset/naive_test.csv', test_rows, delimiter=',')
# load dataset
train = loadtxt('/Users/mengkaixu/Desktop/Time Series Dataset/naive_train.csv', delimiter=',')
test = loadtxt('/Users/mengkaixu/Desktop/Time Series Dataset/naive_test.csv', delimiter=',')
# group data by chunks
train_chunks = to_chunks_after(train)
test_chunks = to_chunks_after(test)
# missing data imputation and convert training data into supervised learning data based on the lag size n_lag
n_lag = 10
train_data, test_data = data_prep(train_chunks, n_lag)
print(train_data.shape, test_data.shape)
# save train and test sets to file
save('/Users/mengkaixu/Desktop/Time Series Dataset/supervised_train.npy', train_data)
save('/Users/mengkaixu/Desktop/Time Series Dataset/supervised_test.npy', test_data)
# load supervised datasets
train = load('/Users/mengkaixu/Desktop/Time Series Dataset/supervised_train.npy', allow_pickle=True)
test = load('/Users/mengkaixu/Desktop/Time Series Dataset/supervised_test.npy', allow_pickle=True)
#print(train.shape, test.shape)
# load test chunks for validation
testset = loadtxt('/Users/mengkaixu/Desktop/Time Series Dataset/naive_test.csv', delimiter=',')
test_chunks = to_chunks_after(testset)
actual = prepare_test_forecasts(test_chunks)
# prepare list of models
models = get_models()
# evaluate models- R_sq: R^2 for training dataset, MAE for testing dataset
Total_MAE, R_sq = evaluate_models(models, train, test, actual)

# plot R squared obtained from training data vs lead time for Multiple Linear and Ridge regression
lead_time = get_lead_times()
for i in range(len(R_sq.keys())):
    plt.subplot(1, 2, i+1)
    for j in range(len(R_sq[R_sq.keys()[i]])):
        plt.plot(lead_time, R_sq[R_sq.keys()[i]][j], label='Ridge Regression')
    plt.xlabel('Lead Time', fontsize=16)
    plt.ylabel('R^2', fontsize=16)
    plt.title(R_sq.keys()[i], fontsize=18)
plt.show()
# print MAE for each model
print('MAE %s:' % Total_MAE)

# Test Model (Ridge Regression) with Different Lag Size
lag_len = [1, 3, 5, 10, 20, 32, 48]
score = list()
for n_lag in lag_len:
    train_data, test_data = data_prep(train_chunks, n_lag)
    # save train and test sets to file with lag size n_lag
    save('/Users/mengkaixu/Desktop/Time Series Dataset/supervised_train.npy', train_data)
    save('/Users/mengkaixu/Desktop/Time Series Dataset/supervised_test.npy', test_data)
    # load supervised datasets with lag size n_lag
    train = load('/Users/mengkaixu/Desktop/Time Series Dataset/supervised_train.npy', allow_pickle=True)
    test = load('/Users/mengkaixu/Desktop/Time Series Dataset/supervised_test.npy', allow_pickle=True)
    # load test chunks for validation
    testset = loadtxt('/Users/mengkaixu/Desktop/Time Series Dataset/naive_test.csv', delimiter=',')
    test_chunks = to_chunks_after(testset)
    actual = prepare_test_forecasts(test_chunks)
    model = models['Ridge']
    fits, R_score = fit_models(model, train)
    # make predictions
    predictions = make_predictions(fits, test)
    # evaluate forecast
    total_mae, _ = evaluate_forecasts(predictions, actual)
    # summarize forecast
    summarize_error('Ridge',total_mae)
    # store the error according to lag size
    score.append(total_mae)

# save MAE obtained from test data for the model with different lag sizes
save('/Users/mengkaixu/Desktop/Time Series Dataset/score.npy', score)
score = load('/Users/mengkaixu/Desktop/Time Series Dataset/score.npy', allow_pickle=True)
# plot lag size vs MAE
plt.plot(lag_len, score, label='Ridge Regression')
plt.legend(loc="upper right")
plt.legend(fontsize=16)
plt.xlabel('Lag Temporal',fontsize=18)
plt.ylabel('MAE',fontsize=18)
plt.show()
