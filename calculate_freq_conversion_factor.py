import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
import numpy as np

plt.rcParams['figure.figsize'] = (13, 9)

# define the linear function to be fitted
def lin(x, a, b):
    """A linear function for fitting"""
    return x*a + b


def gaussian(x, mu, sigma, A, offset):
    """ A Gaussian function """
    return norm.pdf(x, loc=mu, scale=sigma)*A + offset


def doubble_gaussian(x, mu1, mu2, sigma1, sigma2, A1, A2, offset):
    return A1*norm.pdf(x, loc=mu1, scale=sigma1) + A2*norm.pdf(x, loc=mu2,
                                                               scale=sigma2) + offset


def lorenzian(x, gamma, x0, A, offset):
    return A/ ((x**2-x0**2)**2 + gamma**2*x0**2) + offset

def double_lorenz(x, x0, x1, g0, g1, A0, A1, o0):
    return lorenzian(x, g0, x0, A0, o0) + lorenzian(x, g1, x1, A1, o0)

def multiplicative_error_propagation(rel_err):
    return np.sqrt(sum(np.array(rel_err)**2))

def additive_error_propagation(errs):
    return np.sqrt(sum(np.array(errs)**2))

# here the different magic numbers are recorded, that I extracted from looking
# at the reduced dataset
show = True
# The start and end indices for finding the maximum of the triangular ramp
max_ch_3_start_index = 2600
max_ch_3_end_index = 3100

# the start and end indices for finding the minimum of the triangular ramp
min_ch_3_start_index = 6350
min_ch_3_end_index = 6500

data = pd.read_csv('reduced_data.csv', index_col=0)
small_data = pd.read_csv('reduced_data_01.csv', index_col=0)

# shift the small data so that it fits with the rest of it
sdo = 0.001
small_data['high_time'] += sdo
small_data['low_time'] += sdo
# flip small data
small_data['high_time'] *= -1
small_data['low_time'] *= -1

# determine the min and the max of the triangular ramp
max_ramp_timestamp = data[max_ch_3_start_index:max_ch_3_end_index].max()['high_time']
min_ramp_timestamp = data[min_ch_3_start_index:min_ch_3_end_index].min()['high_time']

# there are three ramps and they can be transformed into three pictures of the
# same ramp
# for that the max and the min value are taken as cut points
ramp_0 = data.where(data['high_time'] < max_ramp_timestamp)
ramp_0 = ramp_0.dropna(axis=0)

ramp_1 = data.where((data['high_time'] > max_ramp_timestamp)
                    & (data['high_time'] < min_ramp_timestamp))
ramp_1 = ramp_1.dropna(axis=0)

ramp_2 = data.where(data['high_time'] > min_ramp_timestamp)
ramp_2 = ramp_2.dropna(axis=0)

# correct the time shifts
ramp_0['high_time'] = (ramp_0['high_time'] - max_ramp_timestamp) * (-1)
ramp_0['low_time'] = (ramp_0['low_time'] - max_ramp_timestamp) * (-1)
ramp_0['low_time'] -= ramp_0['low_time'].min()
ramp_0['high_time'] -= ramp_0['low_time'].min()

ramp_1['high_time'] = ramp_1['high_time'] - max_ramp_timestamp
ramp_1['low_time'] = ramp_1['low_time'] - max_ramp_timestamp
ramp_1['low_time'] -= ramp_1['low_time'].min()
ramp_1['high_time'] -= ramp_1['low_time'].min()

ramp_2['high_time'] = (ramp_2['high_time'] - min_ramp_timestamp) * (-1)
ramp_2['low_time'] = (ramp_2['low_time'] - min_ramp_timestamp) * (-1)
ramp_2['high_time'] -= ramp_2.iloc[-1]['low_time']
ramp_2['low_time'] -= ramp_2.iloc[-1]['low_time']

# we now have three sets of data
ramps = [ramp_0, ramp_1, ramp_2, small_data]

spike_sections = [(0.001, 0.004), (0.004, 0.0060),
                  (0.006, 0.008), (0.008, 0.010)]
spike_width = 0.0003
spike_params = []
for i, ramp in enumerate(ramps):
    rmt = (ramp['low_time'] + ramp['high_time'])/2
    plt.plot(rmt, ramp['ch2'])
    plt.plot(rmt, ramp['ch3'])
    plt.grid()
    plt.savefig('raw_ramp_{}.pdf'.format(i))
    plt.close()
    spike_train_params = []
    for spike in spike_sections:
        sd = ramp.loc[:, ['low_time', 'high_time', 'ch2', 'std_ch2']]
        sd = sd.where((sd['low_time'] > spike[0]) &
                      (sd['low_time'] < spike[1]))
        sd = sd.dropna(axis=0)

        mean_time = (sd['high_time'] + sd['low_time'])/2
        gplot_time = np.linspace(min(mean_time), max(mean_time), 5000)

        spike_height = max(sd['ch2'])
        max_time = float(sd.where(sd['ch2'] ==
                         spike_height)['high_time'].dropna())

        plt.plot(mean_time, sd['ch2'], marker='.', color='blue', linestyle='')

        spopt, spcov = curve_fit(lorenzian, mean_time, sd['ch2'],
                                 p0=(spike_width, max_time,
                                     spike_width*spike_height/2, 0))
        spike_train_params.append((spopt, spcov))
        plt.plot(gplot_time, lorenzian(gplot_time, *spopt),
                 linestyle='--', color='orange')
    plt.title("FP-Resonances with corresponding fits")
    plt.xlabel("t [s]")
    plt.ylabel("U [V]")
    plt.grid()
    plt.savefig("fitted_spikes_ramp_{}.pdf".format(i))
    plt.close()
    print("ramp {}".format(i))
    for p in spike_train_params:
        fwhm = (np.sqrt(p[0][1]**2+p[0][0]*p[0][1])-np.sqrt(p[0][1]**2-p[0][0]*p[0][1]))/2
        print("Spike Location: {:.8f} +- {:.8f}".format(p[0][1], fwhm))
    spike_params.append(spike_train_params)

time_dist_all = []
time_dist_std_all = []
freq_dist_spikes =     2500000000
freq_dist_spikes_std =    1000000

# after fitting the spikes, the conversion from time to frequency done
for i, (train, ramp) in enumerate(zip(spike_params, ramps)):
    # get the time of the spike and its uncertainties from the fitted data
    spike_pos = []
    spike_std = []
    reference_freq_time = train[0][0][1]
    reference_std = np.sqrt(train[0][1][1][1])
    spike_pos.append(reference_freq_time)
    spike_std.append(reference_std)
    for spike in train[1:]:
        spike_pos.append(spike[0][1])
        fwhm = np.sqrt(spike[0][1]**2+spike[0][0]*spike[0][1])-np.sqrt(spike[0][1]**2-spike[0][0]*spike[0][1])
        spike_std.append(fwhm/2)
    spike_pos = np.array(spike_pos)
    spike_std = np.array(spike_std)
    # now that the timestamps of the spikes are known, the uncertainties
    # can be calculated
    # calculate spike distance
    time_dist = spike_pos[1:] - spike_pos[:-1]
    for d in time_dist:
        time_dist_all.append(d)
    # calculate the uncertainty for every spike distance
    time_dist_std = np.sqrt(spike_std[1:]**2 + spike_std[:-1]**2)
    for stdd in time_dist_std:
        time_dist_std_all.append(stdd)
    # as the dataset contains the upper and lower bound on a measuring interval
    # convert this to the mean and use the width of the interval as std
    mean_time = (ramp['high_time'] + ramp['low_time'])/2
    time_std = np.absolute(ramp['high_time'] - ramp['low_time'])
    mean_time = mean_time - reference_freq_time
    # delete the uneccesary columns
    del ramp['high_time']
    del ramp['low_time']

    mf = pd.Series(mean_time, name='time')
    stdf = pd.Series(time_std, name='std_time')

    # add the converted columns back into the dataset
    ramp = pd.concat([ramp, mf, stdf], axis=1)
    ramps[i] = ramp

# calculate the mean distance and propagate errors
time_dist_mean = np.mean(time_dist_all)
time_dist_mean_std = additive_error_propagation(errs=time_dist_std_all)/len(time_dist_std_all)
print("Mean time between peaks: {:.5f} +- {:.5f}".format(time_dist_mean, time_dist_mean_std))
# calculate frequency to time conversion factor and error
freq_conversion_factor = freq_dist_spikes / time_dist_mean
freq_conversion_factor_std = multiplicative_error_propagation(
        [freq_dist_spikes_std/freq_dist_spikes,
         time_dist_mean_std/time_dist_mean]) * freq_conversion_factor
print("frequency Conversion Factor: {:.5f} +- {:.5f}%".format(freq_conversion_factor, freq_conversion_factor_std/freq_conversion_factor*100))

for d, std in zip(time_dist_all, time_dist_std_all):
    df = d * freq_conversion_factor
    dfstd = multiplicative_error_propagation([std/d,
        freq_conversion_factor_std/freq_conversion_factor]) * df
    print("Time delta converted to frequency: {:.5f} +- {:.5f}".format(df, dfstd))

for i, ramp in enumerate(ramps):
    f = ramp['time'] * freq_conversion_factor
    stdf = [multiplicative_error_propagation([std_time/time,
                                              freq_conversion_factor_std/freq_conversion_factor]) * np.abs(fr)
                                              for std_time, time, fr in
                                              zip(ramp['std_time'], ramp['time'], f)]
    del ramp['time']
    del ramp['std_time']
    mf = pd.Series(f, name='f', index=ramp.index)
    sf = pd.Series(stdf, name='std_f', index=ramp.index)
    ramp = pd.concat([ramp, mf, sf], axis=1)
    ramps[i] = ramp

    # plot to see the error bars
    plt.errorbar(ramp['f'], ramp['ch2'], xerr=ramp['std_f'], yerr=ramp['std_ch2'],
                 marker='', color='blue', linestyle='', label="CH2")
    plt.errorbar(ramp['f'], ramp['ch3'], xerr=ramp['std_f'], yerr=ramp['std_ch3'],
                 marker='', color='darkred', linestyle='', label="CH3")
    plt.grid()
    plt.legend()
    plt.xlabel("offset frequency [Hz]")
    plt.ylabel("voltage [V]")
    plt.savefig("ramp_{}_with_error_propagation.pdf".format(i))
    plt.close()
    ramp.to_csv("ramp_{}.csv".format(i))
