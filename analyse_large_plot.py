import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
import numpy as np

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

def double_lorenz(x, g0, g1, x0, x1, A0, A1, o0, o1):
    return lorenzian(x, g0, x0, A0, o0) + lorenzian(x, g1, x1, A1, o1)

def multiplicative_error_propagation(rel_err):
    return np.sqrt(sum(np.array(rel_err)**2))

def additive_error_propagation(errs):
    return np.sqrt(sum(np.array(errs)**2))
# here the different magic numbers are recorded, that I extracted from looking
# at the reduced dataset
show = False
# The start and end indices for finding the maximum of the triangular ramp
max_ch_3_start_index = 2600
max_ch_3_end_index = 3100

# the start and end indices for finding the minimum of the triangular ramp
min_ch_3_start_index = 6350
min_ch_3_end_index = 6500

data = pd.read_csv('reduced_data.csv', index_col=0)

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

ramp_1['high_time'] = ramp_1['high_time'] - max_ramp_timestamp
ramp_1['low_time'] = ramp_1['low_time'] - max_ramp_timestamp

ramp_2['high_time'] = (ramp_2['high_time'] - min_ramp_timestamp) * (-1)
ramp_2['low_time'] = (ramp_2['low_time'] - min_ramp_timestamp) * (-1)
ramp_2['high_time'] -= ramp_2.iloc[-1]['low_time']
ramp_2['low_time'] -= ramp_2.iloc[-1]['low_time']

# we now have three sets of data
ramps = [ramp_0, ramp_1, ramp_2]

spike_sections = [(0.001, 0.004), (0.004, 0.0060),
                  (0.006, 0.008), (0.008, 0.010), (0.010, 0.013)]
spike_width = 0.0003
spike_params = []
for ramp in ramps:
    rmt = (ramp['low_time'] + ramp['high_time'])/2
    plt.plot(rmt, ramp['ch2'])
    plt.plot(rmt, ramp['ch3'])
    plt.grid()
    plt.savefig('tmp.pdf')
    if show:
        plt.show()
    else:
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
                 linestyle='--', color='blue')
        if show:
            plt.show()
        else:
            plt.close()
    spike_params.append(spike_train_params)

train_params = []
freq_dist_spikes = 2500000000
freq_dist_spikes_std = 1000000

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
        spike_std.append(spike[1][1][1])
    spike_pos = np.array(spike_pos)
    spike_std = np.array(spike_std)
    # now that the timestamps of the spikes are known, the uncertainties
    # can be calculated
    # calculate spike distance
    time_dist = spike_pos[1:] - spike_pos[:-1]
    # calculate the uncertainty for every spike distance
    time_dist_std = np.sqrt(spike_std[1:]**2 + spike_std[:-1]**2)
    # calculate the mean distance and propagate errors
    time_dist_mean = np.mean(time_dist)
    time_dist_mean_std = additive_error_propagation(errs=time_dist_std)/2
    # calculate frequency to time conversion factor and error
    freq_conversion_factor = freq_dist_spikes / time_dist_mean
    freq_conversion_factor_std = multiplicative_error_propagation(
            [freq_dist_spikes_std/freq_dist_spikes,
             time_dist_mean_std/time_dist_mean]) * freq_conversion_factor

    # as the dataset contains the upper and lower bound on a measuring interval
    # convert this to the mean and use the width of the interval as std
    mean_time = (ramp['high_time'] + ramp['low_time'])/2
    time_std = ramp['high_time'] - ramp['low_time']
    mean_time = mean_time - reference_freq_time
    # delete the uneccesary columns
    del ramp['high_time']
    del ramp['low_time']

    freq = mean_time * freq_conversion_factor
    # calculate the frequency uncertainty for every point of the plot
    fstd = np.array([multiplicative_error_propagation([tstd/tmean,
                     freq_conversion_factor_std/freq_conversion_factor])
                     for tstd, tmean in zip(time_std, mean_time)]) * freq
    mf = pd.Series(freq, name='freq')
    stdf = pd.Series(fstd, name='std_freq')

    # add the converted columns back into the dataset
    ramp = pd.concat([ramp, mf, stdf], axis=1)
    ramps[i] = ramp

    # plot to see the error bars
    plt.errorbar(freq, ramp['ch2'], xerr=fstd, yerr=ramp['std_ch2'],
                 marker='', color='blue', linestyle='')
    plt.errorbar(freq, ramp['ch3'], xerr=fstd, yerr=ramp['std_ch3'],
                 marker='', color='darkred', linestyle='')
    plt.grid()
    if show:
        plt.show()
    else:
        plt.close()

# now, to fit the falling ramp, the resonance absorption peaks have to be
# removed (the background is being fitted)
# the format is one tuple per peak with the
# (start_index_of_cut, end_index_of_cut), the first two peaks are in the
# first interval together
peak_cuts = [(-3e9, -0.5e9), (1e9, 2.2e9), (2.2e9, 2.5e9),
             (2.45e9, 4e9), (5e9, 6.8e9),
             (7.5e9, 9e9), (1.25e10, 1.4e10)]
for i, r in enumerate(ramps):
    # remove the peaks so that the ramp can be fitted
    ramp = r.copy()
    for peak in peak_cuts:
        ramp = ramp.where(
                (ramp['freq'] < peak[0]) |
                (ramp['freq'] > peak[1]))
        ramp = ramp.dropna(axis=0)
        plt.plot(ramp['freq'], ramp['ch3'], linestyle='', marker='x')
        if show:
            plt.show()
        else:
            plt.close()
    lin_popt, pcov = curve_fit(lin, ramp['freq'], ramp['ch3'],
                               sigma=ramp['std_ch3'])
    plt.plot(ramp['freq'], ramp['ch3'], linestyle='', marker='x')
    plt.plot(ramp['freq'], lin(ramp['freq'], *lin_popt), linestyle='--',
             label='ramp fit', color='blue')
    if show:
        plt.show()
    else:
        plt.close()
    ramps[i]['ch3'] = ramps[i]['ch3'] - lin(ramps[i]['freq'], *lin_popt)

    # propagate the fit error of the slope into the points, use additive error
    # propagation
    # as the fitted value is subtracted from the original Data
    ramps[i]['std_ch3'] = np.sqrt(ramps[i]['std_ch3']**2 +
                                  (np.sqrt(pcov[0][0]) *
                                   lin(ramps[i]['freq'], *lin_popt))**2)

# the slope has been fitted and subtracted from the data the peaks of the
# non-absorption spectroscopy are now fitted
cuts_in_peak_region = [10, 11, 4.5, 1.3]
cut_in_double_peak = (1.55e9, 1.7e9)
actual_peaks = [peak_cuts[1], peak_cuts[3], peak_cuts[4], peak_cuts[5]]
fitfunc = [gaussian, gaussian, gaussian, gaussian]
peak_parameters = []
peak_regions = []
for ramp in ramps:
    peak_train_params = []
    for i, (pc, ffnc, cut_thresh) in enumerate(zip(actual_peaks, fitfunc,
                                                     cuts_in_peak_region)):
        peak = ramp.where(
            (ramp['freq'] > pc[0]) & (ramp['freq'] < pc[1]))
        peak = peak.dropna(axis=0)

        # now invert the peak
        peak['ch3'] = -peak['ch3']
        peak_regions.append(peak.copy())
        # now after narrowing to the peak region, as we want to fit without the
        # disturbances caused by the saturation absorbance, the tips of the
        # peaks are removed
        peak = peak.where(peak['ch3'] < cut_thresh)
        peak = peak.dropna(axis=0)
        peak.plot(x='freq', y='ch3')
        if show:
            plt.show()
        else:
            plt.close()

        # the starting parameters for the fit must be approximated
        width = peak['freq'].max() - peak['freq'].min()
        height = peak['ch3'].max()
        Area = width * height
        peak_loc = (peak['freq'].max() + peak['freq'].min())/2
        # if i == 0:
        #     parameters = (peak_loc,  peak_loc + width,
        #                   width, width, Area/2, Area/2, 0)
        # else:
        parameters = (peak_loc, width, Area/2, 0)
        # fit Gaussian to peak
        plt.plot(peak['freq'], peak['ch3'], marker='.', linestyle='')
        popt, pcov = curve_fit(ffnc, peak['freq'], peak['ch3'],
                               sigma=peak['std_ch3'], p0=parameters,
                               maxfev=10000)
        plt_freq = np.linspace(peak['freq'].min(), peak['freq'].max(), 10000)
        peak_train_params.append((popt, pcov))
        plt.plot(plt_freq, ffnc(plt_freq, *popt))
        if show:
            plt.show()
        else:
            plt.close()
    peak_parameters.append(peak_train_params)

peaks =[[[],[]], [[],[]], [[],[]], [[],[]]]
averaged_peaks = []
for train in peak_parameters:
    for i, peak in enumerate(train):
        peaks[i][0].append(peak[0][0])
        peaks[i][1].append(np.sqrt(peak[1][0][0]))
for peak in peaks:
    mu = np.mean(peak[0])
    sigma = additive_error_propagation(peak[1])/len(peak[1])
    averaged_peaks.append((mu, sigma))

peak_distances = []
peak_0 = averaged_peaks[0]
for peak in averaged_peaks[1:]:
    dist = peak[0] - peak_0[0]
    dist_std = additive_error_propagation([peak_0[1], peak[1]])
    peak_distances.append((dist, dist_std))
    peak_0 = peak

absorption_cuts = [[(1.5e9, 1.7e9), (2.7e9, 2.9e9), (5.6e9, 5.78e9), (8.15e9, 8.45e9)],
                  [(1.7e9, 1.9e9), (3.e9, 3.2e9), (6e9, 6.18e9), (8.35e9, 8.6e9)],
                  [(1.5e9, 1.7e9), (2.7e9, 2.9e9), (5.6e9, 5.78e9), (8.15e9, 8.45e9)]]
affs = [double_lorenz, double_lorenz, lorenzian, double_lorenz]
# reshape peak array
pregions = [[p for p in peak_regions[::4]],
            [p for p in peak_regions[1::4]],
            [p for p in peak_regions[2::4]],
            [p for p in peak_regions[3::4]]]
peak_parameters = [[p[0] for p in peak_parameters],
                   [p[1] for p in peak_parameters],
                   [p[2] for p in peak_parameters],
                   [p[3] for p in peak_parameters]]
absorption_cuts = [[c[0] for c in absorption_cuts],
                   [c[1] for c in absorption_cuts],
                   [c[2] for c in absorption_cuts],
                   [c[3] for c in absorption_cuts]]
# finally fit the absorption peaks
for region, params, ffs, cut in zip(pregions, peak_parameters, fitfunc, absorption_cuts):
    for i, (p, r, c) in enumerate(zip(params, region, cut)):
        r.plot(x='freq', y='ch3')
        r = r.where((r['freq'] < c[1]) &
                    (r['freq'] > c[0]))
        r.plot(x='freq', y='ch3')
        plt.plot(r['freq'], ffs(r['freq'], *p[0]))
        r['ch3'] -= ffs(r['freq'], *p[0])
        r.plot(x='freq', y='ch3')
        plt.show()
