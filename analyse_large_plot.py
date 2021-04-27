import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
import numpy as np

# define the linear function to be fitted
def lin(x, a, b):
    """A linear function for fitting"""
    return x*a + b


def gaussian(x, mu, sigma, A):
    return norm.pdf(x, loc=mu, scale=sigma)*A


def doubble_gaussian(x, mu1, mu2, sigma1, sigma2, A1, A2):
    return A1*norm.pdf(x, loc=mu1, scale=sigma1) + A2*norm.pdf(x, loc=mu2, scale=sigma2)


def lorenzian(x, gamma, x0, A, offset):
    return A/ ((x**2-x0**2)**2 + gamma**2*x0**2) + offset
# here the different magic numbers are recorded, that I extracted from looking
# at the reduced dataset

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

# there are three ramps and they can be transformed into three pictures of the same ramp
# for that the max and the min value are taken as cut points
ramp_0 = data.where(data['high_time'] < max_ramp_timestamp)
ramp_0 = ramp_0.dropna(axis=0)
print(ramp_0)

ramp_1 = data.where((data['high_time'] > max_ramp_timestamp)
                    & (data['high_time'] < min_ramp_timestamp))
ramp_1 = ramp_1.dropna(axis=0)
print(ramp_1)

ramp_2 = data.where(data['high_time'] > min_ramp_timestamp)
ramp_2 = ramp_2.dropna(axis=0)
print(ramp_2)

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
mean_spike_dist = []
spike_pos_std = []
spike_dist_std = []
for ramp in ramps:
    spike_sections = [(0.001, 0.003), (0.0035, 0.0055), (0.006, 0.008), (0.0085, 0.0105), (0.011, 0.013)]
    spike_width = 0.0003
    spike_params = []
    for spike in train:
        sd = data.loc[:, ['low_time', 'high_time', 'ch2', 'std_ch2']]
        sd = sd.where((sd['low_time'] > sd.loc[spike[0]]['low_time']) &
                      (sd['low_time'] < sd.loc[spike[1]]['low_time']))
        sd = sd.dropna(axis=0)

        mean_time = (sd['high_time'] + sd['low_time'])/2
        # the middle train needs to be mirrored
        mean_time = mean_time * mf
        gplot_time = np.linspace(min(mean_time), max(mean_time), 5000)

        spike_height = max(sd['ch2'])
        max_time = float(sd.where(sd['ch2'] == spike_height)['high_time'].dropna())

        plt.plot(mean_time, sd['ch2'], marker='.', color='blue', linestyle='')
        spopt, spcov = curve_fit(lorenzian, mean_time, sd['ch2'], p0=(spike_width, max_time, spike_width*spike_height/2, 0))
        spike_params.append((spopt, spcov))
        plt.plot(gplot_time, lorenzian(gplot_time, *spopt), linestyle='--', color='blue')
        plt.show()
    peak_t = []
    for param_set in spike_params:
        # this is the reference frequency
        peak_t.append(param_set[0][1])
        # this is the std_dev in the covariance matrix
        spike_pos_std.append(param_set[1][1][1])

    mean_spike_dist.append(np.mean(np.array(peak_t[1:]) - np.array(peak_t[:-1])))


# now, to fit the falling ramp, the resonance absorption peaks have to be
# removed (the background is being fitted)
# the format is one tuple per peak with the
# (start_index_of_cut, end_index_of_cut), the first two peaks are in the
# first interval together
peak_cuts = [(3800, 4500), (4825, 5175), (5400, 5700)]
params = [(2.8e-3, 3.7e-3, .001, .001, 6e-3, .012),
          (6.6e-3, 1.2e-3, 4.8e-3),
          (8.8e-3, 8e-4, 1.2e-3)]
cuts_in_peak_region = [11, 4.5, 1.3]
fitfunc = [doubble_gaussian, gaussian, gaussian]
data = pd.read_csv('reduced_data.csv')

# cut out the data
slope_fit_data = slope_fit_data.where((data['high_time'] > max_ramp_timestamp) &
                                      (data['high_time'] < min_ramp_timestamp))
for peak in peak_cuts:
    slope_fit_data = slope_fit_data.where(
            (data['high_time'] < data.loc[peak[0]]['high_time']) |
            (data['high_time'] > data.loc[peak[1]]['high_time']))
    slope_fit_data = slope_fit_data.dropna(axis=0)

slope_fit_data.plot(linestyle='', marker='x')
plt.show()

mean_time = (slope_fit_data['high_time'] + slope_fit_data['low_time'])/2
error_width = slope_fit_data['high_time'] - slope_fit_data['low_time']
plt.plot(mean_time, slope_fit_data['ch3'], linestyle='', marker='.')
lin_popt, pcov = curve_fit(lin, mean_time, slope_fit_data['ch3'],
                           sigma=slope_fit_data['std_ch3'])
plt.plot(mean_time, lin(mean_time, *lin_popt), linestyle='--',
         label='ramp fit', color='blue')
plt.show()

peak_parameters = []

for peak, ffnc, initial_params, cut_thresh in zip(peak_cuts, fitfunc,
                                                  params, cuts_in_peak_region):
    peak_1_data = data.loc[:, ['low_time', 'high_time', 'ch3', 'std_ch3']]
    peak_1_data = peak_1_data.where(
        (peak_1_data['high_time'] > peak_1_data.loc[peak[0]]['high_time']) &
        (peak_1_data['high_time'] < peak_1_data.loc[peak[1]]['high_time']))
    # now after narrowing to the peak region, as we want to fit without the
    # disturbances caused by the saturation absorbance, the tips of the peaks
    # are removed
    peak_1_data = peak_1_data.dropna(axis=0)
    mean_time = (peak_1_data['high_time'] + peak_1_data['low_time'])/2

    # correct for the slope that we have now fitted
    peak_1_data['ch3'] = peak_1_data['ch3'] - lin(mean_time, *lin_popt)

    # now invert the peak
    peak_1_data['ch3'] = -peak_1_data['ch3']

    # now after narrowing to the peak region, as we want to fit without the
    # disturbances caused by the saturation absorbance, the tips of the peaks
    # are removed
    peak_1_data = peak_1_data.where(peak_1_data['ch3'] < cut_thresh)
    peak_1_data = peak_1_data.dropna(axis=0)
    mean_time = (peak_1_data['high_time'] + peak_1_data['low_time'])/2
    plt.plot(mean_time, peak_1_data['ch3'], marker='.', linestyle='')
    plt.show()
    gplot_time = np.linspace(min(mean_time), max(mean_time), 5000)

    # fit gaussian to peak
    plt.plot(mean_time, peak_1_data['ch3'], marker='.', linestyle='')
    popt, pcov = curve_fit(ffnc, mean_time, peak_1_data['ch3'],
                           sigma=peak_1_data['std_ch3'], p0=initial_params)
    peak_parameters.append((popt, pcov))
    plt.plot(gplot_time, ffnc(gplot_time, *popt))
    plt.show()

# now that the parameters for the peaks have been fitted, thing left to do is
# to give the conversion factor of the time to a frequency
spike_trans = [[(0, 300), (550, 800), (1150, 1450), (1750, 2000), (2300, 2550)],
               [(3450, 3800), (4100, 4350), (4700, 4950), (5250, 5550), (5800, 6100)],
               [(6950, 7250), (7600, 7900), (8200, 8500), (8800, 9100), (9350, 9650)]]
mirror_factor = [1, -1, 1]
spike_width = 0.0003

spike_params = []
for train, mf in zip(spike_trans, mirror_factor):
    for spike in train:
        sd = data.loc[:, ['low_time', 'high_time', 'ch2', 'std_ch2']]
        sd = sd.where((sd['low_time'] > sd.loc[spike[0]]['low_time']) &
                      (sd['low_time'] < sd.loc[spike[1]]['low_time']))
        sd = sd.dropna(axis=0)

        mean_time = (sd['high_time'] + sd['low_time'])/2
        # the middle train needs to be mirrored
        mean_time = mean_time * mf
        gplot_time = np.linspace(min(mean_time), max(mean_time), 5000)

        spike_height = max(sd['ch2'])
        max_time = float(sd.where(sd['ch2'] == spike_height)['high_time'].dropna())

        plt.plot(mean_time, sd['ch2'], marker='.', color='blue', linestyle='')
        spopt, spcov = curve_fit(lorenzian, mean_time, sd['ch2'], p0=(spike_width, max_time, spike_width*spike_height/2, 0))
        spike_params.append((spopt, spcov))
        plt.plot(gplot_time, lorenzian(gplot_time, *spopt), linestyle='--', color='blue')
        plt.show()

# now the frequency spikes have been fitted (astoundingly well) the reference frequency
# can be selected. I will use the spike after the first absorption peak as reference of 0
