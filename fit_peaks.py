import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.odr import Model, ODR, Data, RealData
from scipy.stats import norm
import numpy as np
# set the default figure size
plt.rcParams['figure.figsize'] = (14, 9)

# define the linear function to be fitted
def lin(b, x):
    """A linear function for fitting"""
    return x*b[0] + b[1]

def cflin(x, a, b):
    return x * a + b

def gaussian(b, x):
    """ A Gaussian function """
    return norm.pdf(x, loc=b[0], scale=b[1])*b[2]


def doubble_gaussian(x, mu1, mu2, sigma1, sigma2, A1, A2, offset):
    return A1*norm.pdf(x, loc=mu1, scale=sigma1) + A2*norm.pdf(x, loc=mu2,
                                                               scale=sigma2) + offset

def multiplicative_error_propagation(rel_err):
    return np.sqrt(sum(np.array(rel_err)**2))

def additive_error_propagation(errs):
    return np.sqrt(sum(np.array(errs)**2))

# here the different magic numbers are recorded, that I extracted from looking
# at the reduced dataset
show = False
r = pd.read_csv(sys.argv[1], index_col=0)
r = r.where(r['f'] > 0)
r = r.dropna(axis=0)
# now, to fit the falling ramp, the resonance absorption peaks have to be
# removed (the background is being fitted)
# the format is one tuple per peak with the
# (start_index_of_cut, end_index_of_cut), the first two peaks are in the
# first interval together
peak_cuts = [(-3e9, -0.5e9), (1e9, 2.2e9), (2.2e9, 2.5e9),
             (2.45e9, 4e9), (5e9, 6.8e9),
             (7.5e9, 9e9), (1.25e10, 1.4e10)]
# remove the peaks so that the ramp can be fitted
ramp = r.copy()
for peak in peak_cuts:
    ramp = ramp.where(
            (ramp['f'] < peak[0]) |
            (ramp['f'] > peak[1]))
    ramp = ramp.dropna(axis=0)
plt.errorbar(ramp['f'], ramp['ch3'], xerr=ramp['std_f'], yerr=ramp['std_ch3'], color='darkred', linestyle='', marker='')

# set up the objects required for orthogonal distance regression
linear = Model(lin)
data = RealData(ramp['f'], ramp['ch3'], sx=ramp['std_f'], sy=ramp['std_ch3'])
initial_param_guess = (-1e-9, 30)
regressor = ODR(data, linear, initial_param_guess)
popt = regressor.run()
odr_popt = popt.beta
odr_sd = popt.sd_beta
plt.plot(ramp['f'], lin(odr_popt, ramp['f']), linestyle='--',
         label='ramp fit', color='red')
plt.grid()
plt.legend()
plt.xlabel('f [Hz]')
plt.ylabel('U [V]')
plt.savefig('linear_fit_{}.pdf'.format(sys.argv[2]))
plt.close()
r['ch3'] = r['ch3'] - lin(odr_popt, r['f'])

# propagate the fit error of the slope into the points, use additive error
# propagation
# as the fitted value is subtracted from the original Data
r['std_ch3'] = np.sqrt(r['std_ch3']**2 +
                       np.sqrt(odr_sd[1])**2 +
                       (r['f'] * odr_sd[0])**2)
r['std_f'] = np.sqrt(r['std_f']**2 +
                     (odr_sd[1] * odr_sd[0])**2 +
                     (odr_sd[0] * np.abs(odr_popt[1] - r['ch3']))**2)


# the slope has been fitted and subtracted from the data the peaks of the
# non-absorption spectroscopy are now fitted
cut_in_peaks = [(1.4e9, 1.875e9), (2.7e9, 3.1e9), (5.7e9, 5.9e9), (8.2e9, 8.6e9)]
actual_peaks = [peak_cuts[1], peak_cuts[3], peak_cuts[4], peak_cuts[5]]
fitfunc = [gaussian, gaussian, gaussian, gaussian]
peak_p = []
for i, (pc, ffnc, cut) in enumerate(zip(actual_peaks, fitfunc,
                                               cut_in_peaks)):
    peak = r.copy().where(
        (r['f'] > pc[0]) & (r['f'] < pc[1]))
    peak = peak.dropna(axis=0)

    # now invert the peak
    peak['ch3'] = -peak['ch3']
    # now after narrowing to the peak region, as we want to fit without the
    # disturbances caused by the saturation absorbance, the tips of the
    # peaks are removed
    peak = peak.where((peak['f'] < cut[0]) | (peak['f'] > cut[1]))
    peak = peak.dropna(axis=0)

    # the starting parameters for the fit must be approximated
    width = peak['f'].max() - peak['f'].min()
    height = peak['ch3'].max()
    Area = width * height
    peak_loc = (peak['f'].max() + peak['f'].min())/2
    parameters = (peak_loc, width, Area/2)
    # fit Gaussian to peak
    gaus = Model(ffnc)
    data = RealData(peak['f'], peak['ch3'],
                    sx=peak['std_f'], sy=peak['std_ch3'])
    regressor = ODR(data, gaus, parameters)
    popt = regressor.run()
    odr_popt = popt.beta
    odr_sd = popt.sd_beta
    peak_p.append((odr_popt, odr_sd))
    plt_freq = np.linspace(peak['f'].min(), peak['f'].max(), 10000)
    plt.errorbar(peak['f'], peak['ch3'], xerr=peak['std_f'], yerr=peak['std_ch3'],
                 marker='', color='darkred', linestyle='')
    plt.plot(plt_freq, ffnc(odr_popt, plt_freq), color='red',
             linestyle='--', label='gaussian fit')
    plt.grid()
    plt.legend()
    plt.savefig('ramp_{}_peak_{}_fit.pdf'.format(sys.argv[2], i))
    plt.close()

# flip r
r['ch3'] *= (-1)
plt.errorbar(r['f'], r['ch3'], xerr=r['std_f'], yerr=r['std_ch3'], color='darkred', marker='', linestyle='', label='CH3')
plt.grid() 
plt.xlabel('f [Hz]')
plt.ylabel('U [V]')
plt.savefig('data_{}_without_slope.pdf'.format(sys.argv[2]))
plt.close()

plt.errorbar(r['f'], r['ch3'], xerr=r['std_f'], yerr=r['std_ch3'], color='darkred', label='Data')
for i, (p, ffs) in enumerate(zip(peak_p, fitfunc)):
    # now remove the peaks from the data and
    plt.plot(r['f'], ffs(p[0], r['f']), label='fit of Peak {}'.format(i))
plt.grid()
plt.xlabel('f [Hz]')
plt.ylabel('-U [V]')
plt.legend()
plt.savefig('peaks_of_measurement_{}.pdf'.format(sys.argv[2]))
plt.close()

mu_0 = peak_p[0][0][0]
dmu_0 = peak_p[0][1][0]
for i, p in enumerate(peak_p[1:]):
    delta_mu = p[0][0] - mu_0
    delta_d = np.sqrt(p[1][0]**2 + dmu_0**2)
    mu_0 = p[0][0]
    dmu_0 = p[1][0]
    print("{} +- {}".format(delta_mu, delta_d))

for i, (p, ffs) in enumerate(zip(peak_p, fitfunc)):
    r['ch3'] -= ffs(p[0], r['f'])
r['ch3'] *= (-1)
r.plot('f', 'ch3', label='CH3', color='darkred', linestyle='', marker='x')
plt.grid()
plt.title('Absorption Peaks')
plt.xlabel('f [Hz]')
plt.ylabel('-U [V]')
plt.legend()
plt.savefig('Absorption_peaks_data_{}.pdf'.format(sys.argv[2]))
plt.close()
r.to_csv('data_{}_absorption_data.csv'.format(sys.argv[2]))
