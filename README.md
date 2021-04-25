Laser-Spektroskopy
---

Here I describe the basic things that are going to be done with the data before fitting. So that the process is somewhat understandable/repeatable.

## Formatting
Turns out, the TEK scope prints plain ASCII and built in some strange artefacts somewhere in the middle of the file. As pandas can read in plain ASCII
no change in formatting was done. The section with the strange artefacts was removed by hand using a text editor. As it turns out some measurements
where mangled, even though that should not be too much of a problem.

## The missing Data
I accidentally overwrote the set of scans that we made initially. I only have a few curves left that actually depict the absorbed intensity. As I do
have a set of two 'high res' scans, it may be possible to fit the 'non saturation absorption peaks anyway with good precision.
This however means that the data be combined and properly labled and calibrated first. So this shall be the first step.

There are many frequency scans and one entire ramp. As the hight of the ramp corresponds to the frequency and I assume, that all plots are from the
absorption spektroskopy, the voltage of the slope corresponds to a frequency. With this information the frequency can be inferred in every peice of
data measured.

As it turns out the scope records the time relative to the trigger. So the time can be correlated with the frequency very nicely and stays valid
for each csv file.

## Reducing the resolution
Thankfully, the data fits into ram without any problems, and after conversion everything works fine.
The voltage resolution of the data is pretty shoddy though. As a remedy, an averaging filter can be used and the measurements devided uinto reasonable
subsections that then are used to average out the measurements. Also a standard deviation can be computed that scales with sqrt of the number of
measurements.

## Calibrating the frequency
Also the Scope seemingly uses the trigger to synchronize the time of each event, and as we did not touch the trigger, the time can be directly
translated into frequency. For checking this, there are multiple scans of the resonance peaks.


