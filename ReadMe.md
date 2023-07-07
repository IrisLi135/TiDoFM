<p align="center">
    <img alt="TiDoFM" src="./TiDoFM_logo.png" style="width: 500px; height: 250px;">
</p>
<h1 align="center">Time Domain Fisher Matrix Code</h1>

<p align="center">
  A package calculating the Antenna Pattern, SNR, and the Fisher Matrix with time domain signals. 
</p>

## About

TiDoFM (Time Domain Fisher Matrix Code) includes three classes respectively in three files:

- TiDoFM/antenna_pattern_class.py: calculate the antenna pattern
- TiDoFM/fm_class.py: calculate the fisher matrix for input time domain signals
- TiDoFM/snr_class.py: calculate the snr for input time domain signals
## Documentation

Please click [**https://irisli135.github.io/docs/**](https://irisli135.github.io/docs/) for detailed information of this package.

## Usage
To use the code, you can import the necessary functions from the appropriate file. 

Example usage for calculating the ****Antenna Pattern****:
```python
from antenna_pattern_class import AntennaPattern
# Define source and detector parameters
source_params = {'ra': 0.5, 'dec': 0.3, 'pol': 0.0}
detector_params = {'Det': 'ET_1_10km_cryo'}
GPST = 1234567890.0
# Initialize the class
ap = AntennaPattern(source_params, detector_params, GPST)
# Calculate the antenna pattern
Fp, Fc = ap.AP()
```
Example usage for calculating the **Signal-to-Noise Ratio**:

```python
from snr_class import SNRCalculator
# Define source and detector parameters
source_params = {'Mtot': 2.8*SNRCalculator.MSUN, 'eta': 0.25, 'ra': 0.27, 'dec': 0.31, 'pol': 0.0, 'iota': 0.78}
detector_params = {'Det': 'ET_1_10km_cryo'}
GPST = 1234567890.0
# Initialize the class
snr = SNRCalculator(sample_rate = 4096, segment_size = 100, GPST = GPST, source_params = source_params, detector_params = detector_params)
# Note: You can simulate signals using other tools. The following are some test signals generated to demonstrate how to calculate the SNR.
simulated_signal = list(snr.generate_signal(duration = 2030))
hp_hc = {'hp':simulated_signal, 'hc': simulated_signal}
# Calculate the SNR
snr_info = snr.calculate_snr(signal = hp_hc)
# Print the results
print('time',snr_info['time_to_coalescence'],'SNR =', snr_info['snrs'])
```

Example usage for calculating the **Fisher Matrix**:
```python
from fm_class import FisherMatrix
delta_param = {'Mtot': 1e-6*lal.MSUN_SI, 'eta': 1e-6, 'ra': 1e-6}
instance_fm = fm_class.FisherMatrix(sample_rate = 4096, segment_size = 100, GPST = 1e9, source_params = {}, detector_params = {}, delta_params= delta_param)
# Note: You can simulate signals using other tools. The following are some test signals generated to demonstrate how to calculate the Fisher Matrix.
hp_sim_signal = instance_fm.generate_signal(duration=2030, amplitude=1e-23)
hc_sim_signal = instance_fm.generate_signal(duration=2030, amplitude=1e-23)
hp_M = instance_fm.generate_signal(duration=2030, amplitude=1e-23)
hc_M = instance_fm.generate_signal(duration=2030, amplitude=1e-23)
hp_eta = instance_fm.generate_signal(duration=2030, amplitude=1e-23)
hc_eta = instance_fm.generate_signal(duration=2030, amplitude=1e-23)
hp_ra = instance_fm.generate_signal(duration=2030, amplitude=1e-23)
hc_ra = instance_fm.generate_signal(duration=2030, amplitude=1e-23)
sig_dict = [{'hp':hp_M,'hc':hc_M},{'hp':hp_eta,'hc':hc_eta},{'hp':hp_ra,'hc':hc_ra},{'hp':hp_sim_signal,'hc':hc_sim_signal}]
info = instance_fm.FM(signal_dict = sig_dict)
print(info['fm'], info['time_to_coalescence'])
```
## Contributors:
- [Ik Siong Heng](mailto:ik.heng@glasgow.ac.uk): Provided guidance and mentorship.
- [Man Leong Chan](mailto:mervync@phas.ubc.ca): Set up the entire codebase, developed the initial functionality, and contributed to the ongoing development of the code.
- [Yufeng Li](mailto:yufengli@bnu.edu.cn): Expanded and optimized the codebase, improving its functionality and performance.

### Please cite the following papers if you publish results based on TiDoFM, Thanks!
1. [Man Leong Chan, Chris Messenger, Ik Siong Heng et al. (2018)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.97.123014)
2. [Yufeng Li, Ik Siong Heng, Man Leong Chan et al. (2022)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.105.043010)