#-*-coding:utf-8-*-
import lal
import math
import pyfftw
import random
import numpy as np
import antenna_pattern_class
from scipy import interpolate
from scipy.interpolate import UnivariateSpline

class SNRCalculator:
    '''
    **This class implements an SNR (Signal-to-Noise Ratio) calculator for time domain gravitational wave signals.**

    Example usage of SNRCalculator class

    >>> from snr_class import SNRCalculator
    >>> # Define source and detector parameters
    >>> source_params = {'Mtot': 2.73*SNRCalculator.MSUN, 'eta': 0.249, 'ra': 2.16, 'dec': -0.4, 'pol': 0.0, 'iota': 0.34}
    >>> detector_params = {'Det': 'ET_1'}
    >>> GPST = 1187008882.4
    >>> # Initialize the class
    >>> snr = SNRCalculator(sample_rate = 4096, segment_size = 100, GPST = GPST, source_params = source_params, detector_params = detector_params)
    >>> # generate the tested gravitational wave signal
    >>> simulated_signal = list(snr.generate_signal(duration = 2030))
    >>> hp_hc = {'hp':simulated_signal, 'hc': simulated_signal}
    >>> # Calculate the SNR
    >>> snr_info = snr.calculate_snr(signal = hp_hc)
    >>> # Print the results
    >>> print('time',snr_info['time_to_coalescence'],'SNR =', snr_info['snrs'])
    '''
    # the solar mass unit
    MSUN = lal.MSUN_SI
    # the radius of the Earth
    RE = 6378.140e3

    def __init__(self, sample_rate, segment_size, GPST, source_params, detector_params):
        '''
        Initialize the class.

        :param sample_rate: The sampling rate of the gravitational wave data.
        :type sample_rate: float
        :param segment_size: The duration of each segment in seconds in which the data is analyzed. For example, we use 100 seconds.
        :type segment_size: float
        :param GPST: The GPS time of the start of the gravitational wave data.
        :type GPST: float
        :param source_params: A dictionary of parameters describing the gravitational wave source.
                                  - ``Mtot``: The total mass of the binary system emitting gravitational waves.
                                  - ``eta``: The symmetric mass ratio of the binary system emitting gravitational waves.
                                  - ``ra``: The right ascension of the source.
                                  - ``dec``: The declination of the source.
                                  - ``pol``: The polarization angle of the source.
                                  - ``iota``: The inclination angle of the binary system's orbital plane with respect to the line of sight.
        :type source_params: dict
        :param detector_params: A dictionary of parameters describing the gravitational wave detector used to collect the data.
                            - ``Det``: The name of the gravitational wave detector used to collect the data.
        :type detector_params: dict
        :param Default_source_params: {'Mtot': 2.73*SNRCalculator.MSUN, 'eta': 0.249, 'ra': 2.16, 'dec': -0.4, 'pol': 0.0, 'iota': 0.34}
        :type Default_source_params: dict
        :param Default_detector_params: {'Det': 'ET_1'}
        :type Default_detector_params: dict
        '''
        self.sample_rate = sample_rate
        self.segment_size = segment_size
        self.Mtot = source_params.get('Mtot',2.73*SNRCalculator.MSUN)
        self.eta = source_params.get('eta', 0.249)
        self.ra = source_params.get('ra',2.16)
        self.dec = source_params.get('dec', -0.4)
        self.pol = source_params.get('pol', 0.0)
        self.iota = source_params.get('iota', 0.34)
        self.Det = detector_params.get('Det', 'ET_1')
        self.GPST = GPST


    def divide_signal(self, signal):
        '''
        This function divides the gravitational wave data into segments of self.segment_size.

        :param signal: The gravitational wave data.
        :type signal: list
        :return: A list of dictionaries containing the start and end times and indices of each segment.
        '''
        num_samples = len(signal)
        segment_samples = self.segment_size * self.sample_rate
        df = 1 / self.segment_size
        dt = 1 / self.sample_rate

        num_segments = num_samples // segment_samples
        if num_samples % segment_samples > 0:
            num_segments += 1

        segment_info = []

        for i in range(num_segments):
            start_sample = i * segment_samples
            end_sample = min(start_sample + segment_samples, num_samples)
            start_time = start_sample / self.sample_rate
            end_time = end_sample / self.sample_rate

            segment_info.append({
                'start_time': start_time,
                'end_time': end_time,
                'start_index': start_sample,
                'end_index': end_sample,
                'df': df,
                'dt': dt
            })
        return segment_info

    def Ctime(self, M_tot, eta, F_min):
        '''
        This function calculates the time to coalescence of the binary system given its total mass,
        symmetric mass ratio, and minimum frequency.

        :param M_tot: The total mass of the binary system emitting gravitational waves.
        :type M_tot: float
        :param eta: The symmetric mass ratio of the binary system emitting gravitational waves.
        :type eta: float
        :param F_min: The minimum frequency of the gravitational wave signal.
        :type F_min: float
        :return: The time to coalescence of the binary system.
        '''
        # Define constants
        T_s = 4.925668152631749e-06
        # Calculate time to coalescence
        tau0 = 5.0 / (256.0 * (M_tot / self.MSUN) ** (5.0/3.0) * T_s ** (5.0/3.0) * (np.pi * F_min) ** (8.0/3.0) * eta)
        de2 = 64512.0 * eta * (M_tot/ self.MSUN) *(np.pi * F_min) ** 2.0
        tau2 = (3715.0 + 4620.0 * eta) / de2 / T_s
        tau3 = np.pi / (8.0 * eta * (M_tot/self.MSUN) ** (2.0/3.0) * (np.pi * F_min) ** (5.0/3.0)) / T_s ** (2.0/3.0)
        de4 = 128.0 * eta * (M_tot / self.MSUN * T_s) ** (1.0/3.0) * (np.pi * F_min) ** (4.0/3.0)
        no4 = 5.0 * (3058673.0 / 1016064.0 + 5429.0 * eta / 1008.0 + 617.0 * eta**2 / 144.0)
        tau4 = no4 / de4
        ToC = tau0 + tau2 - tau3 + tau4
        return ToC

    def freq(self, time_difference, length):
        '''
        This function calculates the frequency of the gravitational wave signal at a given time.

        :param time_difference: The time difference between the start of the gravitational wave data and the start of the segment.
        :type time_difference: float
        :param length: The length of the gravitational wave data.
        :type length: float

        :return: The frequency of the gravitational wave signal at a given time.
        '''
        frequencies = np.linspace(0.5, 2000, 15000)
        time_of_coalescence = np.zeros_like(frequencies)
        time_of_coalescence = self.Ctime(self.Mtot, self.eta, frequencies)

        frequencies = frequencies[time_of_coalescence > 0]
        time_of_coalescence = time_of_coalescence[time_of_coalescence > 0]
        interpolator = UnivariateSpline(time_of_coalescence[::-1], frequencies[::-1], w=1.0*np.ones_like(frequencies), s=0)

        interpolated_frequency = interpolator(length - time_difference)
        return interpolated_frequency

    def DetectorAngles(self, x):
        '''
        This function returns the detector info:
        [laltitude, longitude, orientation of arms, angle between arms].

        :param x: The name of the detector.
        :type x: str
        '''
        return {
                'ET_1': np.array([43.63, 10.5, 115.27, 90.0])*np.pi/180,
                'ET_2': np.array([43.63, 10.5, 115.27, 90.0])*np.pi/180,
                'ET_3': np.array([43.63, 10.5, 115.27, 90.0])*np.pi/180,
            }[x]

    def readnos(self, f_points):
        '''
        This function reads the noise curve of the detector corresponding to frequency series f_points.

        :param f_points: The frequency series.
        :type f_points: numpy.ndarray

        :return: The noise curve of the detector corresponding to frequency series f_points.
        '''
        def ASDtxt(x):
            return {
                'ET_1': 'ASD/ET_D.txt',
                'ET_2': 'ASD/ET_D.txt',
                'ET_3': 'ASD/ET_D.txt',
            }[x]

        nos_file = ASDtxt(self.Det)
        nos_divider = 1.0 / np.sqrt(3.0)

        with open(nos_file, 'r') as f:
            data = np.loadtxt(f)

        f = np.log10(data[:, 0])
        ASD = np.log10(data[:, 1] / nos_divider)

        nosextrapolate = interpolate.InterpolatedUnivariateSpline(f, ASD, k=1)
        nos = nosextrapolate(np.log10(f_points))
        return 10 ** nos

    def delay(self, GPST):
        '''
        This function calculates the time delay between detector and earth.

        :param GPST: The GPS time.
        :type GPST: float

        :return: The time delay between detector and earth.
        '''
        Detlocation = self.DetectorAngles(self.Det)
        x = SNRCalculator.RE*np.cos(Detlocation[0])*np.cos(Detlocation[1])
        y = SNRCalculator.RE*np.cos(Detlocation[0])*np.sin(Detlocation[1])
        z = SNRCalculator.RE*np.sin(Detlocation[0])
        tdelay = lal.TimeDelayFromEarthCenter([x, y, z], self.ra, self.dec, GPST)
        return tdelay

    def dhat(self, GPST, hp, hc, Fp, Fc, dt, t_earth):
        '''
        This function calculates the frequency domain strain.

        :param GPST: The GPS time.
        :type GPST: float
        :param hp: The plus polarization of the gravitational wave signal.
        :type hp: numpy.ndarray
        :param hc: The cross polarization of the gravitational wave signal.
        :type hc: numpy.ndarray
        :param Fp: The plus polarization of the antenna pattern.
        :type Fp: float
        :param Fc: The cross polarization of the antenna pattern.
        :type Fc: float
        :param dt: The time resolution.
        :type dt: float
        :param t_earth: The time series.
        :type t_earth: numpy.ndarray

        :return: The frequency domain strain.
        '''
        tdelay = self.delay(GPST)
        t_points = t_earth + tdelay
        hpf = interpolate.splrep(t_points, hp, w=1.0*np.ones(len(hp)), s=0)
        hcf = interpolate.splrep(t_points, hc, w=1.0*np.ones(len(hc)), s=0)
        hp_shifted = interpolate.splev(t_earth-t_earth[0], hpf, der = 0, ext = 1)
        hc_shifted = interpolate.splev(t_earth-t_earth[0], hcf, der = 0, ext = 1)
        N = np.size(hp)
        win = np.ones(N)
        h_shifted = win * (Fp * hp_shifted + Fc * hc_shifted)
        fftinput = pyfftw.empty_aligned(len(h_shifted), dtype='complex128')
        fft_object = pyfftw.builders.rfft(fftinput)
        d_hat = fft_object(h_shifted * dt)
        return d_hat

    def calculate_snr(self, signal):
        '''
        This function calculates the SNR of the signal.

        :param signal: The gravitational wave signal.
        :type signal: dict

        :return: a dict of (SNR, time to coalescence)
        '''
        hp_whole = signal['hp']
        hc_whole = signal['hc']
        duration = len(hp_whole)/self.sample_rate

        hp_segment_info = self.divide_signal(hp_whole)

        # initialize the snr and time to coalescence list
        snrs = []
        time_to_coalescence = []
        SNR = 0
        GPSTinitial = self.GPST
        for segment in hp_segment_info:
            ## extract the info of the segment
            df = segment['df']
            dt = segment['dt']

            ## determine the frequency range
            f_series = np.arange(0, self.sample_rate/2, df)
            f_cut_start = self.freq(segment['start_time'], duration)-3*df
            f_cut_end = self.freq(segment['end_time'], duration)+3*df
            if segment == hp_segment_info[-1]:
                f_cut_end = 220.0*(20.0*self.MSUN/self.Mtot)
            f_start_index = int(math.ceil(f_cut_start/df))-1
            f_end_index = int(math.ceil(f_cut_end/df))

            ## downsample the signal to reduce the computation time
            if self.sample_rate < 10*f_cut_end:
                down_sample_factor = 1
            else:
                down_sample_factor = math.floor(self.sample_rate/(10*f_cut_end))

            ## read noise
            noise_asd = self.readnos(f_series)
            PSD = noise_asd[f_start_index : f_end_index]**2

            ## calculate the GPST of the segment and the antenna pattern
            GPST_piece = GPSTinitial + segment['start_time']
            ap = antenna_pattern_class.AntennaPattern(source_params = {'ra': self.ra, 'dec': self.dec, 'pol': self.pol}, detector_params = {'Det': self.Det}, GPST = GPST_piece)
            Fp_0, Fc_0 = ap.AP()
            Fp = Fp_0[0]
            Fc = Fc_0[0]

            ## calculate the SNR of the segment
            hp_segment_signal = hp_whole[segment['start_index']:segment['end_index']][::down_sample_factor]
            hc_segment_signal = hc_whole[segment['start_index']:segment['end_index']][::down_sample_factor]
            time_series = np.arange(len(hp_segment_signal)) * dt
            d_hat = self.dhat(GPST_piece, hp_segment_signal, hc_segment_signal, Fp, Fc, dt, time_series)
            SNR = np.sqrt(SNR**2 + 4*sum((abs(d_hat[f_start_index:f_end_index])**2)*df/PSD[:len(d_hat[f_start_index:f_end_index])]))
            snrs.append(SNR)
            time_to_coalescence.append(segment['end_time']-duration)
        info = {'snrs': snrs, 'time_to_coalescence': time_to_coalescence}
        return info

    def generate_signal(self, signal_duration, amplitude=1e-23):
        '''
        This function generates a random signal.

        :param signal_duration: The duration of the signal.
        :type signal_duration: float
        :param amplitude: The amplitude of the signal.
        :type amplitude: float

        :return: The random signal.
        '''
        num_samples = signal_duration * self.sample_rate
        signal = []
        for i in range(num_samples):
            signal.append(random.uniform(-amplitude, amplitude))
        return signal
