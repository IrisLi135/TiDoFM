#-*-coding:utf-8-*-
import lal
import math
import pyfftw
import random
import numpy as np
import antenna_pattern_class
from scipy import interpolate
from scipy.interpolate import UnivariateSpline

class FisherMatrix:
    '''
    **This class implements a Fisher Matrix calculator for time domain gravitational wave signals.**

    Example usage of FisherMatrix class

    >>> from fm_class import FisherMatrix
    >>> delta_param = {'Mtot': 1e-6*lal.MSUN_SI, 'eta': 1e-6, 'ra': 1e-6}
    >>> instance_fm = fm_class.FisherMatrix(sample_rate = 4096, segment_size = 100, GPST = 1e9, source_params = {}, detector_params = {}, delta_params= delta_param)
    >>> # generate tested random signal
    >>> hp_sim_signal = instance_fm.generate_signal(duration=2030, amplitude=1e-23)
    >>> hc_sim_signal = instance_fm.generate_signal(duration=2030, amplitude=1e-23)
    >>> hp_M = instance_fm.generate_signal(duration=2030, amplitude=1e-23)
    >>> hc_M = instance_fm.generate_signal(duration=2030, amplitude=1e-23)
    >>> hp_eta = instance_fm.generate_signal(duration=2030, amplitude=1e-23)
    >>> hc_eta = instance_fm.generate_signal(duration=2030, amplitude=1e-23)
    >>> hp_ra = instance_fm.generate_signal(duration=2030, amplitude=1e-23)
    >>> hc_ra = instance_fm.generate_signal(duration=2030, amplitude=1e-23)
    >>> sig_dict = [{'hp':hp_M,'hc':hc_M},{'hp':hp_eta,'hc':hc_eta},{'hp':hp_ra,'hc':hc_ra},{'hp':hp_sim_signal,'hc':hc_sim_signal}]
    >>> info = instance_fm.FM(signal_dict = sig_dict)
    >>> print(info['fm'], info['time_to_coalescence'])
    '''
    # the solar mass unit
    MSUN = lal.MSUN_SI
    # the radius of the Earth
    RE = 6378.140e3

    def __init__(self, sample_rate, segment_size, GPST, source_params, detector_params, delta_params):
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
        :param Default_source_params: {'Mtot': 2.8*SNRCalculator.MSUN, 'eta': 0.25, 'ra': 0.27, 'dec': 0.31, 'pol': 0.0, 'iota': 0.78}
        :type Default_source_params: dict
        :param Default_detector_params: {'Det': 'ET_1_10km_cryo'}
        :type Default_detector_params: dict
        '''
        self.sample_rate = sample_rate
        self.segment_size = segment_size
        self.source_params = source_params
        self.Mtot = source_params.get('Mtot',2.8*FisherMatrix.MSUN)
        self.eta = source_params.get('eta', 0.25)
        self.ra = source_params.get('ra',0.27)
        self.dec = source_params.get('dec', 0.31)
        self.pol = source_params.get('pol', 0.0)
        self.iota = source_params.get('iota', 0.78)
        self.detector_params = detector_params
        self.Det = detector_params.get('Det', 'ET_1_10km_cryo')
        self.delta_params = delta_params
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
                'ET_1': np.array([40.31, 9.25, 243.0, 90.0])*np.pi/180,
                'ET_2': np.array([40.31, 9.25, 243.0, 90.0])*np.pi/180,
                'ET_3': np.array([40.31, 9.25, 243.0, 90.0])*np.pi/180,
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
        x = FisherMatrix.RE*np.cos(Detlocation[0])*np.cos(Detlocation[1])
        y = FisherMatrix.RE*np.cos(Detlocation[0])*np.sin(Detlocation[1])
        z = FisherMatrix.RE*np.sin(Detlocation[0])
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

    def FM(self, signal_dict):
        '''
        This function calculates the Fisher matrix.

        :param signal_dict: The dictionary of the gravitational wave signal.
        :type signal_dict: dict

        :return: The dictionary of time to coalescence and the Fisher matrix.
        '''
        segment_info = self.divide_signal(signal_dict[-1]['hp'])
        # initialize the snr and time to coalescence list
        pieces = len(segment_info)
        signal_length = len(signal_dict[-1]['hp'])/self.sample_rate

        time_to_coalescence = []
        GPSTinitial = self.GPST
        num_params = len(self.delta_params)
        # initialize fisher matrix
        fisher_matrix = np.zeros((num_params, num_params))
        wholeFM = np.array([np.array(np.zeros((num_params,num_params))) for i in range(pieces)])
        piece = 0

        for segment in segment_info:
            ## extract the info of the segment
            df = segment['df']
            dt = segment['dt']

            ## determine the frequency range
            f_series = np.arange(0, self.sample_rate/2, df)
            f_cut_start = self.freq(segment['start_time'], signal_length)-3*df
            f_cut_end = self.freq(segment['end_time'], signal_length)+3*df
            if segment == segment_info[-1]:
                f_cut_end = 220.0*(20.0*FisherMatrix.MSUN/self.Mtot)

            f_start_index = int(math.ceil(f_cut_start/df))-1
            f_end_index = int(math.ceil(f_cut_end/df))

            ## downsample the signal to reduce the computation time
            if self.sample_rate < 10*f_cut_end:
                down_sample_factor = 1
            else:
                down_sample_factor = math.floor(self.sample_rate/(10*f_cut_end))
            ## read noise
            noise_asd = self.readnos(f_series)
            ASD = noise_asd[f_start_index : f_end_index]

            ## calculate the GPST of the segment and the antenna pattern
            GPST_piece = GPSTinitial + segment['start_time']
            ap = antenna_pattern_class.AntennaPattern(source_params = self.source_params, detector_params = self.detector_params, GPST = GPST_piece)
            Fp, Fc = ap.AP()[0][0], ap.AP()[1][0]

            ## the signal for the current segment
            hp_segment_signal = signal_dict[-1]['hp'][segment['start_index']:segment['end_index']][::down_sample_factor]
            hc_segment_signal = signal_dict[-1]['hc'][segment['start_index']:segment['end_index']][::down_sample_factor]
            time_series = np.arange(len(hp_segment_signal)) * dt
            d_hat_ori = self.dhat(GPST_piece, hp_segment_signal, hc_segment_signal, Fp, Fc, dt, time_series)
            d_hat_total = []
            for num in range(num_params):
                param_name = list(self.delta_params.keys())[num]
                if param_name == 'ra':
                    ap= antenna_pattern_class.AntennaPattern(source_params = {'ra': self.ra+self.delta_params[param_name]}, detector_params = self.detector_params, GPST = GPST_piece)
                    Fp, Fc = ap.AP()[0][0], ap.AP()[1][0]
                elif param_name == 'dec':
                    ap= antenna_pattern_class.AntennaPattern(source_params = {'dec': self.dec+self.delta_params[param_name]}, detector_params = self.detector_params, GPST = GPST_piece)
                    Fp, Fc = ap.AP()[0][0], ap.AP()[1][0]
                elif param_name == 'pol':
                    ap= antenna_pattern_class.AntennaPattern(source_params = {'pol': self.pol+self.delta_params[param_name]}, detector_params = self.detector_params, GPST = GPST_piece)
                    Fp, Fc = ap.AP()[0][0], ap.AP()[1][0]
                else:
                    Fp, Fc = Fp, Fc
                hp_tem, hc_tem = signal_dict[num]['hp'][segment['start_index']:segment['end_index']][::down_sample_factor], signal_dict[num]['hc'][segment['start_index']:segment['end_index']][::down_sample_factor]

                d_hat = (self.dhat(GPST_piece, hp_tem, hc_tem, Fp, Fc, dt, np.arange(len(hp_tem)) * dt)-d_hat_ori)[f_start_index:f_end_index]/self.delta_params[param_name]/ASD[:len(d_hat_ori[f_start_index:f_end_index])]
                d_hat_total.append(d_hat)
            for i in range(num_params):
                for j in range(num_params):
                    fisher_matrix[i][j] += 2.0*df*(np.vdot(d_hat_total[i], d_hat_total[j]) + np.vdot(d_hat_total[j], d_hat_total[i]))

            wholeFM[piece] = fisher_matrix
            piece += 1
            time_to_coalescence.append(segment['end_time']-signal_length)

        info = {'fm': wholeFM, 'time_to_coalescence': time_to_coalescence}
        return info

    def generate_signal(self, duration=2030, amplitude=1e-23):
        '''
        This function generates a random signal.

        :param signal_duration: The duration of the signal.
        :type signal_duration: float
        :param amplitude: The amplitude of the signal.
        :type amplitude: float

        :return: The random signal.
        '''
        num_samples = duration * self.sample_rate
        signal = np.zeros(num_samples)
        for i in range(num_samples):
            signal[i] = random.uniform(-amplitude, amplitude)
        # generate a noise with mean 0 and standard deviation of amplitude/10
        noise = np.random.normal(loc=0, scale=amplitude/10, size=num_samples)
        # add the noise to the signal
        noisy_signal = signal + noise
        return noisy_signal
