## -*- coding: utf-8 -*-
## Wrap the antenna pattern function in a class
import lal
import numpy as np
"""Class description."""
class AntennaPattern:
    '''
    **This class calculates the antenna pattern of a given detector.**
    
    Example usage of AntennaPattern class
    
    >>> from antenna_pattern_class import AntennaPattern
    >>> # Define source and detector parameters
    >>> source_params = {'ra': 0.5, 'dec': 0.3, 'pol': 0.0}
    >>> detector_params = {'Det': 'ET_1_10km_cryo'}
    >>> GPST = 1234567890.0
    >>> # Initialize the class
    >>> ap = AntennaPattern(source_params, detector_params, GPST)
    >>> # Calculate the antenna pattern
    >>> Fp, Fc = ap.AP()
    >>> # Print the results
    >>> print('Fp =', Fp, 'Fc =', Fc)
    '''

    def __init__(self, source_params, detector_params, GPST):
        '''
        Initialize the class.

        :param source_params: A dictionary that contains the source parameters. The keys are 'ra', 'dec', 'pol', and the values are floats.
        :type source_params: dict
        :param detector_params: A dictionary that contains the detector parameters. The key is 'Det', and the value is a string that represents the detector name.
        :type detector_params: dict
        :param Default_source_params: {'ra': 0.27, 'dec': 0.31, 'pol': 0.0}. 
        :type Default_source_params: dict
        :param Default_detector_params: {'Det':'ET_1_10km_cryo'} 
        :type Default_detector_params: dict
        :param GPST: The GPS time in seconds.
        :type GPST: float
        :raises ValueError: If a given detector name is not in the list.
        '''
        self.ra = source_params.get('ra',0.27)
        self.dec = source_params.get('dec', 0.31)
        self.pol = source_params.get('pol', 0.0)
        self.Det = detector_params.get('Det', 'ET_1_10km_cryo')
        self.GPST = GPST
        self.detector_info = {
            'ET_1_10km_cryo': np.array([40.31, 9.25, 243.0, 90.0]) * np.pi / 180,
            'ET_2_10km_cryo': np.array([40.31, 9.25, 243.0, 90.0]) * np.pi / 180,
            'ET_3_10km_cryo': np.array([40.31, 9.25, 243.0, 90.0]) * np.pi / 180,}
        if self.Det not in self.detector_info:
            raise ValueError("Invalid detector name.")
            
    def AP(self):
        '''
        Return the antenna pattern.

        :return:
            - ``Fp``: The plus polarization of the antenna pattern.
            - ``Fc``: The cross polarization of the antenna pattern.

        '''
        Detectors = self.detector_info[self.Det]
        sx = np.cos(self.dec) * np.cos(self.ra)
        sy = np.cos(self.dec) * np.sin(self.ra)
        sz = np.sin(self.dec)
            
        GST = lal.GreenwichSiderealTime(self.GPST,0)
        lst = GST%(2*np.pi) + Detectors[1]
        temp = np.dot(self.rotation_matrix('z',lst), np.array([[sx],[sy],[sz]]))
        s_at_d = np.dot(self.rotation_matrix('y',Detectors[0]), temp)
        theta_d = np.pi/2.0 - np.arctan2(s_at_d[2], np.sqrt(s_at_d[1]**2.0 + s_at_d[0]**2.0))
        azimuth_d = np.arctan2(s_at_d[1], s_at_d[0])
        
        if self.Det.find('ET_1') != -1:
            azimuth_det = azimuth_d
        elif self.Det.find('ET_2') != -1:
            azimuth_det = azimuth_d + 2.0*np.pi/3.0
        elif self.Det.find('ET_3') != -1:
            azimuth_det = azimuth_d + 4.0*np.pi/3.0
        else:
            print('Input detector is not in the list!')
        
        Fp_1 = 1.0/2.0 * (1.0 + np.cos(theta_d) ** 2.0) * np.cos(2.0 * azimuth_det) * np.cos(2.0 * self.pol)
        Fp_2 = np.cos(theta_d) * np.sin(2.0 * azimuth_det) * np.sin(2.0 * self.pol)
            
        Fc_1 = 1.0/2.0 * (1.0 + np.cos(theta_d) ** 2.0) * np.cos(2.0 * azimuth_det) * np.sin(2.0 * self.pol)
        Fc_2 = np.cos(theta_d) * np.sin(2.0 * azimuth_det) * np.cos(2.0 * self.pol)     
               
        Fp = np.sqrt(3.0)/2.0 * (Fp_1 - Fp_2)
        Fc = np.sqrt(3.0)/2.0 * (Fc_1 + Fc_2)            

        return Fp, Fc
 

    def rotation_matrix(self, axis, angle):
        '''
        Return the rotation matrix around a given axis.

        :param axis: The axis of rotation. Only 'x', 'y', or 'z' are accepted.
        :type axis: str
        :param angle: The angle of rotation in radians.
        :type angle: float

        :return: The rotation matrix as a 3x3 nested list.

        :raises ValueError: If an invalid rotation axis is given.

        :Example: 

        >>> ap = AntennaPattern({'ra':0.27, 'dec':0.31, 'pol':0.0},
                            {'Det':'ET_1_10km_cryo'}, 1254355500.0)
        >>> ap.rotation_matrix('x', 0.5)
        [[1.0, 0.0, 0.0],
         [0.0, 0.8775825618903728, 0.479425538604203],
         [0.0, -0.479425538604203, 0.8775825618903728]]
        '''
        if axis == 'x':
            rot = np.array([[1, 0, 0],[0, np.cos(angle), np.sin(angle)],[0, -np.sin(angle), np.cos(angle)]]).tolist()
        elif axis == 'y':
            rot = np.array([[np.cos(angle), 0, -np.sin(angle)], [0, 1, 0], [np.sin(angle), 0, np.cos(angle)]]).tolist()
        elif axis == 'z':
            rot = np.array([[np.cos(angle), np.sin(angle), 0],[-np.sin(angle), np.cos(angle), 0], [0, 0, 1]]).tolist()
        else:
            raise ValueError("Invalid rotation axis.")
        return rot