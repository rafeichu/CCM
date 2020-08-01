# DEPENDENCIES
from image_utils import subtract_mean, adjust_gamma, mean_channels

# Others
# ----------------------------------------------------------

# OpenCV
import cv2 as cv

# Numpy
import numpy as np

# Scipy
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# Matplotlib, for drawing auxiliary graphs when debugging
import matplotlib.pyplot as plt

# MAIN CLASSES
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------


def model(x, a1, a2, a3, a4):
    """
    Model provided by the paper. Equation (4).
    Args:
        x:  input values
        a1, a2, a3, a4: coefficients to be minimized
    Returns:
        Estimation
    """
    return a1 + a2 * x + a3 * np.sin(x) + a4 * np.exp(2*x)


def lineal(x, a1, a2, a3, a0):
    return a1*x[0] + a2*x[1] + a3*x[2] + a0


# def cuadratic(x, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10):
def cuadratic(x, a1, a2, a3, a4, a5, a6, a7):
    return a1*x[0] + a2*x[1] + a3*x[2] + a4*x[0]**2 + a5*x[1]**2 + a6*x[2]**2 + a7
           # + a8*x[0]*x[1] + a9*x[0]*x[2] + a10*x[1]*x[2]


class TransformBased():
    """
    Based on 'Color Correction System Using a Color Compensation Chart for the Images from Digital Camera'
    by Seok-Han Lee, Sang-Won Um, and Jong-Soo Choi
    """
    def __init__(self, Nh: int, Nv: int):
        super().__init__()

        self._Nh = Nh     # number of patches in the horizontal direction
        self._Nv = Nv     # number of patches in the vertical direction
        self._TRC = []    # TRC coefficients
        self._LUT_output = []    # to optimize TRC application
        self._M = None    # Matrix transformation (done in XYZ color space)
        pass

    def _brightness_compensation(self,
                                observed_colors: np.ndarray,
                                reference_colors: np.ndarray) -> np.ndarray:

        """
        Section 3.1.1 "Brightness Compensation of the Chart".

        Args:
            self:
            observed_colors: chart colors found in target image
            reference_colors:  chart colors defined on source/reference image

        Returns:
            brightness compensated image.
        """
        # for every color in the chart on the target image, convert from BGR to HLS
        observed_HLS = cv.cvtColor(observed_colors, cv.COLOR_BGR2HLS)
        observed_L = observed_HLS[:, :, 1]     # use L (lightness) channel only
        observed_L_corners = self._get_corner_colors(observed_L)

        # for corner colors in the chart on the source image, convert from BGR to HLS
        # use L (lightness) channel only
        reference_L = cv.cvtColor(reference_colors, cv.COLOR_BGR2HLS)[:, :, 1]
        reference_L_corners = self._get_corner_colors(reference_L)

        # TODO: check dtype
        # get the smallest brightness difference.
        diff = np.subtract(observed_L_corners, reference_L_corners, dtype=np.int16)   # substraction itself
        # indexes = np.unravel_index(np.argmin(np.abs(diff), axis=None), diff.shape)    # get indexes of min abs diff
        # Ds = diff[indexes[0], indexes[1]]                                       # in order to preserve sign

        # adjust brightness in the corners
        # Ps = observed_L_corners.astype(np.int16) - Ds
        # Ps = np.clip(Ps, 0, 255)    # to prevent from overflowing

        # get remaining colors adjustment by bilinear interpolation
        # TODO: optimization hint: f could be smaller
        f = np.zeros((self._Nv, self._Nh))
        for i in range(self._Nv):
            # f[i][0] = Ps[1][0] * i/(self._Nv - 1) + Ps[0][0] * (self._Nv - 1 - i)/(self._Nv - 1)
            f[i][0] = diff[1][0] * i/(self._Nv - 1) + diff[0][0] * (self._Nv - 1 - i)/(self._Nv - 1)
            # f[i][self._Nh - 1] = Ps[1][1] * i/(self._Nv - 1) + Ps[0][1] * (self._Nv - 1 - i)/(self._Nv - 1)
            f[i][self._Nh - 1] = diff[1][1] * i/(self._Nv - 1) + diff[0][1] * (self._Nv - 1 - i)/(self._Nv - 1)

        g = np.zeros((self._Nv, self._Nh))
        # TODO: optimization hint: further adjustment might not be needed in the corners
        for i in range(self._Nv):
            for j in range(self._Nh):
                g[i][j] = f[i][0] * (self._Nh - 1 - j)/(self._Nh - 1) + f[i][self._Nh - 1] * j/(self._Nh - 1)

        # compute brightness correction for every color in the target chart
        corrected_brightness = np.subtract(observed_L, g.astype(np.int16), dtype=np.int16) # TODO: CHECK THIS STEP
        corrected_brightness = np.clip(corrected_brightness, 0, 255)  # https://stackoverflow.com/questions/33382466/opencv-hls-color-space-range

        # convert back to bgr
        observed_HLS[:, :, 1] = corrected_brightness   # correction was only made in lightness channel
        corrected_brightness_bgr = cv.cvtColor(observed_HLS, cv.COLOR_HLS2BGR)

        return corrected_brightness_bgr

    def _get_corner_colors(self, color_array: np.ndarray) -> np.ndarray:
        tl = color_array[0][0]
        tr = color_array[0][self._Nh - 1]
        bl = color_array[self._Nv - 1][0]
        br = color_array[self._Nv - 1][self._Nh - 1]

        return np.array([(tl, tr), (bl, br)])

    def _TRC_estimation(self,
                       observed_colors: np.ndarray,
                       reference_colors: np.ndarray):
        """
        Section 3.1.2 "Estimation of the Tone Reproduction Curves"

        Args:
            observed_colors: chart colors found in target image
            reference_colors:  chart colors defined on source/reference image

        Returns:
            Nothing.
        """

        # get normalized BGR values in the interval [0, 1]
        observed_colors = observed_colors[3:5, 1:5]/255
        reference_colors = reference_colors[3:5, 1:5]/255

        # include in both of the arrays, to force the curve to go through this points.
        fixed_points = np.array([0, 1])

        # perform LS minimization in each channel
        for i in range(3):
            y = reference_colors[:, :, i].flatten()
            y = np.concatenate((y, fixed_points), axis=0)
            x = observed_colors[:, :, i].flatten()
            x = np.concatenate((x, fixed_points), axis=0)
            # provide: fun2min, initial guess and input arrays
            TRC, _ = curve_fit(model, x, y, p0=[1, 1, 1, 1])
            self._TRC.append(TRC)

        self._build_TRC_LUT()

        return

    def _build_TRC_LUT(self):

        x = np.linspace(0, 1, 256)
        for coeff in self._TRC:
            table = model(x, coeff[0], coeff[1], coeff[2], coeff[3])
            self._LUT_output.append(table)

        self._LUT_output = np.dstack((self._LUT_output[0], self._LUT_output[1], self._LUT_output[2]))
        return

    def show_curves(self):
        x = np.linspace(0, 1, 256)
        plt.plot(x, self._LUT_output[0, :, 0])
        plt.plot(x, self._LUT_output[0, :, 1])
        plt.plot(x, self._LUT_output[0, :, 2])
        plt.show()
        return

    def show_CCM(self):
        print(self._M)
        return

    def _TRC_estimation2(self,
                        observed_colors: np.ndarray,
                        reference_colors: np.ndarray):

        observed_colors = observed_colors[3:5, 1:5] / 255
        reference_colors = reference_colors[3:5, 1:5] / 255

        # include in both of the arrays, to force the curve to go through this points.
        fixed_points = np.array([0, 1])

        n = x = np.linspace(0, 1, 256)
        # perform LS minimization in each channel
        for i in range(3):
            y = reference_colors[:, :, i].flatten()
            y = np.concatenate((y, fixed_points), axis=0)
            x = observed_colors[:, :, i].flatten()
            x = np.concatenate((x, fixed_points), axis=0)
            spline = interp1d(x, y, kind='cubic')
            self._LUT_output.append(np.array(spline(n)))

        self._LUT_output = np.dstack((self._LUT_output[0], self._LUT_output[1], self._LUT_output[2]))

        return

    def _get_matrix_transformation(self,
                        observed_colors: np.ndarray,
                        reference_colors: np.ndarray,
                        approx: str = 'lineal'):
        """
        Gets matrix transformation by LS minimization.

        Args:
            observed_colors: chart colors found in target image
            reference_colors:  chart colors defined on source/reference image

        Returns:
            Nothing.
        """

        # XYZ color conversion
        observed_XYZ = cv.cvtColor(observed_colors, cv.COLOR_BGR2XYZ)
        observed_XYZ = np.reshape(observed_XYZ, (observed_XYZ.shape[0] * observed_XYZ.shape[1],
                                                 observed_XYZ.shape[2]))

        reference_XYZ = cv.cvtColor(reference_colors, cv.COLOR_BGR2XYZ)
        reference_XYZ = np.reshape(reference_XYZ, (reference_XYZ.shape[0] * reference_XYZ.shape[1],
                                                   reference_XYZ.shape[2]))

        # mean subtraction
        observed_XYZ = subtract_mean(observed_XYZ)
        reference_XYZ = subtract_mean(reference_XYZ)

        matrix = []

        if approx == 'lineal':
            p0 = [1, 1, 1, 1]
            func = lineal
        elif approx == 'cuadratic':
            p0 = [1, 1, 1, 1, 1, 1, 1]
            func = cuadratic

        for i in range(3):
            y = reference_XYZ[:, i].flatten()
            mat_col, _ = curve_fit(func, observed_XYZ.T, y, p0=p0)
            matrix.append(mat_col)

        self._M = np.array(matrix)
        return

    def _apply_profile(self, img: np.ndarray) -> np.ndarray:
        """
        Section 3.2 "Profile Application Process"

        Args:
            img: Cropped image to be corrected.

        Returns:
            Corrected image.
        """
        # Revert gamma compression
        img = adjust_gamma(img, gamma=1.7)

        # Apply TRC correction
        corrected_img = cv.LUT(img, self._LUT_output) * 255
        corrected_img = np.clip(corrected_img, 0, 255)

        # Apply color correction
        corrected_img = cv.cvtColor(corrected_img.astype(np.float32), cv.COLOR_BGR2XYZ)
        # corrected_img = cv.cvtColor(img.astype(np.float32), cv.COLOR_BGR2XYZ)
        # corrected_img = img

        corrected_img = corrected_img.reshape((corrected_img.shape[0]*corrected_img.shape[1], corrected_img.shape[2]))
        m = mean_channels(corrected_img)
        corrected_img = subtract_mean(corrected_img)

        # This next line should only be uncommented if method 3 and a cuadratic approx were previously used.
        v = np.ones((corrected_img.shape[0], 1), dtype=np.float32)
        # corrected_img = np.concatenate((corrected_img, v), axis=1)
        corrected_img = np.concatenate((corrected_img, np.square(corrected_img), v), axis=1)


        corrected_img = np.dot(self._M, corrected_img.T).T.reshape(img.shape)
        corrected_img = corrected_img + m
        corrected_img = cv.cvtColor(corrected_img.astype(np.float32), cv.COLOR_XYZ2BGR)
        corrected_img = np.clip(corrected_img, 0, 255)

        # Apply gamma
        corrected_img = adjust_gamma(corrected_img.astype(np.uint8), gamma=1/1.7)

        return corrected_img

    def _correct(self, img: np.ndarray,
                 observed_colors: np.ndarray,
                 reference_colors: np.ndarray) -> np.ndarray:
        """
        Color correction. Private Method. Does all the job.
        Args:
            img: Cropped image to be corrected.
            observed_colors: chart colors found in target image
            reference_colors:  chart colors defined on source/reference image
            params: -

        Returns:
            Fully corrected image.
        """
        # corrected_brightness_observed = self._brightness_compensation(observed_colors, reference_colors)
        self._TRC_estimation(observed_colors, reference_colors)
        self._get_matrix_transformation(observed_colors, reference_colors, approx='cuadratic')
        corrected_img = self._apply_profile(img)

        return corrected_img



