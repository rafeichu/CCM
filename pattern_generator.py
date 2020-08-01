import os
import numpy as np
import cv2 as cv
import pandas as pd
import typing


class PatternGenerator:
    """This class generates an image with the desired color pattern"""

    def __init__(self, input: typing.Union[str, np.ndarray], colorspace_conv: int = None, rows: int = 1, cols: int = None,
                 color_size: int = 10):
        """
        Args:
            input: path to the csv that specifies the colors to be used
            rows: number of rows
            cols: number of columns
            colorspace_conv: OpenCV colorspace conversion code. e.g: cv.COLOR_HLS2BGR
            color_size: size of the color on the pattern.
        """
        if type(input) == str:
            self.input = input
            self.reference_colors = self._read_colors()
        elif type(input) == np.ndarray:
            self.reference_colors = input

        self.colorspace_conv = colorspace_conv
        self.rows = rows
        self.color_size = color_size

        if cols is None:
            self.cols = self.reference_colors.shape[1]
        else:
            self.cols = cols

        return

    def _read_colors(self) -> np.ndarray:
        """
        Returns:
            OpenCV structured array of colors
        """
        # searches for the requested file
        folder_data = os.path.dirname(__file__)
        path = os.path.normpath(os.path.join(folder_data, self.input))

        # grabs the colors from pandas dataframe
        df = pd.read_csv(path)
        return df.values

    def return_colors(self) -> np.ndarray:
        return scale_img(self.input, 1/self.color_size, 1/self.color_size)

    def run(self, action: str):
        """
            Does the job itself. Call this method once you've initialized the class.
        """
        # get pattern spacial distribution
        pattern = np.reshape(self.reference_colors, (self.rows, self.cols, 3))
        pattern = pattern.astype(np.float32)

        # convert to a specified color space if needed
        if self.colorspace_conv is None:
            pass
        else:
            pattern = cv.cvtColor(pattern, self.colorspace_conv) * 255

        # converts to desired size
        image = scale_img(pattern, self.color_size, self.color_size)
        image = image.astype(np.uint8)

        # writes image in the folder..
        if action == 'write':
            cv.imwrite('pattern.png', image)

        # returns pattern as an image..
        elif action == 'get':
            return image
        return


# generator = PatternGenerator('mobile_v3a_reference_colors.csv', cv.COLOR_LAB2BGR, rows=8, cols=6, color_size=50)
# generator = PatternGenerator('center.csv', None, rows=8, cols=6, color_size=50)
# generator.run()

def scale_img(img: np.ndarray, scale_percent_h: float, scale_percent_v: float) -> np.ndarray:
    """
    This function enlarges or shrinks the image symmetrically.
    Args:
        img: image to be modified.
        scale_percent_h: horizontal OUT/IN size ratio.
        scale_percent_v: vertical OUT/IN size ratio.

    Returns:
        Modified image.
    """
    width = int(img.shape[1] * scale_percent_h)
    height = int(img.shape[0] * scale_percent_v)
    dim = (width, height)
    resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    return resized
