from utils import get_logger
from copy import deepcopy
import numpy as np

logger = get_logger(__name__)


class CONreaderVM:

    def __init__(self, file_name):
        """
        Reads in a con file and saves the curves grouped according to its corresponding slice, frame and place.
        Finds the tags necessary to calculate the volume metrics.
        """
        self.file_name = file_name
        self.container = []
        self.contours = None

        con_tag = "XYCONTOUR"  # start of the contour data
        stop_tag = "POINT"     # if this is available, prevents from reading unnecessary lines
        volumerelated_tags = [
            'Study_id=',
            'Field_of_view=',
            'Image_resolution=',
            'Slicethickness=',
            'Patient_weight=',
            'Patient_height',
            'Study_description=',
            'Patient_gender='
        ]

        self.volume_data = {
            volumerelated_tags[0]: None, 
            volumerelated_tags[1]: None, 
            volumerelated_tags[2]: None,
            volumerelated_tags[3]: None,
            volumerelated_tags[4]: None,
            volumerelated_tags[5]: None,
            volumerelated_tags[6]: None,
            volumerelated_tags[7]: None
        }

        con = open(file_name, 'rt')
        
        def find_volumerelated_tags(line):
            for tag in volumerelated_tags:
                if line.find(tag) != -1:
                    value = line.split(tag)[1]  # the place of the tag will be an empty string, second part: value
                    self.volume_data[tag] = value
        
        def mode2colornames(mode):
            if mode == 0:
                return 'ln'   # left (endo)
            elif mode == 1:
                return 'lp'   # left (epi) contains the myocardium
            elif mode == 2:
                return 'rp'   # right (epi)
            elif mode == 5:
                return 'rn'   # right (endo)
            else:
                logger.warning('Unknown mode {}'.format(mode))
                return 'other'

        def find_xycontour_tag():
            line = con.readline()
            find_volumerelated_tags(line)
            while line.find(con_tag) == -1 and line.find(stop_tag) == -1 and line != "":
                line = con.readline()
                find_volumerelated_tags(line)
            return line

        def identify_slice_frame_mode():
            line = con.readline()
            splitted = line.split(' ')
            return int(splitted[0]), int(splitted[1]), mode2colornames(int(splitted[2]))

        def number_of_contour_points():
            line = con.readline()
            return int(line)

        def read_contour_points(num):
            contour = []
            for _ in range(num):
                line = con.readline()
                xs, ys = line.split(' ')
                contour.append((float(xs), float(ys)))  # unfortubately x and y are interchanged
            return contour

        line = find_xycontour_tag()
        while line.find(stop_tag) == -1 and line != "":

            slice, frame, mode = identify_slice_frame_mode()
            num = number_of_contour_points()
            contour = read_contour_points(num)
            self.container.append((slice, frame, mode, contour))
            line = find_xycontour_tag()

        con.close()
        return

    def get_hierarchical_contours(self):
        # if it is not initializedyet, then create it
        if self.contours is None:

            self.contours = {}
            for item in self.container:
                slice = item[0]
                frame = item[1]   # frame in a hearth cycle
                mode = item[2]    # mode can be red, green, yellow
                contour = item[3]

                # rearrange the contour
                d = {'x': [], 'y': []}
                for point in contour:
                    d['x'].append(point[0])
                    d['y'].append(point[1])

                if not(slice in self.contours):
                    self.contours[slice] = {}

                if not(frame in self.contours[slice]):
                    self.contours[slice][frame] = {}

                if not(mode in self.contours[slice][frame]):
                    x = d['x']
                    y = d['y']
                    N = len(x)
                    contour_mtx = np.zeros((N, 2))
                    contour_mtx[:, 0] = np.array(x)
                    contour_mtx[:, 1] = np.array(y)
                    self.contours[slice][frame][mode] = contour_mtx

        return self.contours

    def contour_iterator(self, deep=True):
        self.get_hierarchical_contours()
        for slice, frame_level in self.contours.items():
            for frame, mode_level in frame_level.items():
                if deep:
                    mode_level_cp = deepcopy(mode_level)
                else:
                    mode_level_cp = mode_level
                yield slice, frame, mode_level_cp

    def get_volume_data(self):
        # process field of view
        fw_string = self.volume_data['Field_of_view=']
        sizexsize_mm = fw_string.split('x')  # variable name shows the format
        size_h = float(sizexsize_mm[0])
        size_w = float(sizexsize_mm[1].split(' mm')[0])  # I cut the _mm ending

        # process image resolution
        img_res_string = self.volume_data['Image_resolution=']
        sizexsize = img_res_string.split('x')
        res_h = float(sizexsize[0])
        res_w = float(sizexsize[1])

        # process slice thickness
        width_string = self.volume_data['Slicethickness=']
        width_mm = width_string.split(' mm')
        width = float(width_mm[0])

        # process weight
        weight_string = self.volume_data['Patient_weight=']
        weight_kg = weight_string.split(' kg')
        weight = float(weight_kg[0])

        # process height
        # Unfortunately, patient height is not always available.
        # Study description can help in that case but its form changes heavily.
        if 'Patient_height=' in self.volume_data.keys():  
            height_string = self.volume_data['Patient_height=']
            height = height_string.split(" ")[0]
        else:
            height_string = str(self.volume_data['Study_description='])
            height = ''
            for char in height_string:
                if char.isdigit():
                    height += char
        if height == '':
            logger.warning('Unknown height in con file {}'.format(self.file_name))
            height = 178
        else:
            try:
                height = float(height)
            except ValueError:
                height = 178
                logger.error(' Wrong height format in con file {}'.format(self.file_name))

        # gender
        gender = self.volume_data['Patient_gender=']
        
        return (size_h/res_h, size_w/res_w), width, weight, height, gender
