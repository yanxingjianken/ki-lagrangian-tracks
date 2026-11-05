# -*- coding:utf-8 -*-
import cartopy
import copy
import math
import numpy as np
from pyproj import Geod
from datetime import timedelta

from dypy.intergrid import Intergrid
from dypy.constants import rdv, L, R, cp, t_zero

__all__ = ['CrossSection', 'interpolate', 'rotate_points', 'interval',
           'dewpoint', 'esat', 'moist_lapse']


class CrossSection(object):
    """
    Create a cross-section
    return the distance, the pressure levels,
    and the variables.
    """

    def __init__(self, variables, coo, pressure,
                 int2p=False, pollon=-170, pollat=43, nbre=1000, order=1):
        """
            variables: dictionary of variables {'name': np.array},
                need to contain rlat, rlon
            coo: list of coordinates [(startlon, startlat), (endlon, endlat)]
            pressure: pressure coordinate 1d np.array
            int2p: True if variables need to be interpolated,
                require P in the variables
            pollon: pole_longitude of the rotated grid
            pollat: pole_latitude of the rotated grid
            nre: nbre of points along the cross-section
            order: order of interpolation see for details
        """
        variables = copy.deepcopy(variables)
        # test the input
        required = ['rlon', 'rlat']

        if int2p:
            required.append('p')
        for var in required:
            if var not in variables.keys():
                raise Exception('{} not in variables dictionary'.format(var))

        rlon = variables.pop('rlon')
        rlat = variables.pop('rlat')
        self.coo = coo
        self.nbre = nbre
        self.pressure = pressure
        self.nz = pressure.size
        if int2p:
            p = variables['p']

        # find the coordinate of the cross-section
        self.lon, self.lat, crlon, crlat, self.distance = find_cross_points(
            coo, nbre, pollon, pollat)

        self.distances = np.linspace(0, self.distance, nbre)

        # get the information for intergrid
        self._lo = np.array([rlat.min(), rlon.min()])
        self._hi = np.array([rlat.max(), rlon.max()])
        self._query_points = [[lat, lon] for lat, lon in zip(crlat, crlon)]

        if int2p:
            for var, data in variables.items():
                if data.ndim > 2:
                    dataint = interpolate(data, p, pressure)
                    variables[var] = dataint

        variables = self._int2cross(variables)
        self.__dict__.update(variables)

    def _int2cross(self, variables):
        for var, data in variables.items():
            if data.squeeze().ndim > 2:
                datacross = np.zeros((self.nz, self.nbre))
                for i, datah in enumerate(data.squeeze()):
                    intfunc = Intergrid(datah, lo=self._lo, hi=self._hi,
                                        verbose=False, order=1)
                    datacross[i, :] = intfunc.at(self._query_points)
            else:
                datacross = np.zeros((self.nbre, ))
                intfunc = Intergrid(data.squeeze(), lo=self._lo, hi=self._hi,
                                    verbose=False, order=1)
                datacross[:] = intfunc.at(self._query_points)
            variables[var] = datacross
        return variables


def find_cross_points(coo, nbre, pole_longitude, pole_latitude):
    """ give nbre points along a great circle between coo[0] and coo[1]
        iand rotated the points
    """

    g = Geod(ellps='WGS84')
    cross_points = g.npts(coo[0][0], coo[0][1], coo[1][0], coo[1][1], nbre)
    lat = np.array([point[1] for point in cross_points])
    lon = np.array([point[0] for point in cross_points])
    rlon, rlat = rotate_points(
        pole_longitude, pole_latitude, lon, lat)
    distance = great_circle_distance(coo[0][0], coo[0][1],
                                     coo[1][0], coo[1][1])
    return lon, lat, rlon, rlat, distance


def great_circle_distance(lon1, lat1, lon2, lat2):
    """
    return the distance (km) between points following a great circle

    based on : https://gist.github.com/gabesmed/1826175

    >>> great_circle_distance(0, 55, 8, 45.5)
    1199.13065955064
    """

    EARTH_CIRCUMFERENCE = 6378137     # earth circumference in meters

    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat / 2) * math.sin(dLat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dLon / 2) * math.sin(dLon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = EARTH_CIRCUMFERENCE * c

    return distance/1000


def interpolate(data, grid, interplevels):
    """
    intrerpolate data to grid at interplevels
    data and grid are numpy array in the form (z,lat,lon)
    """
    data = data.squeeze()
    grid = grid.squeeze()
    shape = list(data.shape)
    if (data.ndim > 3) | (grid.ndim > 3):
        message = "data and grid need to be 3d array"
        raise IndexError(message)

    try:
        nintlev = len(interplevels)
    except:
        interplevels = [interplevels]
        nintlev = len(interplevels)
    shape[-3] = nintlev

    outdata = np.ones(shape)*np.nan
    if nintlev > 20:
            for idx, _ in np.ndenumerate(data[0]):
                column = grid[:, idx[0], idx[1]]
                column_GRID = data[:, idx[0], idx[1]]

                value = np.interp(
                    interplevels,
                    column,
                    column_GRID,
                    left=np.nan,
                    right=np.nan)
                outdata[:, idx[0], idx[1]] = value[:]
    else:
        for j, intlevel in enumerate(interplevels):
            for lev in range(grid.shape[0]):
                cond1 = grid[lev, :, :] > intlevel
                cond2 = grid[lev-1, :, :] < intlevel
                right = np.where(cond1 & cond2)
                if right[0].size > 0:
                    sabove = grid[lev, right[0], right[1]]
                    sbelow = grid[lev-1, right[0], right[1]]
                    dabove = data[lev, right[0], right[1]]
                    dbelow = data[lev-1, right[0], right[1]]
                    result = (intlevel-sbelow)/(sabove-sbelow) * \
                        (dabove-dbelow)+dbelow
                    outdata[j, right[0], right[1]] = result
    return outdata


def rotate_points(pole_longitude, pole_latitude, lon, lat, direction='n2r'):
    """
        rotate points from non-rotated system to rotated system
        n2r : non-rotated to rotated (default)
        r2n : rotated to non-rotated
        return rlon,rlat,
    """
    lon = np.array(lon)
    lat = np.array(lat)
    rotatedgrid = cartopy.crs.RotatedPole(
        pole_longitude=pole_longitude,
        pole_latitude=pole_latitude
        )
    standard_grid = cartopy.crs.Geodetic()

    if direction == 'n2r':
        rotated_points = rotatedgrid.transform_points(standard_grid, lon, lat)
    elif direction == 'r2n':
        rotated_points = standard_grid.transform_points(rotatedgrid, lon, lat)

    rlon, rlat, _ = rotated_points.T
    return rlon, rlat


def interval(starting_date, ending_date, delta=timedelta(hours=1)):
    curr = starting_date
    while curr < ending_date:
        yield curr
        curr += delta


def dewpoint(p, qv):
    """
    Calculate the dew point temperature based on p and qv following (eq.8):
    Lawrence, M.G., 2005. The Relationship between Relative Humidity
    and the Dewpoint Temperature in Moist Air:
    A Simple Conversion and Applications.
    Bulletin of the American Meteorological Society 86,
    225–233. doi:10.1175/BAMS-86-2-225
    """
    B1 = 243.04  # °C
    C1 = 610.94  # Pa
    A1 = 17.625

    e = p*qv/(rdv+qv)
    td = (B1*np.log(e/C1))/(A1-np.log(e/C1))

    return td


def esat(t):
    """
    Calculate the saturation vapor pressure for t in °C

    Following eq. 6 of
    Lawrence, M.G., 2005. The Relationship between Relative Humidity
    and the Dewpoint Temperature in Moist Air:
    A Simple Conversion and Applications.
    Bulletin of the American Meteorological Society 86,
    225–233. doi:10.1175/BAMS-86-2-225
    """
    C1 = 610.94  # Pa
    A1 = 17.625    #
    B1 = 243.03  # °C

    return C1*np.exp((A1 * t) / (B1 + t))


def moist_lapse(t, p):
    """ Calculates moist adiabatic lapse rate for T (Celsius) and p (Pa)
    Note: We calculate dT/dp, not dT/dz
    See formula 3.16 in Rogers&Yau for dT/dz, but this must be combined with
    the dry adiabatic lapse rate (gamma = g/cp) and the
    inverse of the hydrostatic equation (dz/dp = -RT/pg)"""

    a = 2. / 7.
    b = rdv * L * L / (R * cp)
    c = a * L / R
    e = esat(t)
    wsat = rdv * e / (p - e)  # Rogers&Yau 2.18
    numer = a * (t + t_zero) + c*wsat
    denom = p * (1 + b * wsat / ((t + t_zero)**2))

    return numer / denom
