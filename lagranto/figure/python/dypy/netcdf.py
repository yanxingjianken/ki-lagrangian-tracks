# coding:utf-8

"""
Module to read and write netcdf file.

Interface to netCDF4

"""

import sys
import netCDF4
import numpy as np


def read_var(filename, variables, index=slice(None)):
    """ Extract variables from a netCDF

        return a list of netCDF4 array
        with time as a datetime object if possible
        index should be a slice object like np.s_[0, :, 10]
    """
    if type(variables) is not list:
        variables = [variables]

    vararray = []

    def check_var(nvariables, variables):
        nv = set(nvariables)
        v = set(variables)
        ok = set(nv).issuperset(v)
        if not ok:
            diffvar = set(v).difference(nv)
            err = '{} not found in {}\n. Available :{}'.format(
                ",".join(diffvar), filename, ",".join(nvariables))
            raise Exception(err)

    with netCDF4.MFDataset(filename) as ncfile:
        check_var(list(ncfile.variables.keys()), variables)
        for var in variables:
            if var == 'time':
                time = ncfile.variables[var]
                try:
                    vardata = netCDF4.num2date(time[:], units=time.units)
                except AttributeError:
                    vardata = np.array(time)
            else:
                vardata = ncfile.variables[var][index]
            vararray.append(vardata.squeeze())

    return vararray


def read_gattributes(filename):
    """ Read global attributes from a netCDF"""
    gattributes = netCDF4.Dataset(filename).__dict__
    return gattributes


def read_dimensions(filename):
    """ Read dimensions"""
    dimensions = netCDF4.Dataset(filename).dimensions
    return dimensions


def read_variables(filename):
    """ Read variables"""
    variables = list(netCDF4.Dataset(filename).variables.keys())
    return variables


def create(outname, dimensions, gattributes):
    """ create a netCDF """
    ncfile = netCDF4.Dataset(outname, 'w', format='NETCDF3_CLASSIC')
    for dim in dimensions:
        ncfile.createDimension(dim[0], dim[1])
    ncfile.setncatts(gattributes)
    return ncfile


def addvar(ncfile, varname, vardata, dimensions):
    """ Add a variable to a netcdf"""
    var = ncfile.createVariable(varname, vardata.dtype, dimensions)
    try:
        var[:] = vardata
    except Exception as e:
        sys.stderr.write('{0} in addvar()\n'.format(e))
        sys.exit(1)
