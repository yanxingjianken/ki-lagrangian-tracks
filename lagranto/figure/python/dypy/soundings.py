# coding: utf-8

try:
    from urllib.request import urlopen
except:
    from urllib2 import urlopen
import numpy as np


def create_sounding_url(station, year, month, day, hour):
    '''
    create the url to acces sounding based on
    the University of Wyoming website:
        http://weather.uwyo.edu/upperair/sounding.html
    usage:
         create_url('06610', 1993, 12, 12, 12)

    return:
        url as string
    input:
    station = Number of the station
    year = yyyy
    month = mm
    day = dd
    hour = hh
    station numbers:
                    10410 Essen (D)
                    10618 Idar-Oberstein (D)
                    07145 Trappes (F)
                    06260 De Bilt (NL)
                    10238 Bergen (D)
                    06610 Payerne (CH)
                    16080 Milan (I)
                    10868 Munich (D)
     '''

    soundinginfo = {
        'station': station,
        'yyyy': year,
        'mm': month,
        'dd': day,
        'hh': hour,
        }

    template_url = 'http://weather.uwyo.edu/cgi-bin/sounding?TYPE=TEXT%3ALIST&'
    template_url += 'YEAR={yyyy}&MONTH={mm:02d}&FROM={dd:02d}{hh:02d}'
    template_url += '&STNM={station}'

    return template_url.format(**soundinginfo)


def get_sounding(file_or_url):
    '''
    input:
        file_or_url (string): the name of the sounding file
                              or the URL of the sounding file
    output:
        sounding (dictionary): dictionary containing the sounding data
    '''
    # load the data from a sounding, using the university of wyoming website
    try:
        soundingfile = urlopen(file_or_url).read()
    except:
        soundingfile = open(file_or_url, 'r').read()

    filelines = soundingfile.splitlines()

    # The first line is found by detecting the first line
    # where all entries are numbers
    i_firstrow = None
    for n, line in enumerate(filelines):
        line = line.split()
        if line:
            if line[0] == 'PRES':
                i_colnames = n
            cond1 = False not in [is_number(element) for element in line]
            cond2 = i_firstrow is None
            if cond1 and cond2:
                i_firstrow = n
            if '</PRE><H3>' in line:
                i_lastrow = n

    indices = range(7, 11*8, 7)
    indices.insert(0, 0)

    splitindices = [(a, a+7) for a in indices]
    columndict = dict((key, ncol) for ncol, key in
                      zip(splitindices, filelines[i_colnames].split()))

    variables = filelines[i_colnames].split()
    datadict = dict((key, []) for key in filelines[i_colnames].split()
                    if key in variables)

    for line in filelines[i_firstrow:i_lastrow]:
        for key in datadict:
            colstart, colend = columndict[key]
            data = line[colstart:colend]
            if data.isspace():
                data = 'nan'
            datadict[key].append(data)
    for key in datadict:
        datadict[key] = np.array(datadict[key], dtype=float)

    # convert wind from polar to carthesian coordinates
    if 'DRCT' in datadict:
        datadict['DRCT'] = datadict['DRCT']*np.pi/180
    if 'SKNT' in datadict and 'DRCT' in datadict:
        datadict['U'], datadict['V'] = (
            datadict['SKNT']*np.sin(-datadict['DRCT']),
            -datadict['SKNT']*np.cos(datadict['DRCT'])
        )

    return datadict
