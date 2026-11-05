# -*- coding:utf-8 -*-

t_zero = 273.15  # 0°C in [K]
cp = 1005.  # specific heat of dry air, constant pressure [J K^-1 kg^-1]
cv = 718.  # sepcific heat of dry air, constant volume J K^-1 kg^-1
L = 2.501e6  # latent heat of vaporization at 0°C J kg^-1
R = 287.04  # gas constant of dry air J kg^-1 K^-1
Rv = 461.5  # gas constant for water vapor J kg^-1 K^-1
g = 9.81  # gravitational acceleration in m s^-2
rdv = R / Rv
kappa = (cp - cv) / cp
