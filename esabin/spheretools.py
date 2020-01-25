import numpy as np

def angle_unit_as_radians(units):
    if units == 'radians':
        ang2rad = 1.
    elif units == 'degrees':
        ang2rad = np.pi/180.
    elif units == 'hours':
        ang2rad = np.pi/12
    else:
        raise ValueError(('Invalid angle unit {}'.format(units)
                          +' valid values are radians, degrees or hours'))
    return ang2rad

def angle_to_radians(angle,units):
    ang2rad = angle_unit_as_radians(units)
    return angle*ang2rad

def radians_to_angle(angle,units):
    rad2ang = 1./angle_unit_as_radians(units)
    return angle*rad2ang

def angle_difference(ang1,ang2,units):
    """Difference between two angles in degrees or hours (ang2-ang1),
    taking into account wrapping
    """
    ang1r = angle_to_radians(ang1,units)
    ang2r = angle_to_radians(ang2,units)
    y = np.sin(ang2r-ang1r)
    x = np.cos(ang2r-ang1r)
    angdiffr = np.arctan2(y,x)
    return radians_to_angle(angdiffr,units)

def angle_midpoint(ang1,ang2,units):
    """
    Midpoint between two angles in degrees or hours
    """
    return ang1 + angle_difference(ang1,ang2,units)/2.
