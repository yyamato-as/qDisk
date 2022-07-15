import numpy as np
import astropy.units as u
import astropy.constants as ac

def Keplerian_velocity(r, theta=0.0, Mstar=1.0, distance=None, incl=None):
    """Calculate Keplerian velocity at each (r, theta) point.

    Parameters
    ----------
    r : float or array_like
        Radial coordinate of the disk in au or arcsec (if in arcsec, *distance* is mandatory to convert *r* to au).
    theta : float or array_like, optional
        Azimuthal coordinate of the disk in radian, by default 0.0, i.e., along major axis
    Mstar : float, optional
        Central stellar mass in solar mass, by default 1.0
    distance : float, optional
        Distance to the central star, by default None, should set only when *r* is in arcsec
    incl : float, optional
        Disk inclination in degree, by default None, i.e., no inclination correction

    Returns
    -------
    float or array_like
        Keplerian velocity in km/s
    """
    if distance is not None:
        r *= distance  # in au

    r = abs(r) * u.au
    Mstar *= ac.M_sun

    vkep = np.sqrt(ac.G * Mstar / r) * np.cos(theta)
    if incl is not None:
        vkep *= np.sin(np.radians(incl))

    return vkep.to(u.km/u.s).value