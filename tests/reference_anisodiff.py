# SPDX-License-Identifier: GPL-3.0-or-later
# Frozen verbatim from tools.anisodiff at d7bbd46; pygeoflood authors,
# original diffusion algorithm from GeoNet. Test oracle only.
import numpy as np


def anisodiff(
    img: np.ndarray,
    niter: int,
    kappa: float,
    gamma: float,
    step: tuple[float, float] = (1.0, 1.0),
    option: str = "PeronaMalik2",
) -> np.ndarray:
    """
    Anisotropic diffusion.

    Parameters
    ----------
    img : `numpy.ndarray`
        Array of input image values.
    niter : `int`
        Number of iterations.
    kappa : `float`
        Edge threshold value.
    gamma : `float`
        Time increment.
    step : `tuple`, optional
        Step size. Default is (1.0, 1.0).
    option : `str`, optional
        Diffusion option. Default is "PeronaMalik2".

    Returns
    -------
    imgout : `numpy.ndarray`
        Array of filtered image values.
    """

    # initialize output array
    img = img.astype("float32")
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()
    for _ in range(niter):
        # calculate the diffs
        deltaS[:-1, :] = np.diff(imgout, axis=0)
        deltaE[:, :-1] = np.diff(imgout, axis=1)
        if option == "PeronaMalik2":
            # gS = gs_diff(deltaS,kappa,step1)
            # gE = ge_diff(deltaE,kappa,step2)
            gS = 1.0 / (1.0 + (deltaS / kappa) ** 2.0) / step[0]
            gE = 1.0 / (1.0 + (deltaE / kappa) ** 2.0) / step[1]
        elif option == "PeronaMalik1":
            gS = np.exp(-((deltaS / kappa) ** 2.0)) / step[0]
            gE = np.exp(-((deltaE / kappa) ** 2.0)) / step[1]
        # update matrices
        E = gE * deltaE
        S = gS * deltaS
        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't ask questions. just do it. trust me.
        # **above comments from original GeoNet code**
        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]
        # update the image
        mNS = np.isnan(NS)
        mEW = np.isnan(EW)
        NS[mNS] = 0
        EW[mEW] = 0
        NS += EW
        mNS &= mEW
        NS[mNS] = np.nan
        imgout += gamma * NS
    return imgout
