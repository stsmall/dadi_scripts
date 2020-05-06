import numpy
from dadi import Numerics, PhiManip, Integration
from dadi.Spectrum_mod import Spectrum


def IMasym(params, ns, pts):
    """IM model
    To make this a pure isolation model, m = 0
    ns = (n1,n2)
    params = (s,nu1,nu2,T,m12,m21)
    Isolation-with-migration model with exponential pop growth.
    s: Size of pop 1 after split. (Pop 2 has size 1-s.)
    nu1: Final size of pop 1.
    nu2: Final size of pop 2.
    T: Time in the past of split (in units of 2*Na generations)
    m12: Migration from pop 2 to pop 1 (2*Na*m12)
    m21: Migration from pop 1 to pop 2
    n1,n2: Sample sizes of resulting Spectrum
    pts: Number of grid points to use in integration.

    """
    s, nu1, nu2, T, m12, m21 = params

    xx = Numerics.default_grid(pts)

    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)

    nu1_func = lambda t: s * (nu1/s)**(t/T)
    nu2_func = lambda t: (1-s) * (nu2/(1-s))**(t/T)
    phi = Integration.two_pops(phi, xx, T, nu1_func, nu2_func,
                               m12=m12, m21=m21)

    fs = Spectrum.from_phi(phi, ns, (xx, xx))
    return(fs)


def IMsym(params, ns, pts):
    """IM model
    To make this a pure isolation model, m = 0
    ns = (n1,n2)
    params = (s,nu1,nu2,T,m)
    Isolation-with-migration model with exponential pop growth.
    s: Size of pop 1 after split. (Pop 2 has size 1-s.)
    nu1: Final size of pop 1.
    nu2: Final size of pop 2.
    T: Time in the past of split (in units of 2*Na generations)
    m: Migration from between pops (2*Na*m)
    n1,n2: Sample sizes of resulting Spectrum
    pts: Number of grid points to use in integration.

    """
    s, nu1, nu2, T, m = params

    xx = Numerics.default_grid(pts)

    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)

    nu1_func = lambda t: s * (nu1/s)**(t/T)
    nu2_func = lambda t: (1-s) * (nu2/(1-s))**(t/T)
    phi = Integration.two_pops(phi, xx, T, nu1_func, nu2_func,
                               m12=m, m21=m)

    fs = Spectrum.from_phi(phi, ns, (xx, xx))
    return(fs)


def IMnogrowth_sym(params, ns, pts):
    """IM model with no growth and symmetrical migration
    To make this a pure isolation model, m = 0
    ns = (n1,n2)
    params = (nu1,nu2,T,m)
    Isolation-with-migration model with no exponential pop growth.
    nu1: Final size of pop 1.
    nu2: Final size of pop 2.
    T: Time in the past of split (in units of 2*Na generations)
    m: Migration from between pops (2*Na*m)
    n1,n2: Sample sizes of resulting Spectrum
    pts: Number of grid points to use in integration.

    """
    nu1, nu2, T, m = params

    xx = Numerics.default_grid(pts)

    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)

    phi = Integration.two_pops(phi, xx, T, nu1, nu2,
                               m12=m, m21=m)

    fs = Spectrum.from_phi(phi, ns, (xx, xx))
    return fs


def IMnogrowth_asym(params, ns, pts):
    """IM model with no growth and asymmetrical migration
    ns = (n1,n2)
    params = (nu1,nu2,T,m12,m21)
    Isolation-with-migration model with no exponential pop growth.
    nu1: Final size of pop 1.
    nu2: Final size of pop 2.
    T: Time in the past of split (in units of 2*Na generations)
    m12: Migration from pop 2 to pop 1 (2*Na*m12)
    m21: Migration from pop 1 to pop 2
    n1,n2: Sample sizes of resulting Spectrum
    pts: Number of grid points to use in integration.

    """
    nu1, nu2, T, m12, m21 = params

    xx = Numerics.default_grid(pts)

    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)

    phi = Integration.two_pops(phi, xx, T, nu1, nu2,
                               m12=m12, m21=m21)

    fs = Spectrum.from_phi(phi, ns, (xx, xx))
    return fs


def prior_onegrow_mig((nu1F, nu2B, nu2F, m, Tp, T), (n1, n2), pts):
    """
    Model with growth, split, bottleneck in pop2, exp recovery, migration

    nu1F: The ancestral population size after growth. (Its initial size is
          defined to be 1.)
    nu2B: The bottleneck size for pop2
    nu2F: The final size for pop2
    m: The scaled migration rate
    Tp: The scaled time between ancestral population growth and the split.
    T: The time between the split and present

    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = yy = Numerics.default_grid(pts)

    # phi for the equilibrium ancestral population
    phi = PhiManip.phi_1D(xx)
    # Now do the population growth event.
    phi = Integration.one_pop(phi, xx, Tp, nu=nu1F)

    # The divergence
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    # We need to define a function to describe the non-constant population 2
    # size. lambda is a convenient way to do so.
    nu2_func = lambda t: nu2B*(nu2F/nu2B)**(t/T)
    phi = Integration.two_pops(phi, xx, T, nu1=nu1F, nu2=nu2_func,
                               m12=m, m21=m)

    # Finally, calculate the spectrum.
    sfs = Spectrum.from_phi(phi, (n1, n2), (xx, yy))
    return sfs


def prior_onegrow_nomig((nu1F, nu2B, nu2F, Tp, T), (n1, n2), pts):
    """
    Model with growth, split, bottleneck in pop2, exp recovery, no migration

    nu1F: The ancestral population size after growth. (Its initial size is
          defined to be 1.)
    nu2B: The bottleneck size for pop2
    nu2F: The final size for pop2
    m: migration, here must be 0
    Tp: The scaled time between ancestral population growth and the split.
    T: The time between the split and present

    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    return prior_onegrow_mig((nu1F, nu2B, nu2F, 0, Tp, T), (n1, n2), pts)
