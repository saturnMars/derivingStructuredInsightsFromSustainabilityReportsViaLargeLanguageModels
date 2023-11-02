
import numpy as np

# SOURCE: https://github.com/aestrivex/bctpy/blob/master/bct/utils/visualization.py#L488
def reorder_matrix(m1, cost='line', verbose=False, H=1e4, Texp=10, T0=1e-3, Hbrk=10):
    '''
    This function rearranges the nodes in matrix M1 such that the matrix
    elements are squeezed along the main diagonal.  The function uses a
    version of simulated annealing.

    Parameters
    ----------
    M1 : NxN np.ndarray
        connection matrix weighted/binary directed/undirected
    cost : str
        'line' or 'circ' for shape of lattice (linear or ring lattice).
        Default is linear lattice.
    verbose : bool
        print out cost at each iteration. Default False.
    H : int
        annealing parameter, default value 1e6
    Texp : int
        annealing parameter, default value 1. Coefficient of H s.t.
        Texp0=1-Texp/H
    T0 : float
        annealing parameter, default value 1e-3
    Hbrk : int
        annealing parameter, default value = 10. Coefficient of H s.t.
        Hbrk0 = H/Hkbr

    Returns
    -------
    Mreordered : NxN np.ndarray
        reordered connection matrix
    Mindices : Nx1 np.ndarray
        reordered indices
    Mcost : float
        objective function cost of reordered matrix

    Notes
    -----
    Note that in general, the outcome will depend on the initial condition
    (the setting of the random number seed).  Also, there is no good way to
    determine optimal annealing parameters in advance - these paramters
    will need to be adjusted "by hand" (particularly H, Texp, and T0).
    For large and/or dense matrices, it is highly recommended to perform
    exploratory runs varying the settings of 'H' and 'Texp' and then select
    the best values.

    Based on extensive testing, it appears that T0 and Hbrk can remain
    unchanged in most cases.  Texp may be varied from 1-1/H to 1-10/H, for
    example.  H is the most important parameter - set to larger values as
    the problem size increases.  It is advisable to run this function
    multiple times and select the solution(s) with the lowest 'cost'.

    Setting 'Texp' to zero cancels annealing and uses a greedy algorithm
    instead.
    '''
    from scipy import linalg, stats
    n = len(m1)
    if n < 2:
        raise Exception("align_matrix will infinite loop on a singleton "
                            "or null matrix.")

    # generate cost function
    if cost == 'line':
        profile = stats.norm.pdf(range(1, n + 1), loc=0, scale=n / 2)[::-1]
    elif cost == 'circ':
        profile = stats.norm.pdf(
            range(1, n + 1), loc=n / 2, scale=n / 4)[::-1]
    else:
        raise Exception('cost must be line or circ')

    costf = linalg.toeplitz(profile, r=profile) * np.logical_not(np.eye(n))
    costf = costf / np.sum(costf)

    # establish maxcost, lowcost, mincost
    maxcost = np.sum(np.sort(costf.flat) * np.sort(m1.flat))
    lowcost = np.sum(m1 * costf) / maxcost
    mincost = lowcost

    # initialize
    anew = np.arange(n)
    amin = np.arange(n)
    h = 0
    hcnt = 0

    # adjust annealing parameters
    # H determines the maximal number of steps (user specified)
    # Texp determines the steepness of the temperature gradient
    Texp = 1 - Texp / H
    # T0 sets the initial temperature and scales the energy term (user provided)
    # Hbrk sets a break point for the stimulation
    Hbrk = H / Hbrk

    while h < H:
        h += 1
        hcnt += 1
        # terminate if no new mincost has been found for some time
        if hcnt > Hbrk:
            break
        T = T0 * Texp**h
        atmp = anew.copy()
        r1, r2 = np.random.randint(n, size=(2,))
        while r1 == r2:
            r2 = np.random.randint(n)
        atmp[r1] = anew[r2]
        atmp[r2] = anew[r1]
        costnew = np.sum((m1[np.ix_(atmp, atmp)]) * costf) / maxcost
        # annealing
        if costnew < lowcost or np.random.random_sample() < np.exp(-(costnew - lowcost) / T):
            anew = atmp
            lowcost = costnew
            # is this a new absolute best?
            if lowcost < mincost:
                amin = anew
                mincost = lowcost
                if verbose:
                    print('step %i ... current lowest cost = %f' % (h, mincost))
                hcnt = 0

    if verbose:
        print('step %i ... final lowest cost = %f' % (h, mincost))

    M_reordered = m1[np.ix_(amin, amin)]
    M_indices = amin
    cost = mincost
    return M_reordered, M_indices, cost