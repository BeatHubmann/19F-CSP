import numpy as np
import matplotlib.pyplot as plt
import sys
# from scipy.optimize import curve_fit

np.random.seed(42)

def nn_sum(x, i, j, k, L):
    """
    Args:
        x: Spin configuration
        i, j, k: Indices describing the position of one spin
        L: System side length
    Returns:
        Sum of the spins in x which are nearest neighbors of (i, j, k)
    """

    L_m = L - 1 # save for cheap reuse
    result =  x[(i+1) if i < L_m else 0, j, k] + \
              x[(i-1) if i > 0 else L_m, j, k]
    result += x[i, (j+1) if j < L_m else 0, k] + \
              x[i, (j-1) if j > 0 else L_m, k]
    result += x[i, j, (k+1) if k < L_m else 0] + \
              x[i, j, (k-1) if k > 0 else L_m]
    return int(result)


def move(x, E, J, L, n_moves=0, E_max=0, E_d=0, E_init=0):
    """
    Args:
        x: Spin configuration
        E: Current energy of system x
        J: Coupling constant
        L: System side length
        n_moves: Number of moves == flip attempts
        E_max: Energy limit
        E_d: Demon energy storage level
        E_init: If given and non-zero: Target E_ens to reach during init

    Returns:
        x, E, demon energy distribution E_d_distr and calculated inverse 
            temperature beta after n_moves Monte Carlo steps
    """
    two_J = 2 * J # for cheaper reuse
    
    if E_init != 0: # init part: bring the energy to starting level
        print('Starting from energy: {}, raising to: {}'.format(E, E_init))
        while E < E_init:
            i, j, k = np.random.randint(L, size=3)
            x_old = x[i, j, k]
            nn = nn_sum(x, i, j, k, L)
            delta_E = two_J * x_old * nn
            if delta_E > 0:
                x[i, j, k] *= -1
                E += delta_E
        return x, E
    
    else: # this is the main part: flip attempts using demon
        ijk = np.random.randint(L, size=(n_moves, 3))
        E_d_distr = np.zeros(n_moves)
        for n in range(n_moves):
            i, j, k = ijk[n]
            x_old = x[i, j, k]
            nn = nn_sum(x, i, j, k, L)
            delta_E = two_J * x_old * nn
            if E_max >= E_d - delta_E >= 0:
                x[i, j, k] *= -1
                E += delta_E
                E_d -= delta_E
            E_d_distr[n] = E_d
        beta = 0.25 * np.log(1.0 + 4.0 * J / np.mean(E_d_distr))
    return x, E, E_d_distr, beta


def plot(E_d_distr, L, E_ens):
    """
    Args:
        E_d_distr: Array of E_d values observed over Monte Carlo steps
        L: System side length (for plot title only)
        E_ens: Target ensemble energy (for plot title only)
    Returns:
        Inverse temperature beta from slope of linear fit to plot
    """
    binwidth = 4 # energy step size we get in 3d Ising from Creutz algorithm
    plt.figure()
    n, bin_edges, _ = plt.hist(E_d_distr, bins=range(int(min(E_d_distr)), int(max(E_d_distr))+binwidth, binwidth),
     density=True,
     histtype='step',
     label='$P(E_d)$')
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    p = np.polyfit(bin_centres, np.log(n), 1)
    plt.semilogy(bin_centres, np.exp(bin_centres * p[0] + p[1]), '--', label='linear fit: slope = {:.3f}'.format(p[0]))
    plt.yscale('log')
    plt.xlabel('$E_d$')
    plt.ylabel('$P(E_d)$')
    plt.title('Distribution of demon energy $E_d$ for $L=${}, $E=${}, {} samples'.format(L, int(E_ens), len(E_d_distr)))
    plt.legend()
    plt.savefig('E_d_L{}_E{}.pdf'.format(L, np.abs(int(E_ens))))
    plt.close('all')
    return -p[0]

def simulate(n_moves, E_ens, L):
    """
    Args:
        n_moves: Number of MC steps to perform
        E_ens: Target energy for NVE-ensemble
        L: System side length
    Returns:
        Inverse temperatures beta (calculated and from E_d distribution plot)
    """
    LLL = L**3  # L^3 for cheaper reuse
    J = 1.0  # coupling constant
    E = -6 * J * LLL / 2  # every lattice site contributes an energy -3J
    E_max = np.abs(0.05 * E)  # heuristic energy limit for Creutz algorithm
    print(E_max)
    E_d = 0 # demon starting energy


    if (n_moves < 10 * LLL): # number of MC moves for high quality data
        print('Consider 10 * L^3 MC steps or more for high quality (better than 5%) results')
    print('Running simulation: L = {}, E_ens = {}, {} MC steps per sample'.format(
           L, E_ens, n_moves))

    x = np.ones((L, L, L)) # initial configuration: all spins aligned up
    print('Energy after init: {}'.format(E))

    # reach target energy E_ens
    x, E = move(x, E, J, L, E_init=E_ens)
    print('Energy now : {}, releasing demon...'.format(E))

    # run MC steps 
    x, E, E_d_distr, beta_calc = move(x, E, J, L, n_moves, E_max, E_d)

    # plot E_d distribution and get -slope == beta
    beta_plot = plot(E_d_distr, L, E_ens)
    return [beta_calc, beta_plot]


def main():
    if (len(sys.argv) < 4):
        print('Usage: python3 ./creutz3d.py n_moves L avg_E1 [avg_E2 avg_E3 ...]')
        print('Example: python3 ./creutz3d.py 1000000 10 -2.9 -2.5 -2 -1.5 -1 -0.5')
        exit()

    n_moves = int(sys.argv[1])
    L = int(sys.argv[2])
    E_ens_range = [float(E_avg) * L * L * L for E_avg in sys.argv[3:]]  # system sizes to simulate

    results = [simulate(n_moves, E_ens, L) for E_ens in E_ens_range]

    for i in range(len(E_ens_range)):
        print(80*'=')
        print('System of size {} (L = {}) at energy E = {}'.format(L*L*L, L, E_ens_range[i]))
        print('Calculated inverse temperature from <E_d> = {:.3f}'.format(results[i][0]))
        print('Plotted inverse temperature from log(P(E_d)) = {:.3f}'.format(results[i][1]))

    T_calc = [1 / betas[0] for betas in results]
    T_plot = [1 / betas[1] for betas in results]
    plt.figure()
    plt.plot(T_calc, np.array(E_ens_range) / (L*L*L), marker='o', label='$T$ calculated')
    plt.plot(T_plot, np.array(E_ens_range) / (L*L*L), marker='o', label='$T$ plotted')
    plt.legend()
    plt.xlabel('Temperature $T$')
    plt.ylabel('Energy $E$')
    plt.title('$T$ vs $E$ for L={}, system size={}, {} samples'.format(L, L*L*L, n_moves))
    plt.savefig('T_vs_E_L{}.pdf'.format(L))

if __name__ == '__main__':
    main()
