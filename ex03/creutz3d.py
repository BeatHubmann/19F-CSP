import numpy as np
import matplotlib.pyplot as plt
import sys
import timeit

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
        E_max: Energy limit for system x
        E_d: Demon energy storage level
        E_init: If given and non-zero: Target E_ens to reach during init

    Returns:
        Updated x, M and E after one Monte Carlo move
    """
    two_J = 2 * J # for cheaper reuse
    
    if E_init != 0: # we have to bring the energy to the starting level
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
    
    else: # flip attempts using demon
        E_d_sum = 0
        ijk = np.random.randint(L, size=(n_moves, 3))
        for n in range(n_moves):
            i, j, k = ijk[n]
            x_old = x[i, j, k]
            nn = nn_sum(x, i, j, k, L)
            delta_E = two_J * x_old * nn
            if E_max >= E_d - delta_E >= 0:
                x[i, j, k] *= -1
                E += delta_E
                E_d -= delta_E
            E_d_sum += E_d
        beta = inv_temp(L, J, n_moves, E_d_sum)
    return x, E, beta


def plot(L, beta_range, E_arr, E_err, M_arr, M_err, chi_arr, cv_arr):
    # plot magnetization
    plt.figure()
    plt.errorbar(beta_range, M_arr, yerr=M_err)
    plt.xlabel('Inverse temperature')
    plt.ylabel('Magnetization')
    plt.savefig('M_L' + str(L) + '.pdf')

    # plot magnetic susceptibility
    plt.figure()
    plt.plot(beta_range, chi_arr)
    plt.xlabel('Inverse temperature')
    plt.ylabel('Susceptibility')
    plt.savefig('chi_L' + str(L) + '.pdf')

    # plot energy
    plt.figure()
    plt.errorbar(beta_range, E_arr, yerr=E_err)
    plt.xlabel('Inverse temperature')
    plt.ylabel('Energy')
    plt.savefig('E_L' + str(L) + '.pdf')

    # plot heat capacity
    plt.figure()
    plt.plot(beta_range, cv_arr)
    plt.xlabel('Inverse temperature')
    plt.ylabel('Heat capacity')
    plt.savefig('cv_L' + str(L) + '.pdf')

    plt.close('all')

def inv_temp(L, J, n_moves, E_d_sum):
    E_d_avg = E_d_sum / n_moves / L**3
    return 4.0 / np.log(1.0 + 4.0 * J / E_d_avg)

def simulate(Nsample, E_ens, L):
    """
    Args:
        Nsample: Number of samples to collect
        E_ens: Target energy for NVE-ensemble
        L: System side length
    Returns:
        Average temperature T of system
    """
    LLL = L**3  # L^3 for cheaper reuse
    LLL_inv = 1 / LLL  # L^-3 for cheaper reuse
    J = 1.0  # coupling constant
    E_max = 50 # energy limit for Creutz algorithm
    E_d = 25 # demon starting energy

    n_moves = 3 * LLL # number of MC moves per sample

    print('Running simulation: L = {}, E_ens = {}, taking {} samples, {} MC steps per sample'.format(
           L, E_ens, Nsample, n_moves))

    x = np.ones((L, L, L)) # initial configuration: all spins aligned up
    E = -6 * J * LLL / 2  # every lattice site contributes an energy -3J

    # calculate the relevant physical quantities for different temperatures
    print('Energy after init: {}'.format(E))


    # reach target energy E_ens
    x, E = move(x, E, J, L, E_init=E_ens)
    print('Energy now : {}, releasing demon...'.format(E))

    # tracking E
    E_range = []
    beta_range = []
    # take samples
    for _ in range(Nsample):
        x, E, beta = move(x, E, J, L, n_moves, E_max, E_d)
        E_range.append(E)
        beta_range.append(beta)

    E_avg = np.mean(E_range)
    E_err = np.std(E_range)
    beta_avg = np.mean(beta_range)
    beta_err = np.std(beta_range)

    return E_avg, E_err, beta_avg, beta_err


def main():
    if (len(sys.argv) < 4):
        print('Usage: ./python3 creutz3d.py Nsamples E_ens L1 [L2 L3 ...]')
        exit()

    Nsample = int(sys.argv[1])
    E_ens = int(sys.argv[2])
    L_range = [int(L) for L in sys.argv[3:]]  # system sizes to simulate

    start = timeit.default_timer()
    results = [simulate(Nsample, E_ens, L) for L in L_range]
    end = timeit.default_timer()
    print(results)
    print('Simulation time: {:.2f} seconds'.format(end - start))

    # poly = np.poly1d(np.polyfit(np.log10(L_range), np.log10(results), 1))

    # plt.figure()
    # plt.plot(np.log10(L_range), np.log10(results), label='simulated data')
    # plt.plot(np.linspace(0.7, 1.4, 20), poly(np.linspace(0.7, 1.4, 20)),
    #          label='linear fit: slope = {:.2f}'.format(poly[1]))
    # plt.xlabel('$log_{10}(L)$')
    # plt.ylabel('$log_{10}(\chi_{max})$')
    # plt.legend()
    # plt.savefig('log_chi_max.pdf')


if __name__ == '__main__':
    main()
