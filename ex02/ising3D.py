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

    Returns:
        Sum of the spins in x which are nearest neighbors of (i, j, k)
    """

    L_m = L - 1
    result = x[(i+1) if i < L_m else 0, j, k] + \
        x[(i-1) if i > 0 else L_m, j, k]
    result += x[i, (j+1) if j < L_m else 0, k] + \
        x[i, (j-1) if j > 0 else L_m, k]
    result += x[i, j, (k+1) if k < L_m else 0] + \
        x[i, j, (k-1) if k > 0 else L_m]
    return int(result)


def move(x, E, J, L, M, table_spin_down_up, n_moves):
    """
    Args:
        x: Spin configuration
        M: Magnetization of x
        E: Energy of x

    Returns:
        Updated x, M and E after one Monte Carlo move
    """

    ijk = np.random.randint(L, size=(n_moves, 3))
    eta_accept = np.random.rand(n_moves)
    for n in range(n_moves):
        i, j, k = ijk[n]
        x_old = x[i, j, k]
        nn = nn_sum(x, i, j, k, L)
        R = table_spin_down_up[int(0.5 * (x_old + 1))][int(0.5 * (nn + 6))]
        if R > eta_accept[n]:
            x[i, j, k] *= -1
            M -= 2*x_old
            E += 2*J*x_old*nn
    return x, M, E


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


def simulate(Nsample, L):
    LLL = L**3  # L^3 for cheaper reuse
    LLL_inv = 1 / LLL  # L^-3 for cheaper reuse
    J = 1.0  # coupling constant
    T_c_inv = 1 / 4.51
    # beta_range = np.linspace(0.0, 0.6, 10)  # inverse temperatures
    beta_range = np.hstack((np.linspace(T_c_inv - 0.1, T_c_inv, 2, endpoint=False), np.linspace(T_c_inv, T_c_inv + 0.1, 3))) # inverse temperatures
    # beta_range = [T_c_inv]
    Nthermalization = 100 * LLL  # number of thermalization steps
    # number of subsweeps (to generate better samples: decorrelation)
    Nsubsweep = 3 * LLL
    M_arr = []  # average magnetizations
    E_arr = []  # average energies
    M_err = []  # standard deviations of the magnetizations
    E_err = []  # standard deviations of the energies
    chi_arr = []  # magnetic susceptibilities
    cv_arr = []  # heat capacities

    print('Running simulation: L= {}, taking {} samples, {} subsweeps == {} sweeps for thermalization, {} subsweeps for sample decorrelation'.format(
           L, Nsample, Nthermalization, int(Nthermalization*LLL_inv), Nsubsweep))

    # random initial configuration
    x = np.ones((L, L, L))
    M = LLL
    E = -6 * J * LLL / 2  # every lattice site contributes an energy -3J
    for i in range(L):
        for j in range(L):
            for k in range(L):
                if np.random.rand() > 0.5:
                    x[i, j, k] = -1
                    M -= 2
                    E += 2 * J * nn_sum(x, i, j, k, L)

    # calculate the relevant physical quantities for different temperatures
    for beta in beta_range:
        print('Running for inverse temperature =', beta)

        # Probability look-up tables
        table_spin_up = np.exp(-2 * J * beta *
                               np.array([-6, -4, -2, 0, 2, 4, 6]))
        table_spin_down = np.exp(+2 * J * beta *
                                 np.array([-6, -4, -2, 0, 2, 4, 6]))
        table_spin_down_up = np.vstack((table_spin_down, table_spin_up))

        print('Thermalizing ...')

        # thermalization
        x, M, E = move(x, E, J, L, M, table_spin_down_up, Nthermalization)

        print('Computing M and E ...')

        # computation of M and E
        M_data = [np.abs(M) * LLL_inv]
        E_data = [E * LLL_inv]

        for _ in range(Nsample):
            x, M, E = move(x, E, J, L, M, table_spin_down_up, Nsubsweep)
            M_data.append(np.abs(M) * LLL_inv)
            E_data.append(E * LLL_inv)

        M_arr.append(np.mean(M_data))  # average magnetization
        E_arr.append(np.mean(E_data))  # average energy
        M_err.append(np.std(M_data))  # standard deviation of the magnetization
        E_err.append(np.std(E_data))  # standard deviation of the energy
        chi_arr.append(beta * M_err[-1]**2)  # magnetic susceptibility
        cv_arr.append(beta**2 * E_err[-1]**2)  # heat capacity

    chi_max = np.max(chi_arr)
    plot(L, beta_range, E_arr, E_err, M_arr, M_err, chi_arr, cv_arr)

    return chi_max


def main():
    if (len(sys.argv) < 3):
        print('Usage: ./python3 ising3d.py Nsamples L1 [L2 L3 ...]')
        exit()

    # global Nsample, J, beta, M_arr, M_err, E_arr, E_err, cv_arr, chi_arr
    # number of samples (= size of the Markov chain)
    Nsample = int(sys.argv[1])
    L_range = [int(L) for L in sys.argv[2:]]  # system sizes to simulate

    start = timeit.default_timer()
    results = [simulate(Nsample, L) for L in L_range]
    end = timeit.default_timer()
    print(results)
    print('Simulation time: {:.2f} seconds'.format(end - start))

    poly = np.poly1d(np.polyfit(np.log10(L_range), np.log10(results), 1))

    plt.figure()
    plt.plot(np.log10(L_range), np.log10(results), label='simulated data')
    plt.plot(np.linspace(0.7, 1.4, 20), poly(np.linspace(0.7, 1.4, 20)),
             label='linear fit: slope = {:.2f}'.format(poly[1]))
    plt.xlabel('$log_{10}(L)$')
    plt.ylabel('$log_{10}(\chi_{max})$')
    plt.legend()
    plt.savefig('log_chi_max.pdf')


if __name__ == '__main__':
    main()
