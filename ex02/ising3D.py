import numpy as np
import matplotlib.pyplot as plt


def nn_sum(x, i, j, k):
    """
    Args:
        x: Spin configuration
        i, j, k: Indices describing the position of one spin

    Returns:
        Sum of the spins in x which are nearest neighbors of (i, j, k)
    """
    result = x[(i+1)%L, j, k] + x[(i-1)%L, j, k]
    result += x[i, (j+1)%L, k] + x[i, (j-1)%L, k]
    result += x[i, j, (k+1)%L] + x[i, j, (k-1)%L]
    return int(result)


def move(x, M, E):
    """
    Args:
        x: Spin configuration
        M: Magnetization of x
        E: Energy of x

    Returns:
        Updated x, M and E after one Monte Carlo move
    """
    # pick one spin at random
    i = int(L*np.random.rand())
    j = int(L*np.random.rand())
    k = int(L*np.random.rand())
    x_old = x[i, j, k]

    # flip the spin according to the Metropolis algorithm
    nn = nn_sum(x, i, j, k)
    if x_old == 1:
        R = table_spin_up[int((nn+6)/2)] # Metropolis acceptance probability
    else:
        R = table_spin_down[int((nn+6)/2)]
    eta = np.random.rand()
    if R > eta:
        x[i, j, k] *= -1
        M -= 2*x_old
        E += 2*J*x_old*nn

    return x, M, E


L = 10 # size of the system
J = 1 # coupling constant
beta = np.linspace(0, 0.6, 10) # inverse temperatures
Nthermalization = 100000 # number of thermalization steps
Nsample = 100 # number of samples (= size of the Markov chain)
Nsubsweep = 1000 # number of subsweeps (to generate better samples)

M_arr = np.zeros_like(beta) # average magnetizations
E_arr = np.zeros_like(beta) # average energies
M_err = np.zeros_like(beta) # standard deviations of the magnetizations
E_err = np.zeros_like(beta) # standard deviations of the energies
chi_arr = np.zeros_like(beta) # magnetic susceptibilities
cv_arr = np.zeros_like(beta) # heat capacities

# calculate the relevant physical quantities for different temperatures
for t in range(beta.size):
    print('Running for inverse temperature =', beta[t])

    # Probability look-up tables
    table_spin_up = np.exp(-2*J*beta[t]*np.array([-6, -4, -2, 0, 2, 4, 6]))
    table_spin_down = np.exp(+2*J*beta[t]*np.array([-6, -4, -2, 0, 2, 4, 6]))

    # random initial configuration
    x = np.ones((L, L, L))
    M = L**3
    E = -3*J*L**3 # every lattice site contributes an energy -3J
    for i in range(L):
        for j in range(L):
            for k in range(L):
                if np.random.rand() < 0.5:
                    x[i, j, k] = -1
                    M -= 2
                    E += 2*J*nn_sum(x, i, j, k)

    print('Thermalizing ...')

    # thermalization
    for nt in range(Nthermalization):
        x, M, E = move(x, M, E)

    print('Computing M and E ...')

    # computation of M and E
    M_data = np.zeros(Nsample)
    E_data = np.zeros(Nsample)

    M_data[0] = np.abs(M)/L**3
    E_data[0] = E/L**3

    for n in range(1, Nsample):
        for N in range(Nsubsweep):
            x, M, E = move(x, M, E)
        M_data[n] = np.abs(M)/L**3
        E_data[n] = E/L**3

    M_arr[t] = np.mean(M_data) # average magnetization
    E_arr[t] = np.mean(E_data) # average energy
    M_err[t] = np.std(M_data) # standard deviation of the magnetization
    E_err[t] = np.std(E_data) # standard deviation of the energy
    chi_arr[t] = beta[t]*M_err[t]**2 # magnetic susceptibility
    cv_arr[t] = beta[t]**2*E_err[t]**2 # heat capacity

# plot magnetization
plt.figure()
plt.errorbar(beta, M_arr, yerr=M_err)
plt.xlabel('Inverse temperature')
plt.ylabel('Magnetization')
plt.savefig('M.pdf')

# plot magnetic susceptibility
plt.figure()
plt.plot(beta, chi_arr)
plt.xlabel('Inverse temperature')
plt.ylabel('Susceptibility')
plt.savefig('chi.pdf')

# plot energy
plt.figure()
plt.errorbar(beta, E_arr, yerr=E_err)
plt.xlabel('Inverse temperature')
plt.ylabel('Energy')
plt.savefig('E.pdf')

# plot heat capacity
plt.figure()
plt.plot(beta, cv_arr)
plt.xlabel('Inverse temperature')
plt.ylabel('Heat capacity')
plt.savefig('cv.pdf')
