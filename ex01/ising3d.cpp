#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include <tuple>
#include <algorithm>

static int L;                  // lattice side length
static int thermalization;     // steps to reach thermal equilibrium
static int decorrelation;      // steps btw measurements to avoid correlation
static const double J{1.0};    // exchange coupling constant
static const double T_c{4.51}; // critical temp for 3D Ising model

const int h(const std::vector<int> grid, const int i) // calculate h(i) w/ periodic b.c.
{
    const int LL{L * L};     // used several times below
    const int lvl{i / (LL)}; // level in 3d cube
    const int ii{i % (LL)};  // local running index on level
    const int row{ii / L};   // row on level
    const int col{ii % L};   // col on level
    // below: periodic boundary conditions
    const int west{col - 1 < 0 ? i + L - 1 : i - 1};
    const int east{col + 1 < L ? i + 1 : i - (L - 1)};
    const int south{row - 1 < 0 ? i + (L - 1) * L : i - L};
    const int north{row + 1 < L ? i + L : i - (L - 1) * L};
    const int up{lvl + 1 < L ? i + (LL) : i - (L - 1) * LL};
    const int down{lvl - 1 < 0 ? i + (L - 1) * LL : i - LL};
    // below: return sum of nn neighbours of index i
    return grid[west] +
           grid[east] +
           grid[north] +
           grid[south] +
           grid[up] +
           grid[down];
}

void FillLookup(double exp_dE[], const double T) // precalculate values for MC step
{
    for (auto s_i = -1; s_i < 2; s_i += 2)     // s_i = -1, +1 : allowed spin values
        for (auto h_i = -6; h_i < 7; h_i += 2) // h_i = -6, -4, -2, 0, +2, +4, +6 : all possible h_i
            exp_dE[(s_i + 1) / 2 * 7 + (h_i + 6) / 2] = std::exp(-2 * (double)s_i * (double)h_i * J / T);
}

double GetExpDeltaE(const int s_i, const int h_i, const double exp_dE[]) // lookup precalculated values in MC step
{
    return exp_dE[(s_i + 1) / 2 * 7 + (h_i + 6) / 2]; // this has some index arithmetic for simple array
}

// core Monte Carlo function:
const std::tuple<double, double, std::vector<double>, double, double, std::vector<double>, double, double>
RunExperiment(std::vector<int> &grid, double E, double M, const double T, const int measurements, const int seed,
              const bool cold_start = false)
{
    const double LLL{(double)(L * L * L)}; // used multiple times for site averages
    const double beta{1/T};
    if (cold_start)
    {
        grid.clear(); // reset grid to cold ground state
        for (auto i = 0; i < L * L * L; i++)
            grid.push_back(1);
        M = LLL;              // reset M
        E = -J * 6 * LLL / 2; // reset E
    }

    double exp_dE[14];     // fixed size: s = -1, +1; h = -6, -4, -2, 0, 2, 4, 6 : 2*7=14 entries total
    FillLookup(exp_dE, T); // precalculate values for MC step

    std::mt19937 rd(seed);                                       // random engine
    std::uniform_int_distribution<int> choice(0, L * L * L - 1); // for choosing site
    std::uniform_real_distribution<double> accept(0.0, 1.0);     // for Metropolis MC

    std::vector<double> magnet_steps{M}; // for analysis/plotting
    std::vector<double> energy_steps{E};

    double sum_M{0}; // for equilibrium averages
    double sum_E{0};
    double sum_M2{0};
    double sum_E2{0};

    for (auto s = 0; s < thermalization + measurements * decorrelation; ++s) // perform Metropolis MC steps: 1x steps for equilibrium, 1x steps for measurement
    {
        const int i{choice(rd)};                          // choose site i
        const int h_i{h(grid, i)};                        // get h_i
        const double delta_energy{2 * grid[i] * h_i * J}; // calculate dE
        if (delta_energy < 0)                             // accept
        {
            grid[i] *= -1;     // flip site i
            M += 2 * grid[i];  // update M after flip
            E += delta_energy; // update E after flip
        }
        else // Monte Carlo step
        {
            if (accept(rd) < GetExpDeltaE(grid[i], h_i, exp_dE)) // lucky accept
            {
                grid[i] *= -1; // as accept above...
                M += 2 * grid[i];
                E += delta_energy;
            }
        }

        if (s >= thermalization && s % decorrelation == 0) // start recording measurements after having obtained equilibrium; only every 10th to avoid correlation
        {
            sum_M += M; // update w.r.t. equilibrium average
            sum_E += E;
            sum_M2 += M * M;
            sum_E2 += E * E;
            magnet_steps.push_back(M / LLL);     // step's avg site magnetization
            energy_steps.push_back(E / LLL / 2); // steps's avg site energy
        }
    }

    double E_avg{sum_E / (double)measurements / LLL};
    double E2_avg{sum_E2 / (double)measurements / LLL / LLL};
    double M_avg{sum_M / (double)measurements / LLL};
    double M2_avg{sum_M2 / (double)measurements / LLL / LLL};
    double C{beta * beta * (E2_avg - E_avg * E_avg)};
    double X{beta * (M2_avg - M_avg * M_avg)};

    std::cout << "lattice side length= " << L << " , J/T= " << J / T << std::endl; // progress report
    return std::make_tuple(E, // system total energy for next run
                           E_avg,        // ensemble average E
                           energy_steps, // for plotting time dependence of E
                           M, // system total magnetization for next run
                           M_avg, // ensemble average M
                           magnet_steps, // for plotting time dependence of M
                           C, // heat capacity
                           X);  // magnetic susceptibility
}

int main(int argc, char *argv[])
{
    if (argc < 5 || argc > 5 || argv[1] == "-h") // check command line arguments and give some help
    {
        std::cerr << "Usage: " << argv[0]
                  << " steps(int): Number of measurements   seed(int): Initial RNG seed    cold_starts(0/1): Use cold starts     L(int): Lattice side length"
                  << std::endl
                  << std::endl;
        return 1;
    }
    const int measurements{atoi(argv[1])};       // max number of Monte Carlo steps
    const int seed{atoi(argv[2])};        // initial RNG seed
    const bool cold_start{atoi(argv[3])}; // 1 =: start each T with fresh all +1 spin grid - not recommended
    L= atoi(argv[4]);                     // lattice side length
    const double LLL{(double)(L * L * L)};   // used multiple times
    thermalization= LLL * 1000;           // set thermalization steps depending on L
    decorrelation= LLL;                   // set decorrelation steps depending on L

    std::vector<double> temperatures{T_c}; // generate vector of Ts to conduct experiments for; make sure T_c is included
    for (auto t = 1.0; t < 7.1; t += 0.1)
        temperatures.push_back(t);
    std::sort(temperatures.begin(), temperatures.end()); // sort in increasing order

    std::vector<int> grid(LLL, 1); // set up grid
    double E{-J * 6 * LLL / 2};    // initial energy for all sites
    double M{LLL};                 // initial magnetization for all sites

    std::vector<std::tuple<double, double, std::vector<double>, double, std::vector<double>, double, double>> results{};
    for (const auto &T : temperatures)
    {
        const auto result = RunExperiment(grid, E, M, T, measurements, seed + T, cold_start); // run experiment for T
        E = std::get<0>(result);                                                       // to feed into next T's experiment
        const auto E_avg{std::get<1>(result)};                                         // for output
        const auto E_steps{std::get<2>(result)};                                       // for output
        M = std::get<3>(result);                                                       // to feed into next T's experiment
        const auto M_avg{std::get<4>(result)};                                         // for output
        const auto M_steps{std::get<5>(result)};                                       // for output
        const auto C{std::get<6>(result)};                                             // for output
        const auto X{std::get<7>(result)};                                             // for output
        results.push_back(std::make_tuple(T,
                                          E_avg, E_steps,
                                          M_avg, M_steps,
                                          C,
                                          X)); // bundle into vector
    }

    std::cout << std::endl
              << std::endl;

    for (auto &result : results) // Print results
        std::cout << std::get<0>(result) << "\t" // T
                  << std::get<1>(result) << "\t" // <E>
                  << std::get<3>(result) << "\t" // <M>
                  << std::get<5>(result) << "\t" // C
                  << std::get<6>(result) << std::endl; // X

    std::cout << std::endl
              << std::endl;

    return 0;
}
