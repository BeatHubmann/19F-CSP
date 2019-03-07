#include <iostream>
#include <random>
#include <cmath>

#include <mpi.h>



const int h(const std::vector<int> grid, const int i) // calculate h(i) w/ periodic b.c.
{
    const int row{i / L};
    const int col{i % L};
    const int west{col - 1 < 0 ? i + L-1  : i - 1};
    const int east{col + 1 < L ? i + 1    : i - (L-1)};
    const int north{row - 1 < 0 ? i + (L-1) * L : i - L};
    const int south{row + 1 < L ? i + L         : i - (L-1) * L};
    const int up{};
    const int down{};
    return grid[west] + 
           grid[east] +
           grid[north] +
           grid[south] +
           grid[up] +
           grid[down];
}

void FillLookup(double exp_dE[], const double T) // precalculate values for MC step
{
    for (auto s_i= -1; s_i < 2; s_i += 2) // s_i = -1, +1
        for (auto h_i= -4; h_i < 5; h_i += 2) // h_i = -4, -2, 0, +2, +4
            exp_dE[(s_i + 1) / 2 * 5 + (h_i + 4) / 2]= std::exp(-2 * (double)s_i * (double)h_i * J / T); 
}

double GetExpDeltaE(const int s_i, const int h_i, const double exp_dE[]) // lookup precalculated values in MC step
{
    return exp_dE[(s_i + 1) / 2 * 5 + (h_i + 4) / 2];
}


int size, rank;
MPI_Comm_size