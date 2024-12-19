#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <array>
#include <fstream>
#include <chrono>
#include <algorithm>

// parameters
const double G = 6.67e-11;
const int N_BODIES = 1000;
const int N_DIM = 3;
// e.g. if N_BODIES > 896 and N_DIM > 9, the gpu breaks becasue of register preassure, other way around is fine
const double DELTA_T = 1.0;
const int N_SIMULATIONS = 100;
 //  large numbers of bodies, dimensions and simulation steps introduce accumulated miscalculations of small numbers
const double LOWER_M = 1e-6;
const double HIGHER_M = 1e6;
const double LOWER_P = -1e-1;
const double HIGHER_P = 1e-1;
const double LOWER_V = -1e-4;
const double HIGHER_V = 1e-4;

const int MAX_BLOCK_SIZE = 1024; // limit for threads in CUDA

// structures
using Vector = std::array<double, N_DIM>;
using Positions = std::array<Vector, N_BODIES>;
using Velocities = std::array<Vector, N_BODIES>;
using Forces = std::array<Vector, N_BODIES>;
using Masses = std::array<double, N_BODIES>;
using Accelerations = std::array<Vector, N_BODIES>;

double generateRandom(double lower, double upper) {
    return lower + static_cast<double>(std::rand()) / RAND_MAX * (upper - lower);
}

double generateLogRandom(double lower, double upper) {
    return std::pow(10, std::log10(lower) + static_cast<double>(std::rand()) / RAND_MAX * (std::log10(upper) - std::log10(lower)));
}

void initializeMasses(Masses& masses, double LOWER_M, double HIGHER_M) {
    for (double& mass : masses) {
        mass = generateLogRandom(LOWER_M, HIGHER_M);
    }
}

void initializeVectors(Positions& vectors, double lower, double upper) {
    for (auto& vector : vectors) {
        for (double& component : vector) {
            component = generateRandom(lower, upper);
        }
    }
}

void computeForces(const Positions& positions, const Masses& masses, Forces& forces) {
    for (int i = 0; i < N_BODIES; ++i) {
        Vector sum = {};
        for (int j = 0; j < N_BODIES; ++j) {
            if (i == j) continue;

            double distance_squared = 0.0;
            Vector displacement = {};
            for (int k = 0; k < N_DIM; ++k) {
                displacement[k] = positions[j][k] - positions[i][k];
                distance_squared += displacement[k] * displacement[k];
            }

            double distance = std::sqrt(distance_squared);
            double factor = G * masses[i] * masses[j] / (distance_squared * distance);

            for (int k = 0; k < N_DIM; ++k) {
                sum[k] += factor * displacement[k];
            }
        }
        forces[i] = sum;
    }
}

__global__ void computeForcesGpu(double* positions, double* masses, double* forces) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N_BODIES)
        return;
    
    double sum[N_DIM] = {};
    for (int j = 0; j < N_BODIES; ++j) {
        if (idx == j) continue;

        double distance_squared = 0.0;
        double displacement[N_DIM] = {};
        for (int k = 0; k < N_DIM; ++k) {
            displacement[k] = positions[j * N_DIM + k] - positions[idx * N_DIM + k];
            distance_squared += displacement[k] * displacement[k];
        }

        double distance = std::sqrt(distance_squared);
        double factor = G * masses[idx] * masses[j] / (distance_squared * distance);

        for (int k = 0; k < N_DIM; ++k) {
            sum[k] += factor * displacement[k];
        }
    }
    for (int k = 0; k < N_DIM; ++k) {
        forces[idx * N_DIM + k] = sum[k];
    }
}

void updateAccelerations(const Forces& forces, const Masses& masses, Positions& accelerations) {
    for (int i = 0; i < N_BODIES; ++i) {
        for (int k = 0; k < N_DIM; ++k) {
            accelerations[i][k] = forces[i][k] / masses[i];
        }
    }
}

__global__ void updateAccelerationsGpu(double* forces, double* masses, double* accelerations) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N_BODIES)
        return;

    for (int k = 0; k < N_DIM; ++k) {
        accelerations[idx * N_DIM + k] = forces[idx * N_DIM + k] / masses[idx];
    }
}

void updateVelocities(Velocities& velocities, const Positions& accelerations, double DELTA_T) {
    for (int i = 0; i < N_BODIES; ++i) {
        for (int k = 0; k < N_DIM; ++k) {
            velocities[i][k] += accelerations[i][k] * DELTA_T;
        }
    }
}

__global__ void updateVelocitiesGpu(double* velocities, double* accelerations, double DELTA_T) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N_BODIES)
        return;
    
    for (int k = 0; k < N_DIM; ++k) {
        velocities[idx * N_DIM + k] += accelerations[idx * N_DIM + k] * DELTA_T;
    }
}

void updatePositions(Positions& positions, const Velocities& velocities, double DELTA_T) {
    for (int i = 0; i < N_BODIES; ++i) {
        for (int k = 0; k < N_DIM; ++k) {
            positions[i][k] += velocities[i][k] * DELTA_T;
        }
    }
}

__global__ void updatePositionsGpu(double* positions, double* velocities, double DELTA_T) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N_BODIES)
        return;
    
    for (int k = 0; k < N_DIM; ++k) {
        positions[idx * N_DIM + k] += velocities[idx * N_DIM + k] * DELTA_T;
    }
}

void printBodies(const Masses& masses, const Positions& positions, const Velocities& velocities) {
    for (int i = 0; i < N_BODIES; ++i) {
        std::cout << "Body " << i << ":\n";
        std::cout << "  Mass: " << masses[i] << "\n";
        std::cout << "  Position: [ ";
        for (const double& pos : positions[i]) {
            std::cout << pos << ' ';
        }
        std::cout << "]\n";
        std::cout << "  Velocity: [ ";
        for (const double& vel : velocities[i]) {
            std::cout << vel << ' ';
        }
        std::cout << "]\n";
    }
}

void savePositions(std::string& output_str, const Positions& positions, double time) {
    for (int i = 0; i < N_BODIES; ++i) {
        output_str += std::to_string(time) + " " + std::to_string(i) + " ";
        for (const double& pos : positions[i]) {
            output_str += std::to_string(pos) + " ";
        }
        output_str += "\n";
    }
}

void runSimulationCpu(Masses masses, Positions& positions, Velocities velocities) {
    Accelerations accelerations = {};
    Forces forces = {};

    std::ofstream positions_file("positions.txt");
    std::string output_str;

    double absolute_t = 0.0;
    savePositions(output_str, positions, absolute_t);
    //printBodies(masses, positions, velocities);
    
    for (int step = 0; step < N_SIMULATIONS; ++step) {
        absolute_t += DELTA_T;

        computeForces(positions, masses, forces);
        updateAccelerations(forces, masses, accelerations);
        updateVelocities(velocities, accelerations, DELTA_T);
        updatePositions(positions, velocities, DELTA_T);
        savePositions(output_str, positions, absolute_t);
    }
    
    positions_file << output_str;
    positions_file.close();
}

void runSimulationGpu(Masses masses, Positions& positions, Velocities velocities) {
    double* masses_d;
    double* positions_d;
    double* velocities_d;
    double* accelerations_d;
    double* forces_d;

    cudaMalloc( (void**)&masses_d, N_BODIES * sizeof(double));
    cudaMalloc( (void**)&positions_d, N_BODIES * N_DIM * sizeof(double));
    cudaMalloc( (void**)&velocities_d, N_BODIES * N_DIM * sizeof(double));
    cudaMalloc( (void**)&accelerations_d, N_BODIES * N_DIM * sizeof(double));
    cudaMalloc( (void**)&forces_d, N_BODIES * N_DIM * sizeof(double));

    cudaMemcpy( masses_d, masses.data(), N_BODIES * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy( positions_d, positions.data(), N_BODIES * N_DIM * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy( velocities_d, velocities.data(), N_BODIES * N_DIM * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = (N_BODIES <= MAX_BLOCK_SIZE) ? N_BODIES : MAX_BLOCK_SIZE;

    dim3 dimBlock(blockSize);
	dim3 dimGrid((N_BODIES + blockSize - 1) / blockSize);

    double absolute_t = 0.0;
    
    for (int step = 0; step < N_SIMULATIONS; ++step) {
        absolute_t += DELTA_T;

        computeForcesGpu<<<dimGrid, dimBlock>>>(positions_d, masses_d, forces_d);
        cudaDeviceSynchronize();
        updateAccelerationsGpu<<<dimGrid, dimBlock>>>(forces_d, masses_d, accelerations_d);
        cudaDeviceSynchronize();
        updateVelocitiesGpu<<<dimGrid, dimBlock>>>(velocities_d, accelerations_d, DELTA_T);
        cudaDeviceSynchronize();
        updatePositionsGpu<<<dimGrid, dimBlock>>>(positions_d, velocities_d, DELTA_T);
    }
    
    cudaMemcpy( positions.data(), positions_d, N_BODIES * N_DIM * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(masses_d);
    cudaFree(positions_d);
    cudaFree(velocities_d);
    cudaFree(accelerations_d);
    cudaFree(forces_d);
}

void checkEqual(const auto& first, const auto& second, const std::string& name) {
    bool allEqual = true;

    for (size_t i = 0; i < first.size(); ++i) {
        for (size_t j = 0; j < first[i].size(); ++j) {
            if (std::fabs(first[i][j] - second[i][j]) > 1e-3) {
                allEqual = false;
                std::cout << "Difference at index [" << i << "][" << j << "]: "
                          << "first = " << first[i][j]
                          << ", second = " << second[i][j]
                          << " , and the diff is: " << std::fabs(first[i][j] - second[i][j]) << std::endl;
                break;
            }
        }
    }
    if (allEqual) {
        std::cout << "\nThe " << name << " are the same.";
    } else {
        std::cout << "\n\n!!!!! The " << name << " are NOT the same !!!!!\n\n";
    }
}

int main() {
    std::srand(static_cast<unsigned>(std::time(0)));

    // structures
    Masses masses;
    Positions positions;
    Velocities velocities;

    // initialization
    initializeMasses(masses, LOWER_M, HIGHER_M);
    initializeVectors(positions, LOWER_P, HIGHER_P);
    initializeVectors(velocities, LOWER_V, HIGHER_V);

    // cpu simulation run

    Positions positions_cpu = positions;

    auto start_cpu = std::chrono::high_resolution_clock::now();

    runSimulationCpu(masses, positions_cpu, velocities);

    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);

    // gpu simulation run

    Positions positions_gpu = positions;

    auto start_gpu = std::chrono::high_resolution_clock::now();

    runSimulationGpu(masses, positions_gpu, velocities);
    
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto duration_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu);

    std::cout<<std::endl<<std::endl;

    checkEqual(positions_cpu, positions_gpu, "final positions");

    std::cout<<std::endl<<std::endl;

    std::cout << "CPU computation took " << duration_cpu.count() << " milliseconds." << std::endl;
    std::cout << "GPU computation took " << duration_gpu.count() << " milliseconds." << std::endl;

    std::cout<<std::endl<<std::endl;

    return 0;
}
