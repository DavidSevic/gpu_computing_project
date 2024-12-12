#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <array>
#include <fstream>
#include <chrono>
#include <algorithm>

// parameters
const double g = 6.67e-11;
const int n_bodies = 1000;
const int n_dim = 3;
// e.g. if n_bodies > 896 and n_dim > 9, the gpu breaks becasue of register preassure, other way around is fine
const double delta_t = 1.0;
const int n_simulations = 100;
 //  large numbers of bodies, dimensions and simulation steps introduce accumulated miscalculations of small numbers
const double lower_m = 1e-6;
const double higher_m = 1e6;
const double lower_p = -1e-1;
const double higher_p = 1e-1;
const double lower_v = -1e-4;
const double higher_v = 1e-4;

const int MAX_BLOCK_SIZE = 1024; // limit for threads in CUDA

// structures
using Vector = std::array<double, n_dim>;
using Positions = std::array<Vector, n_bodies>;
using Velocities = std::array<Vector, n_bodies>;
using Forces = std::array<Vector, n_bodies>;
using Masses = std::array<double, n_bodies>;
using Accelerations = std::array<Vector, n_bodies>;

// debug
Forces forcesGpu;
Forces forcesCpu;
Accelerations accelerationsGpu;
Accelerations accelerationsCpu;
Velocities velocitiesGpu;
Velocities velocitiesCpu;

double generateRandom(double lower, double upper) {
    return lower + static_cast<double>(std::rand()) / RAND_MAX * (upper - lower);
}

double generateLogRandom(double lower, double upper) {
    return std::pow(10, std::log10(lower) + static_cast<double>(std::rand()) / RAND_MAX * (std::log10(upper) - std::log10(lower)));
}

void initializeMasses(Masses& masses, double lower_m, double higher_m) {
    for (double& mass : masses) {
        mass = generateLogRandom(lower_m, higher_m);
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
    for (int i = 0; i < n_bodies; ++i) {
        Vector sum = {};
        for (int j = 0; j < n_bodies; ++j) {
            if (i == j) continue;

            double distance_squared = 0.0;
            Vector displacement = {};
            for (int k = 0; k < n_dim; ++k) {
                displacement[k] = positions[j][k] - positions[i][k];
                distance_squared += displacement[k] * displacement[k];
            }

            double distance = std::sqrt(distance_squared);
            double factor = g * masses[i] * masses[j] / (distance_squared * distance);

            for (int k = 0; k < n_dim; ++k) {
                sum[k] += factor * displacement[k];
            }
        }
        forces[i] = sum;
    }
}

__global__ void computeForcesGpu(double* positions, double* masses, double* forces) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_bodies)
        return;
    
    double sum[n_dim] = {};
    for (int j = 0; j < n_bodies; ++j) {
        if (idx == j) continue;

        double distance_squared = 0.0;
        double displacement[n_dim] = {};
        for (int k = 0; k < n_dim; ++k) {
            displacement[k] = positions[j * n_dim + k] - positions[idx * n_dim + k];
            distance_squared += displacement[k] * displacement[k];
        }

        double distance = std::sqrt(distance_squared);
        double factor = g * masses[idx] * masses[j] / (distance_squared * distance);

        for (int k = 0; k < n_dim; ++k) {
            sum[k] += factor * displacement[k];
        }
    }
    for (int k = 0; k < n_dim; ++k) {
        forces[idx * n_dim + k] = sum[k];
    }
}

void updateAccelerations(const Forces& forces, const Masses& masses, Positions& accelerations) {
    for (int i = 0; i < n_bodies; ++i) {
        for (int k = 0; k < n_dim; ++k) {
            accelerations[i][k] = forces[i][k] / masses[i];
        }
    }
}

__global__ void updateAccelerationsGpu(double* forces, double* masses, double* accelerations) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_bodies)
        return;

    for (int k = 0; k < n_dim; ++k) {
        accelerations[idx * n_dim + k] = forces[idx * n_dim + k] / masses[idx];
    }
}

void updateVelocities(Velocities& velocities, const Positions& accelerations, double delta_t) {
    for (int i = 0; i < n_bodies; ++i) {
        for (int k = 0; k < n_dim; ++k) {
            velocities[i][k] += accelerations[i][k] * delta_t;
        }
    }
}

__global__ void updateVelocitiesGpu(double* velocities, double* accelerations, double delta_t) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_bodies)
        return;
    
    for (int k = 0; k < n_dim; ++k) {
        velocities[idx * n_dim + k] += accelerations[idx * n_dim + k] * delta_t;
    }
}

void updatePositions(Positions& positions, const Velocities& velocities, double delta_t) {
    for (int i = 0; i < n_bodies; ++i) {
        for (int k = 0; k < n_dim; ++k) {
            positions[i][k] += velocities[i][k] * delta_t;
        }
    }
}

__global__ void updatePositionsGpu(double* positions, double* velocities, double delta_t) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_bodies)
        return;
    
    for (int k = 0; k < n_dim; ++k) {
        positions[idx * n_dim + k] += velocities[idx * n_dim + k] * delta_t;
    }
}

void printBodies(const Masses& masses, const Positions& positions, const Velocities& velocities) {
    for (int i = 0; i < n_bodies; ++i) {
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
    for (int i = 0; i < n_bodies; ++i) {
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
    printBodies(masses, positions, velocities);
    
    for (int step = 0; step < n_simulations; ++step) {
        absolute_t += delta_t;

        computeForces(positions, masses, forces);
        //forcesCpu = forces;
        updateAccelerations(forces, masses, accelerations);
        //accelerationsCpu = accelerations;
        updateVelocities(velocities, accelerations, delta_t);
        //velocitiesCpu = velocities;
        updatePositions(positions, velocities, delta_t);
        savePositions(output_str, positions, absolute_t);
    }

    
    std::cout << "cpu forces:" << std::endl;
    for (const auto& vec : forcesCpu)
        for (const auto& val : vec)
            std::cout << val << " ";
    
    positions_file << output_str;
    positions_file.close();
}

void runSimulationGpu(Masses masses, Positions& positions, Velocities velocities) {
    double* masses_d;
    double* positions_d;
    double* velocities_d;
    double* accelerations_d;
    double* forces_d;

    cudaMalloc( (void**)&masses_d, n_bodies * sizeof(double));
    cudaMalloc( (void**)&positions_d, n_bodies * n_dim * sizeof(double));
    cudaMalloc( (void**)&velocities_d, n_bodies * n_dim * sizeof(double));
    cudaMalloc( (void**)&accelerations_d, n_bodies * n_dim * sizeof(double));
    cudaMalloc( (void**)&forces_d, n_bodies * n_dim * sizeof(double));

    cudaMemcpy( masses_d, masses.data(), n_bodies * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy( positions_d, positions.data(), n_bodies * n_dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy( velocities_d, velocities.data(), n_bodies * n_dim * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = (n_bodies <= MAX_BLOCK_SIZE) ? n_bodies : MAX_BLOCK_SIZE;

    dim3 dimBlock(blockSize);
	dim3 dimGrid((n_bodies + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE);

    double absolute_t = 0.0;
    
    for (int step = 0; step < n_simulations; ++step) {
        absolute_t += delta_t;

        computeForcesGpu<<<dimGrid, dimBlock>>>(positions_d, masses_d, forces_d);
        cudaDeviceSynchronize();
        updateAccelerationsGpu<<<dimGrid, dimBlock>>>(forces_d, masses_d, accelerations_d);
        cudaDeviceSynchronize();
        updateVelocitiesGpu<<<dimGrid, dimBlock>>>(velocities_d, accelerations_d, delta_t);
        cudaDeviceSynchronize();
        updatePositionsGpu<<<dimGrid, dimBlock>>>(positions_d, velocities_d, delta_t);
    }
    
    cudaMemcpy( positions.data(), positions_d, n_bodies * n_dim * sizeof(double), cudaMemcpyDeviceToHost);










    //debug
    /*cudaMemcpy( forcesGpu.data(), forces_d, n_bodies * n_dim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy( accelerationsGpu.data(), accelerations_d, n_bodies * n_dim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy( velocitiesGpu.data(), velocities_d, n_bodies * n_dim * sizeof(double), cudaMemcpyDeviceToHost);
    */
    //Forces forces1;
    //cudaMemcpy( forces1.data(), forces_d, n_bodies * n_dim * sizeof(double), cudaMemcpyDeviceToHost);
    /*
    std::cout << "gpu forces:" << std::endl;
    for (const auto& vec : forcesGpu)
        for (const auto& val : vec)
            std::cout << val << " ";
    */
    //

    cudaFree(masses_d);
    cudaFree(positions_d);
    cudaFree(velocities_d);
    cudaFree(accelerations_d);
    cudaFree(forces_d);
}

// for debugging
/*void checkEqual(auto& first, auto& second, std::string name) {
    if(std::equal(first.begin(), first.end(), second.begin())) {
        std::cout<<std::endl<<std::endl<<"The " << name << " are the same."<<std::endl<<std::endl;
    } else {
        std::cout<<std::endl<<std::endl<<"!!!!! The " << name << " are NOT the same !!!!!"<<std::endl<<std::endl;
    }
}*/
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
    initializeMasses(masses, lower_m, higher_m);
    initializeVectors(positions, lower_p, higher_p);
    initializeVectors(velocities, lower_v, higher_v);

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

    /*
    checkEqual(accelerationsCpu, accelerationsGpu, "accelerations");
    checkEqual(forcesCpu, forcesGpu, "forces");
    checkEqual(velocitiesCpu, velocitiesGpu, "velocities");
    */
    checkEqual(positions_cpu, positions_gpu, "final positions");

    /*for(int k = 0; k < n_dim; ++k)
        std::cout << accelerationsCpu[0][k] << " | ";
    std::cout << std::endl;
    std::cout << std::endl;
    for(int k = 0; k < n_dim; ++k)
        std::cout << accelerationsGpu[0][k] << " | ";
    std::cout << std::endl;
    std::cout << std::endl;

    for(int k = 0; k < n_dim; ++k)
        std::cout << forcesCpu[0][k] << " | ";
    std::cout << std::endl;
    std::cout << std::endl;
    for(int k = 0; k < n_dim; ++k)
        std::cout << forcesGpu[0][k] << " | ";
    std::cout << std::endl;
    std::cout << std::endl;*/

    /*
    if(std::equal(positions_cpu.begin(), positions_cpu.end(), positions_gpu.begin())) {
        std::cout<<std::endl<<std::endl<<"The results are the same."<<std::endl<<std::endl;
    } else {
        std::cout<<std::endl<<std::endl<<"!!!!! The results are NOT the same !!!!!"<<std::endl<<std::endl;
    }
    */
    /*
    std::cout<<"cpu: ";
    for (const auto& vec : forcesCpu)
        for (const auto& val : vec)
            std::cout << val << " ";
    std::cout << std::endl;
    std::cout<<"gpu: ";
    for (const auto& vec : forcesGpu)
        for (const auto& val : vec)
            std::cout << val << " ";
    std::cout << std::endl;
    */
    /*std::cout<<"init: ";
    for (const auto& vec : positions)
        for (const auto& val : vec)
            std::cout << val << " ";
    std::cout << std::endl;
    std::cout << std::endl;
    */

    std::cout<<std::endl<<std::endl;

    std::cout << "CPU computation took " << duration_cpu.count() << " milliseconds." << std::endl;
    std::cout << "GPU computation took " << duration_gpu.count() << " milliseconds." << std::endl;

    std::cout<<std::endl<<std::endl;

    return 0;
}
