#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <array>
#include <fstream>
#include <chrono>
#include <string>

// parameters
const double g = 6.67e-11;
const int n = 50;
const int n_dim = 2;
const double delta_t = 1.0;
const int n_simulations = 100;
const double lower_m = 1e-6;
const double higher_m = 1e6;
const double lower_p = -1e-1;
const double higher_p = 1e-1;
const double lower_v = -1e-4;
const double higher_v = 1e-4;

// structures
using Vector = std::array<double, n_dim>;
using Positions = std::array<Vector, n>;
using Velocities = std::array<Vector, n>;
using Forces = std::array<Vector, n>;
using Masses = std::array<double, n>;
using Accelerations = std::array<Vector, n>;

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
    for (int i = 0; i < n; ++i) {
        Vector sum = {};
        for (int j = 0; j < n; ++j) {
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

void updateAccelerations(const Forces& forces, const Masses& masses, Positions& accelerations) {
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n_dim; ++k) {
            accelerations[i][k] = forces[i][k] / masses[i];
        }
    }
}

void updateVelocities(Velocities& velocities, const Positions& accelerations, double delta_t) {
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n_dim; ++k) {
            velocities[i][k] += accelerations[i][k] * delta_t;
        }
    }
}

void updatePositions(Positions& positions, const Velocities& velocities, double delta_t) {
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n_dim; ++k) {
            positions[i][k] += velocities[i][k] * delta_t;
        }
    }
}

void printBodies(const Masses& masses, const Positions& positions, const Velocities& velocities) {
    for (int i = 0; i < n; ++i) {
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
    for (int i = 0; i < n; ++i) {
        output_str += std::to_string(time) + " " + std::to_string(i) + " ";
        for (const double& pos : positions[i]) {
            output_str += std::to_string(pos) + " ";
        }
        output_str += "\n";
    }
}

void runSimulation(Masses& masses, Positions& positions, Velocities& velocities) {
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
        updateAccelerations(forces, masses, accelerations);
        updateVelocities(velocities, accelerations, delta_t);
        updatePositions(positions, velocities, delta_t);

        savePositions(output_str, positions, absolute_t);
    }

    positions_file << output_str;
    positions_file.close();
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

    // simulation run
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(masses, positions, velocities);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Computation took " << duration.count() << " milliseconds." << std::endl;

    return 0;
}
