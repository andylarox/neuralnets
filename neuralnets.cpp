/**
 * @file neuralnets.cpp
 * @brief Implements an artificial neuron and demonstrates a simple feedforward network.
 *
 * Defines the member functions of the Neuron class and the main entry point of the programme.
 */

#include "neuron.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <string>

 //------------------------------------------------------------------------------
 // Neuron class member function definitions
 //------------------------------------------------------------------------------

Neuron::Neuron(int numInputs, ActivationType actType)
    : weights(numInputs), bias(0.0), activationType(actType) {
    // Randomly initialise weights and bias.
    for (auto& weight : weights) {
        // Random value between -1 and 1.
        weight = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
    bias = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

double Neuron::activate(double x) {
    switch (activationType) {
    case SIGMOID:
        return 1.0 / (1.0 + std::exp(-x));
    case TANH:
        return std::tanh(x);
    case RELU:
        return (x > 0) ? x : 0.0;
    case LEAKY_RELU:
        return (x > 0) ? x : 0.01 * x;
    case LINEAR:
        return x;
    default:
        return x; // Fallback to linear activation.
    }
}

double Neuron::forward(const std::vector<double>& inputs) {
    // Verify that the input size matches the number of weights.
    if (inputs.size() != weights.size()) {
        std::cerr << "Error: Input size does not match number of weights." << std::endl;
        return 0.0;
    }
    double sum = bias;
    // Compute the weighted sum.
    for (std::size_t i = 0; i < weights.size(); i++) {
        sum += weights[i] * inputs[i];
    }
    // Apply the selected activation function.
    return activate(sum);
}

void Neuron::printParameters() const {
    std::cout << "Weights: ";
    for (const auto& w : weights)
        std::cout << w << " ";
    std::cout << "\nBias: " << bias << std::endl;
    std::cout << "Activation Function: " << activationTypeToString() << std::endl;
}

void Neuron::setActivationType(ActivationType actType) {
    activationType = actType;
}

std::string Neuron::activationTypeToString() const {
    switch (activationType) {
    case SIGMOID: return "Sigmoid";
    case TANH: return "Tanh";
    case RELU: return "ReLU";
    case LEAKY_RELU: return "Leaky ReLU";
    case LINEAR: return "Linear";
    default: return "Unknown";
    }
}

//------------------------------------------------------------------------------
// Main entry point
//------------------------------------------------------------------------------

/**
 * @brief Main entry point
 *
 * Seeds the rng, creates Neuron objects with different activation functions, prints their parameters,
 * and computes the output for a sample input vector.
 *
 * The network architecture is as follows:
 * - 3 input nodes.
 * - Hidden layer: 2 neurons, each receiving 3 inputs.
 * - Output layer: 1 neuron receiving 2 inputs (the outputs from the hidden neurons).
 *
 * @return 0 on successful execution.
 */
int main() {
    // Seed the random number generator.
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Example input vector with 3 values.
    std::vector<double> input = { 0.5, -0.3, 0.8 };

    // Create the hidden layer with 2 neurons.
    // Each hidden neuron receives 3 inputs.
    Neuron hiddenNeuron1(3, SIGMOID);
    Neuron hiddenNeuron2(3, SIGMOID);

    // Compute the outputs of the hidden neurons.
    double hiddenOutput1 = hiddenNeuron1.forward(input);
    double hiddenOutput2 = hiddenNeuron2.forward(input);

    // Display hidden layer outputs.
    std::cout << "Hidden Neuron 1 output: " << hiddenOutput1 << std::endl;
    std::cout << "Hidden Neuron 2 output: " << hiddenOutput2 << std::endl;

    // Create the output layer neuron.
    // The output neuron receives 2 inputs (the outputs from the hidden layer).
    Neuron outputNeuron(2, SIGMOID);

    // Prepare the vector of hidden layer outputs.
    std::vector<double> hiddenOutputs = { hiddenOutput1, hiddenOutput2 };

    // Compute the output of the network.
    double networkOutput = outputNeuron.forward(hiddenOutputs);
    std::cout << "Network output: " << networkOutput << std::endl;

    return 0;
}
