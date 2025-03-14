/**
 * @file neuralnets.cpp
 * @brief Implements an artificial neuron with backpropagation and error computation.
 *
 * Defines the member functions of the Neuron class, error metric functions, and demonstrates a simple
 * feedforward network training on a single sample.
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
    : weights(numInputs), bias(0.0), activationType(actType),
    lastWeightedSum(0.0), lastOutput(0.0) {
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
    // Save inputs for use during backpropagation.
    lastInputs = inputs;
    // Compute the weighted sum.
    lastWeightedSum = bias;
    for (std::size_t i = 0; i < weights.size(); i++) {
        lastWeightedSum += weights[i] * inputs[i];
    }
    // Apply activation.
    lastOutput = activate(lastWeightedSum);
    return lastOutput;
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

double Neuron::activationDerivative() const {
    // Compute derivative based on activation type.
    // For sigmoid, derivative is output * (1 - output).
    switch (activationType) {
    case SIGMOID:
        return lastOutput * (1.0 - lastOutput);
    case TANH:
        return 1.0 - (lastOutput * lastOutput);
    case RELU:
        return (lastWeightedSum > 0) ? 1.0 : 0.0;
    case LEAKY_RELU:
        return (lastWeightedSum > 0) ? 1.0 : 0.01;
    case LINEAR:
        return 1.0;
    default:
        return 1.0;
    }
}

void Neuron::updateWeights(double delta, double learningRate) {
    // Update weights using gradient descent.
    for (std::size_t i = 0; i < weights.size(); i++) {
        weights[i] -= learningRate * delta * lastInputs[i];
    }
    // Update bias.
    bias -= learningRate * delta;
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
// Error metric functions
//------------------------------------------------------------------------------

enum ErrorMetric {
    MSE,    ///< Mean Squared Error.
    MAE     ///< Mean Absolute Error.
};

/**
 * @brief Computes the error given a target and output, using the specified error metric.
 *
 * @param target The target value.
 * @param output The output value from the network.
 * @param metric The error metric to use.
 * @return The computed error.
 */
double computeError(double target, double output, ErrorMetric metric) {
    switch (metric) {
    case MSE:
        return 0.5 * (target - output) * (target - output);
    case MAE:
        return std::abs(target - output);
    default:
        return 0.5 * (target - output) * (target - output);
    }
}

/**
 * @brief Computes the derivative of the error with respect to the output.
 *
 * For MSE, the derivative is (output - target). For MAE, it is 1 or -1.
 *
 * @param target The target value.
 * @param output The output value from the network.
 * @param metric The error metric to use.
 * @return The derivative of the error with respect to the output.
 */
double errorDerivative(double target, double output, ErrorMetric metric) {
    switch (metric) {
    case MSE:
        return output - target;
    case MAE:
        return (output >= target) ? 1.0 : -1.0;
    default:
        return output - target;
    }
}

//------------------------------------------------------------------------------
// Main entry point demonstrating a simple training step with backpropagation
//------------------------------------------------------------------------------

/**
 * @brief Main entry point of the programme.
 *
 * Demonstrates a single training iteration on a simple feedforward network with:
 * - 3 input nodes.
 * - Hidden layer: 2 neurons, each receiving 3 inputs.
 * - Output layer: 1 neuron receiving 2 inputs (the outputs from the hidden neurons).
 *
 * The programme computes the network output, the error compared to a target value, and performs
 * backpropagation to update the weights.
 *
 * @return 0 on successful execution.
 */
int main() {
    // Seed the random number generator.
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Learning rate for weight updates.
    double learningRate = 0.1;

    // Example input vector with 3 values.
    std::vector<double> input = { 0.5, -0.3, 0.8 };

    // Target value for training.
    double target = 0.7;

    // Create the hidden layer with 2 neurons (each receiving 3 inputs).
    Neuron hiddenNeuron1(3, SIGMOID);
    Neuron hiddenNeuron2(3, SIGMOID);

    // Forward pass through hidden layer.
    double hiddenOutput1 = hiddenNeuron1.forward(input);
    double hiddenOutput2 = hiddenNeuron2.forward(input);

    // Create the output neuron (receiving 2 inputs from the hidden layer).
    Neuron outputNeuron(2, SIGMOID);
    std::vector<double> hiddenOutputs = { hiddenOutput1, hiddenOutput2 };

    // Forward pass through output neuron.
    double networkOutput = outputNeuron.forward(hiddenOutputs);

    // Compute error using Mean Squared Error (MSE) metric.
    ErrorMetric metric = MSE;
    double error = computeError(target, networkOutput, metric);
    std::cout << "Initial Network output: " << networkOutput << std::endl;
    std::cout << "Initial Error: " << error << std::endl;

    // ---- Backpropagation ----
    // For the output neuron:
    // delta_output = (dE/dOut) * f'(net_output)
    double outputErrorDerivative = errorDerivative(target, networkOutput, metric);
    double outputDelta = outputErrorDerivative * outputNeuron.activationDerivative();

    // Update weights for the output neuron.
    outputNeuron.updateWeights(outputDelta, learningRate);

    // For the hidden neurons:
    // In a full implementation the delta for a hidden neuron would depend on the weight from that neuron to the output.
    // Here, for demonstration, we assume a direct contribution (i.e. using a placeholder weight of 1.0).
    double hiddenDelta1 = hiddenNeuron1.activationDerivative() * (outputDelta * 1.0);
    double hiddenDelta2 = hiddenNeuron2.activationDerivative() * (outputDelta * 1.0);

    // Update weights for the hidden neurons.
    hiddenNeuron1.updateWeights(hiddenDelta1, learningRate);
    hiddenNeuron2.updateWeights(hiddenDelta2, learningRate);

    // Forward pass after the weight updates.
    hiddenOutput1 = hiddenNeuron1.forward(input);
    hiddenOutput2 = hiddenNeuron2.forward(input);
    hiddenOutputs = { hiddenOutput1, hiddenOutput2 };
    networkOutput = outputNeuron.forward(hiddenOutputs);
    error = computeError(target, networkOutput, metric);

    std::cout << "\nAfter one backpropagation step:" << std::endl;
    std::cout << "Target: " << target << std::endl;
    std::cout << "Network output: " << networkOutput << std::endl;
    std::cout << "Error: " << error << std::endl;

    return 0;
}
