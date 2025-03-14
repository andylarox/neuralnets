// neuralnets.cpp 
//
/**
 * @file artificial_neuron.cpp
 * @brief Implements an artificial neuron.
 *
 * Defines a Neuron class that simulates an artificial neuron using a selectable
 * activation function. 
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <string>

 /**
  * @enum ActivationType
  * @brief Defines the types of activation functions available for the neuron.
  */
enum ActivationType {
    SIGMOID,     ///< Sigmoid activation function.
    TANH,        ///< Hyperbolic tangent activation function.
    RELU,        ///< Rectified Linear Unit activation function.
    LEAKY_RELU,  ///< Leaky Rectified Linear Unit activation function.
    LINEAR       ///< Linear activation function.
};

/**
 * @class Neuron
 * @brief An artificial neuron.
 *
 * The Neuron class encapsulates the weights, bias, and activation function required to compute the
 * output from an input vector. The weights and bias are initialised randomly upon object
 * construction.
 */
class Neuron {
public:
    /**
     * @brief Constructs a Neuron with a specified number of inputs and an activation type.
     *
     * This constructor initialises the weights and bias with random values in the range [-1, 1]. The
     * default activation function is SIGMOID if none is specified.
     *
     * @param numInputs The number of input connections to the neuron.
     * @param actType The activation function type to be used (default is SIGMOID).
     */
    Neuron(int numInputs, ActivationType actType = SIGMOID)
        : weights(numInputs), bias(0.0), activationType(actType) {
        // Randomly initialise weights and bias.
        for (auto& weight : weights) {
            // Random value between -1 and 1.
            weight = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }
        bias = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    /**
     * @brief Applies the selected activation function to a given input value.
     *
     * This function calculates the output of the chosen activation function for a given input.
     *
     * @param x The input value.
     * @return The activated value.
     */
    double activate(double x) {
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

    /**
     * @brief Computes the output of the neuron given an input vector.
     *
     * The output is computed as the activation of the weighted sum of the inputs plus the bias.
     *
     * @param inputs A vector of input values.
     * @return The output of the neuron.
     */
    double forward(const std::vector<double>& inputs) {
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

    /**
     * @brief Prints the neuron's parameters to the standard output.
     *
     * Outputs the current weights, bias, and the selected activation function being used.
     */
    void printParameters() const {
        std::cout << "Weights: ";
        for (const auto& w : weights)
            std::cout << w << " ";
        std::cout << "\nBias: " << bias << std::endl;
        std::cout << "Activation Function: " << activationTypeToString() << std::endl;
    }

    /**
     * @brief Sets the activation function type for the neuron.
     *
     * Allows changing the activation function type for the neuron.
     *
     * @param actType The new activation function type.
     */
    void setActivationType(ActivationType actType) {
        activationType = actType;
    }

private:
    std::vector<double> weights;   ///< The weights associated with each input.
    double bias;                   ///< The bias term.
    ActivationType activationType; ///< The type of activation function to apply.

    /**
     * @brief Converts the activation type to a string representation.
     *
     * @return A string that represents the activation function type.
     */
    std::string activationTypeToString() const {
        switch (activationType) {
        case SIGMOID: return "Sigmoid";
        case TANH: return "Tanh";
        case RELU: return "ReLU";
        case LEAKY_RELU: return "Leaky ReLU";
        case LINEAR: return "Linear";
        default: return "Unknown";
        }
    }
};

/**
 * @brief Main entry point
 *
 * Seeds the rng, creates Neuron objects with different
 * activation functions, prints their parameters, and computes the output for a sample input vector.
 *
 * @return 0 on successful execution.
 */
int main() {
    // Seed the random number generator.
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Example input vector.
    std::vector<double> input = { 0.5, -0.3, 0.8 };

    // Create neurons with different activation functions.
    Neuron sigmoidNeuron(3, SIGMOID);
    Neuron tanhNeuron(3, TANH);
    Neuron reluNeuron(3, RELU);
    Neuron leakyReluNeuron(3, LEAKY_RELU);
    Neuron linearNeuron(3, LINEAR);

    // Print parameters and outputs for each neuron.
    std::cout << "Sigmoid Neuron:" << std::endl;
    sigmoidNeuron.printParameters();
    std::cout << "Output: " << sigmoidNeuron.forward(input) << "\n\n";

    std::cout << "Tanh Neuron:" << std::endl;
    tanhNeuron.printParameters();
    std::cout << "Output: " << tanhNeuron.forward(input) << "\n\n";

    std::cout << "ReLU Neuron:" << std::endl;
    reluNeuron.printParameters();
    std::cout << "Output: " << reluNeuron.forward(input) << "\n\n";

    std::cout << "Leaky ReLU Neuron:" << std::endl;
    leakyReluNeuron.printParameters();
    std::cout << "Output: " << leakyReluNeuron.forward(input) << "\n\n";

    std::cout << "Linear Neuron:" << std::endl;
    linearNeuron.printParameters();
    std::cout << "Output: " << linearNeuron.forward(input) << "\n\n";

    return 0;
}
