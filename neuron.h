#ifndef NEURON_H
#define NEURON_H

#include <vector>
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
    Neuron(int numInputs, ActivationType actType = SIGMOID);

    /**
     * @brief Applies the selected activation function to a given input value.
     *
     * This function calculates the output of the chosen activation function for a given input.
     *
     * @param x The input value.
     * @return The activated value.
     */
    double activate(double x);

    /**
     * @brief Computes the output of the neuron given an input vector.
     *
     * The output is computed as the activation of the weighted sum of the inputs plus the bias.
     * Additionally, the input, weighted sum, and output are stored for use in backpropagation.
     *
     * @param inputs A vector of input values.
     * @return The output of the neuron.
     */
    double forward(const std::vector<double>& inputs);

    /**
     * @brief Prints the neuron's parameters to the standard output.
     *
     * Outputs the current weights, bias, and the selected activation function being used.
     */
    void printParameters() const;

    /**
     * @brief Sets the activation function type for the neuron.
     *
     * Allows changing the activation function type for the neuron.
     *
     * @param actType The new activation function type.
     */
    void setActivationType(ActivationType actType);

    /**
     * @brief Computes the derivative of the activation function at the last computed weighted sum.
     *
     * This function utilises the stored output from the last forward pass.
     *
     * @return The derivative of the activation function.
     */
    double activationDerivative() const;

    /**
     * @brief Updates the weights and bias based on the computed delta and learning rate.
     *
     * @param delta The error term (gradient) for this neuron.
     * @param learningRate The learning rate for the update.
     */
    void updateWeights(double delta, double learningRate);

private:
    std::vector<double> weights;   ///< The weights associated with each input.
    double bias;                   ///< The bias term.
    ActivationType activationType; ///< The type of activation function to apply.

    // Variables to store forward pass values for use in backpropagation.
    std::vector<double> lastInputs;   ///< The last input vector.
    double lastWeightedSum;           ///< The weighted sum before activation.
    double lastOutput;                ///< The output after activation.

    /**
     * @brief Converts the activation type to a string representation.
     *
     * @return A string that represents the activation function type.
     */
    std::string activationTypeToString() const;
};

#endif // NEURON_H
