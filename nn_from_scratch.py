import math

inputs_ex = [1.66, 1.56]
weights_ex = [[1.45, -0.66], [1.65, -0.16], [1.95, -0.96]]
biases_ex = [0.0, 0.3, 0.6]


def compute_z(input_vector, weights, bias):
    paired_list = zip(input_vector, weights)
    assert len(input_vector) == len(weights), "Inputs and weights must match!"
    sum_paired_list = sum(x * w for x, w in paired_list)
    return sum_paired_list + bias


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def neuron(input_vector, weights, bias):
    z = compute_z(input_vector, weights, bias)
    return sigmoid(z)


def dense_layer(input_vector, weights, biases):
    output_nodes = []
    for weight, bias in zip(weights, biases):
        output = neuron(input_vector, weight, bias)
        output_nodes.append(output)
    return output_nodes


if __name__ == "__main__":
    layer = dense_layer(inputs_ex, weights_ex, biases_ex)
    print(layer)