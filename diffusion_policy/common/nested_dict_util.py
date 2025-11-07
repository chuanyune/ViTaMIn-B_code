import functools

def nested_dict_map(f, x):
    """
    Map f over all leaf of nested dict x
    """

    if not isinstance(x, dict):
        return f(x)
    y = dict()
    for key, value in x.items():
        y[key] = nested_dict_map(f, value)
    return y

def nested_dict_reduce(f, x):
    """
    Characteristics of reduce:
    Combines multiple values into a single value
    Is an iterative process
    Requires a binary operation function (takes two parameters)
    Operation must satisfy associative law ((a+b)+c = a+(b+c))
    Map f over all values of nested dict x, and reduce to a single value
    """
    if not isinstance(x, dict):
        return x

    reduced_values = list()
    for value in x.values():
        reduced_values.append(nested_dict_reduce(f, value))
    y = functools.reduce(f, reduced_values)
    return y


def nested_dict_check(f, x):
    bool_dict = nested_dict_map(f, x)
    result = nested_dict_reduce(lambda x, y: x and y, bool_dict)
    return result


"""
# Typical deep learning model parameter structure
model_params = {
    'encoder': {
        'conv1': {
            'weights': tensor(...),
            'bias': tensor(...)
        },
        'conv2': {
            'weights': tensor(...),
            'bias': tensor(...)
        }
    },
    'decoder': {
        'deconv1': {...},
        'deconv2': {...}
    }
}

# Use nested_dict_map for parameter operations
# Example: Move all parameters to GPU
gpu_params = nested_dict_map(lambda x: x.cuda(), model_params)

# Typical configuration file structure
config = {
    'training': {
        'batch_size': 32,
        'learning_rate': {
            'initial': 0.001,
            'decay': 0.95
        }
    },
    'model': {
        'architecture': {
            'layers': [64, 128, 256],
            'activation': 'relu'
        }
    }
}

# Use nested_dict_check to validate configuration
is_valid = nested_dict_check(
    lambda x: isinstance(x, (int, float, str)), 
    config
)


"""
