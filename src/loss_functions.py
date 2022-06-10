from matrix import Matrix


def mean_squared_error_derivative(y_true: Matrix, y_pred: Matrix):
    result = y_pred.add(y_true.apply(lambda x: -x))
    result = result.multiply(2)

    size = result.columns
    result = result.apply(lambda x: x/size)

    return result

def mean_squared_error(expected: Matrix, computed: Matrix):
    result = computed.add(expected.apply(lambda x: -x))
    result = result.apply(lambda x: x**2)

    return result.multiply(0.5)

FUNCTIONS = {
    'mse': {
        'function': mean_squared_error,
        'derivative': mean_squared_error_derivative
    }
}