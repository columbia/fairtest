class Correlation():
    """Base class to be used to study the input/output correlations"""
    inputs = []
    outputs = []

    def input(func):
        """A decorator to mark the function whose input will be used to search
        for correlations
        """
        def wrapper(*args, **kwargs):
            inputs = Correlation.inputs
            inputs.append((args, kwargs))
            print("In:",inputs)
            return func(*args, **kwargs)
        return wrapper

    def output(func):
        """A decorator to mark the function whose output will be used to search
        for correlations
        """
        def wrapper(*args, **kwargs):
            outputs = Correlation.outputs
            output = func(*args, **kwargs)
            outputs.append(output)
            print("Out:",outputs)
            return output
        return wrapper
