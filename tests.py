import numpy as np

def derivate(function, x, delta=1e-5):
    """Returns the derivative of a function at x."""
    return (function(x + delta) - function(x - delta)) / (2 * delta)

def test_scalar_derivate():
    def function_square(x):
        """Returns the square of x."""
        return x * x
    # Test the derivate function
    assert derivate(function_square, 2) - 4 < 1e-5, "Test failed!"
    assert derivate(function_square, -3) - -6 < 1e-5, "Test failed!"
    assert derivate(function_square, 0) - 0 < 1e-5, "Test failed!"
    # Test the derivate function with a different function
    def function_cube(x):
        """Returns the cube of x."""
        return x * x * x
    assert derivate(function_cube, 2) - 12 < 1e-5, "Test failed!"
    assert derivate(function_cube, -3) - 27 < 1e-5, "Test failed!"
    assert derivate(function_cube, 0) - 0 < 1e-5, "Test failed!"
    # Test the derivate function with a different function and delta
    def function_sine(x):
        """Returns the sine of x."""
        return np.sin(x)
    assert derivate(function_sine, 0) - 1 < 1e-5, "Test failed!"
    assert derivate(function_sine, np.pi / 2) - 0 < 1e-5, "Test failed!"
    assert derivate(function_sine, np.pi) - -1 < 1e-5, "Test failed!"
    # Test the derivate function with a different function 
    def function_cosine(x):
        """Returns the cosine of x."""
        return np.cos(x)
    assert derivate(function_cosine, 0) - 0 < 1e-5, "Test failed!"
    assert derivate(function_cosine, np.pi / 2) - -1 < 1e-5, "Test failed!"
    assert derivate(function_cosine, np.pi) - 0 < 1e-5, "Test failed!"

def test_vector_partial_derivate():
    # Test the derivate function with a different function
    def function_vector2(x):
        """Returns the vector [x[0]**2, x[1]**2, x[2]**2]."""
        return np.array([x[0]**2, x[1]**2, x[2]**2])
    assert np.allclose(derivate(function_vector2, np.array([1, 2, 3])), np.array([2, 4, 6]), atol=1e-5), "Test failed!"
    assert np.allclose(derivate(function_vector2, np.array([-1, -2, -3])), np.array([-2, -4, -6]), atol=1e-5), "Test failed!"
    assert np.allclose(derivate(function_vector2, np.array([0, 0, 0])), np.array([0, 0, 0]), atol=1e-5), "Test failed!"

if __name__ == "__main__":
    test_scalar_derivate()
    test_vector_partial_derivate()
    print("All tests passed!")
