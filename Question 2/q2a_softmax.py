import numpy as np
from scipy.special import softmax as scipy_softmax

def softmax(x):
    """Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    x_max = np.max(x, axis=-1, keepdims=True)  # Keep dimensions for broadcasting
    x_stable = x - x_max  # Subtract the max for numerical stability
    exp_x = np.exp(x_stable)  # Compute the exponentials
    softmax_x = exp_x / np.sum(exp_x, axis=-1, keepdims=True)  # Normalize to get softmax
    assert x.shape == orig_shape
    return softmax_x

    

def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print("Running basic tests...")
    test1 = softmax(np.array([1, 2]))
    print(test1)
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001, 1002], [3, 4]]))
    print(test2)
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001, -1002]]))
    print(test3)
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print("You should be able to verify these results by hand!\n")


def your_softmax_test():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print("Running your tests...")
    for i in range(1000000):
        # Generate random test cases: 1D and 2D arrays
        test_cases = [
            np.random.randn(10),          # Random 1D vector
            np.random.randn(5, 10),       # Random 2D matrix
        ]
        
        for i, test_case in enumerate(test_cases):
            # Apply custom stable_softmax
            custom_result = softmax(test_case)
            # Apply scipy softmax
            scipy_result = scipy_softmax(test_case, axis=-1)
            # Compare results
            if not np.allclose(custom_result, scipy_result, atol=1e-6):
                print(f"Test case {i + 1} failed!")
                print("Custom result:\n", custom_result)
                print("Scipy result:\n", scipy_result)
                return False
    
    print("All test cases passed!")
    return True

if __name__ == "__main__":
    test_softmax_basic()
    your_softmax_test()
