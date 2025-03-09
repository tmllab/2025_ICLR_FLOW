def add(a, b):
    return a + b # Intentional bug

# Test function to validate code behavior
# Ensures no GUI or unwanted side-effects are triggered
def run_tests():
    if '__main__' not in __name__:
        return  # Prevents accidental execution in GUI environments

    failures = []

    try:
        assert add(2, 3) == 5, "Test failed: add(2,3) should return 5"
    except AssertionError as e:
        failures.append(str(e))

    try:
        assert add(-1, 1) == 0, "Test failed: add(-1,1) should return 0"
    except AssertionError as e:
        failures.append(str(e))

    try:
        assert add(0, 0) == 0, "Test failed: add(0,0) should return 0"
    except AssertionError as e:
        failures.append(str(e))

    if failures:
        print("'Error executing code:'")
        for f in failures:
            print(f)
    else:
        print("All tests passed!")

# Execute tests only if conditions are met
if __name__ == "__main__":
    run_tests()