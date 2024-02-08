import math
#I used ChatGPT to help me write this code. I am going to have it explain the code to me step
#by step so that I can better understand the code and why the code is written this way.
#I had to really work on giving the correct prompts so that the code would run the way
#it needed to run per the homework.

def Simpson(f, a, b, n=1000):
    """
    Calculate the definite integral of f from a to b using Simpson's rule.

    Args:
    f (function): The function to integrate.
    a (float): The lower limit of integration.
    b (float): The upper limit of integration.
    n (int): The number of intervals for integration. Default is 1000.

    Returns:
    float: The approximate integral value.
    """
    if n % 2 == 1:
        raise ValueError("n must be even")

    h = (b - a) / n
    x = [a + i * h for i in range(n + 1)]
    integral = f(a) + f(b)
    for i in range(1, n, 2):
        integral += 4 * f(x[i])
    for i in range(2, n - 1, 2):
        integral += 2 * f(x[i])
    integral *= h / 3
    return integral


def t_distribution_probability(df, z):
    """
    Calculate the probability using the t-distribution equation.

    Args:
    df (int): Degrees of freedom.
    z (float): Value of z.

    Returns:
    float: Probability.
    """
    if df <= 0:
        raise ValueError("Degrees of freedom (df) must be greater than 0.")

    # Define the t-distribution function for integration
    def t_distribution_function(x):
        return math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2)) * (1 + (x ** 2) / df) ** (
                    -(df + 1) / 2)

    # Integrate using Simpson's rule
    probability = Simpson(t_distribution_function, 0, z, 1000) + 0.5  # Adjust the number of intervals as needed

    return probability


def main():
    """
    Main function to prompt the user for input and calculate the t-distribution probability.
    """
    print("T-Distribution Probability Calculator")

    # Prompt user for input for three different degrees of freedom
    for i in range(3):
        df = int(input(f"Enter degrees of freedom {i + 1}: "))

        # For each degree of freedom, prompt for three different z values
        for j in range(3):
            z = float(input(f"Enter value of z {j + 1} for df={df}: "))

            # Calculate the t-distribution probability
            try:
                probability = t_distribution_probability(df, z)
                print(f"The probability for df={df} and z={z} is: {probability:.4f}")
            except ValueError as e:
                print(e)


if __name__ == "__main__":
    main()
