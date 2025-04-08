"""
This program implements the non-dominated-sorting in Deb's NSGA-II algorithm
CISC455/851 Winter'25
"""

from functools import partial
from random import uniform
import matplotlib.pyplot as plt

def dominates(sol1, sol2, eval_func1, eval_func2):
    """Checks if sol1 dominates sol2 based on two objective functions.

    Args:
    - sol1 (tuple): First solution (genotype) to compare.
    - sol2 (tuple): Second solution (genotype) to compare.
    - eval_func1 (function): Evaluation function for objective 1.
    - eval_func2 (function): Evaluation function for objective 2.

    Returns:
    - bool: True if sol1 dominates sol2, otherwise False.
    """
    x1, y1 = eval_func1(sol1), eval_func2(sol1)
    x2, y2 = eval_func1(sol2), eval_func2(sol2)
    return True if (x1 <= x2 and y1 <= y2) and (x1 < x2 or y1 < y2) else False


def evaluation_functions():
    """Defines and returns the two objective evaluation functions.

    Returns:
    - eval_func1 (function): Function to extract the first value of the solution.
    - eval_func2 (function): Function to extract the second value of the solution.
    """
    return lambda x: x[0], lambda x: x[1]


def non_dominated_sorting(population, eval_func1, eval_func2):
    """Sorts the population into Pareto fronts using dominance relationships.

    Args:
    - population (list of tuples): List of candidate solutions.
    - eval_func1 (function): Evaluation function for objective 1.
    - eval_func2 (function): Evaluation function for objective 2.

    Returns:
    - list of lists: A list of Pareto fronts, where each front contains tuples of solutions.
    """
    dominate_func = partial(dominates, eval_func1=eval_func1, eval_func2=eval_func2)

    # x as a solution in population and the set of solutions that dominate x
    solutions = [(x, []) for x in population]

    # initialize fronts
    fronts = [[]]

    # for each solution calculate the domination relations
    for sol1 in solutions:
        for sol2 in solutions:
            # if sol2 dominates sol1, append to sol1's list of dominating solutions
            if dominate_func(sol2[0], sol1[0]):
                sol1[1].append(sol2[0])

    # assign all non-dominated solutions to the first Pareto front, fronts[0]
    for sol in solutions:
        if not sol[1]:
            fronts[0].append(sol[0])
            solutions = [x for x in solutions if x[0] != sol[0]]

    # assign points to the rest of the fronts
    while solutions:  # while "solutions" not empty, i.e., have not been assigned to a front
        new_front = []

        for sol in solutions:
            for front_sol in fronts[-1]:
                if front_sol in sol[1]:
                    sol[1].remove(front_sol)

            # if not dominated add to new front
            if not sol[1]:
                new_front.append(sol[0])
                solutions = [x for x in solutions if x[0] != sol[0]]

        fronts.append(new_front)

    # return the fronts
    return fronts


def plot_pareto_fronts(fronts):
    """Plots the solutions color-coded by their Pareto front.

    Args:
    - fronts (list of lists): A list of Pareto fronts, where each front contains tuples of solutions.

    Displays:
    - A scatter plot of Pareto fronts with different colors.
    """
    num_fronts = len(fronts)
    cmap = plt.get_cmap("tab10", num_fronts)  # Use a categorical colormap for better distinction
    plt.figure(figsize=(8, 6))

    for i, front in enumerate(fronts):
        x_vals = [p[0] for p in front]
        y_vals = [p[1] for p in front]
        plt.scatter(x_vals, y_vals, color=cmap(i), label=f'Front {i + 1}')

    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title("Pareto Fronts")
    plt.legend()
    plt.show()


def initialize_population(num_individuals):
    """Generates a random population of individuals in a 2D space.

    Args:
    - num_individuals (int): Number of individuals to generate.

    Returns:
    - list of tuples: List of generated solutions with two values each.
    """
    return [(round(uniform(0, 1), 2), round(uniform(0, 1), 2)) for _ in range(num_individuals)]


def main():
    """Executes the process of generating a population, sorting it into Pareto fronts, and visualizing the results."""
    # Define evaluation functions
    eval_func1, eval_func2 = evaluation_functions()

    # initialize a population of individuals
    population = initialize_population(25)

    # call function to generate the Pareto fronts assignment
    fronts = non_dominated_sorting(population, eval_func1, eval_func2)

    print(f"population: {population}")
    for idx, front in enumerate(fronts):
        print(f"front {idx}: {front}")

    # Visualize the Pareto fronts
    plot_pareto_fronts(fronts)


if __name__ == '__main__':
    main()
