import tsplib95


if __name__ == "__main__":
    problem = tsplib95.load("data/kroA100.tsp")

    print(problem.get_nodes())
