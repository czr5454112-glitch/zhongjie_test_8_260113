import os
import time

from map import Map
from ccbs import CCBS
from structs import Task, Solution


def main():
    map_path = os.path.join("instances", "roadmaps", "sparse", "map.xml")
    task_path = os.path.join("instances", "roadmaps", "sparse", "test", "10", "1_task-10.xml")

    world = Map(map_path)
    solver = CCBS(world)
    solver.config.use_rl = True
    solver.config.rl_model_path = "ppo_road-sparse.zip"
    solver.solution = Solution()

    task = Task()
    task.load_from_file(task_path)

    start = time.time()
    solution = solver.find_solution(task)
    elapsed = time.time() - start

    print("found:", solution.found)
    print("flowtime:", solution.flowtime)
    print("makespan:", solution.makespan)
    print("elapsed_seconds:", elapsed)

    if solution.paths:
        for idx, path in enumerate(solution.paths):
            print(f"Agent {idx}: cost={path.cost}")
            for node in path.nodes:
                print(f"  node={node.id}, time={node.g}")
    else:
        print("No paths available")


if __name__ == "__main__":
    main()
