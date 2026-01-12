import os

import numpy as np

from structs import *
from sipp import SIPP
from map import Map
from config import Config
import math
import time
import xml.etree.ElementTree as ET
from Animation import *
from stable_baselines3.common.monitor import Monitor
import pandas as pd
import statistics
import multiprocessing as mp
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ccbsenv import CCBSEnv


class CCBS:
    CN_INFINITY = float('inf')
    CN_EPSILON = 1e-9

    def __init__(self, map: Map) -> None:
        self.tree = CBS_Tree()
        self.planner = SIPP(map)
        self.solution = None
        self.map = map

        self.config = Config()

        self.verbose = False

    def init_root(self, task: Task) -> bool:
        if self.verbose:
            print("Finding root solution...")
        root = CBS_Node()
        path = sPath()

        for agent in task.agents:
            path = self.planner.find_path(agent, [])
            if path.cost < 0:
                return False
            root.paths.append(path)
            root.cost += path.cost

        root.low_level_expanded = 0
        root.parent = None
        root.id = 1
        root.id_str = "1"

        conflicts = self.get_all_conflicts(root.paths, -1)
        root.conflicts_num = len(conflicts)

        for conflict in conflicts:
            if not self.config.use_cardinal:
                root.conflicts.append(conflict)
            else:
                pathA = self.planner.find_path(task.get_agent(conflict.agent1),
                                               [self.get_constraint(conflict.agent1, conflict.move1, conflict.move2)])
                pathB = self.planner.find_path(task.get_agent(conflict.agent2),
                                               [self.get_constraint(conflict.agent2, conflict.move2, conflict.move1)])

                if pathA.cost > root.paths[conflict.agent1].cost and pathB.cost > root.paths[conflict.agent2].cost:
                    conflict.overcost = min(pathA.cost - root.paths[conflict.agent1].cost,
                                            pathB.cost - root.paths[conflict.agent2].cost)
                    root.cardinal_conflicts.append(conflict)
                elif pathA.cost > root.paths[conflict.agent1].cost or pathB.cost > root.paths[conflict.agent2].cost:
                    root.semicard_conflicts.append(conflict)
                else:
                    root.conflicts.append(conflict)

        self.solution.init_cost = root.cost
        self.tree.add_node(root)
        return True

    def check_conflict(self, move1: Move, move2: Move) -> bool:
        startTimeA, endTimeA = move1.t1, move1.t2
        startTimeB, endTimeB = move2.t1, move2.t2

        m1x1, m1x2 = self.map.nodes[move1.id1].x, self.map.nodes[move1.id2].x
        m1y1, m1y2 = self.map.nodes[move1.id1].y, self.map.nodes[move1.id2].y
        m2x1, m2x2 = self.map.nodes[move2.id1].x, self.map.nodes[move2.id2].x
        m2y1, m2y2 = self.map.nodes[move2.id1].y, self.map.nodes[move2.id2].y

        A = [m1x1, m1y1]
        B = [m2x1, m2y1]

        VA = [(m1x2 - m1x1) / (move1.t2 - move1.t1 + self.CN_EPSILON), (m1y2 - m1y1) / (move1.t2 - move1.t1 + self.CN_EPSILON)]
        VB = [(m2x2 - m2x1) / (move2.t2 - move2.t1 + self.CN_EPSILON), (m2y2 - m2y1) / (move2.t2 - move2.t1 + self.CN_EPSILON)]

        if startTimeB > startTimeA:
            A[0] += VA[0] * (startTimeB - startTimeA)
            A[1] += VA[1] * (startTimeB - startTimeA)
            startTimeA = startTimeB
        elif startTimeB < startTimeA:
            B[0] += VB[0] * (startTimeA - startTimeB)
            B[1] += VB[1] * (startTimeA - startTimeB)
            startTimeB = startTimeA

        r = 2 * self.config.agent_size
        w = [B[0] - A[0], B[1] - A[1]]
        c = w[0] * w[0] + w[1] * w[1] - r ** 2

        if c < 0:
            # print(f"Checking conflict {move1} / {move2} -> c < 0")
            return True

        v = [VA[0] - VB[0], VA[1] - VB[1]]
        a = v[0] * v[0] + v[1] * v[1]
        b = w[0] * v[0] + w[1] * v[1]
        dscr = b ** 2 - a * c

        if dscr - self.CN_EPSILON < 0:
            # print(f"Checking conflict {move1} / {move2} -> dscr <= 0, no conflict")
            return False

        ctime = (b - math.sqrt(dscr)) / a
        if -self.CN_EPSILON < ctime < min(endTimeB, endTimeA) - startTimeA + self.CN_EPSILON:
            # print(f"Checking conflict {move1} / {move2} -> 0 < ctime < move_duration_time")
            return True

        # print(f"Checking conflict {move1} / {move2} -> no conflict")
        return False

    def check_corridor_conflict(self, conflict: Conflict, task: Task) -> list:
        """

        :param conflict:
        :param task:
        :return: 返回corridor的路径，从agent1的起点到agent2的起点
        """
        agent1 = conflict.agent1
        start1 = task.get_agent(agent1).start_id
        goal1 = task.get_agent(agent1).goal_id
        agent2 = conflict.agent2
        start2 = task.get_agent(agent2).start_id
        goal2 = task.get_agent(agent2).goal_id

        move1 = conflict.move1
        move2 = conflict.move2
        m1x1, m1x2 = self.map.nodes[move1.id1].x, self.map.nodes[move1.id2].x
        m1y1, m1y2 = self.map.nodes[move1.id1].y, self.map.nodes[move1.id2].y
        m2x1, m2x2 = self.map.nodes[move2.id1].x, self.map.nodes[move2.id2].x
        m2y1, m2y2 = self.map.nodes[move2.id1].y, self.map.nodes[move2.id2].y

        dir_1 = (m1x2 - m1x1, m1y2 - m1y1)
        dir_2 = (m2x2 - m2x1, m2y2 - m2y1)

        collision_vertex = (m2x2, m2y2) == (m1x2, m1y2)  # vertex conflict
        collision_edge = (m1x1, m1y1) == (m2x2, m2y2) and (m1x2, m1y2) == (m2x1, m2y1)  # edge conflict

        corridor_1 = []
        corridor_2 = []
        corridor = []
        tmp_open_1 = []
        tmp_open_2 = []
        if collision_vertex:
            if len(self.map.get_valid_moves(move1.id2)) == 2:
                tmp_open_1.append(move1.id2)
                tmp_open_2.append(move1.id2)
        elif collision_edge:
            if len(self.map.get_valid_moves(move1.id1)) == 2:
                tmp_open_1.append(move1.id1)
            if len(self.map.get_valid_moves(move2.id1)) == 2:
                tmp_open_2.append(move2.id1)
        # 智能体相向碰撞，计算corridor长度
        while len(tmp_open_1) > 0 or len(tmp_open_2) > 0:
            if len(tmp_open_1) > 0:
                cur_node_id = tmp_open_1.pop()
                cur_x, cur_y = self.map.nodes[cur_node_id].x, self.map.nodes[cur_node_id].y
                if len(self.map.get_valid_moves(cur_node_id)) != 2 or cur_node_id in [start1, goal1, start2, goal2]:
                    corridor_end_1 = cur_node_id
                    corridor_1.insert(0, corridor_end_1)
                else:
                    for move in self.map.get_valid_moves(cur_node_id):
                        tmp_dir = cur_x - move.x, cur_y - move.y
                        # 方向和move1相同
                        if tmp_dir[0] == dir_1[0] and tmp_dir[1] == dir_1[1]:
                            tmp_open_1.append(move.id)
                        else:
                            continue
                    corridor_1.insert(0, cur_node_id)
            if len(tmp_open_2) > 0:
                cur_node_id = tmp_open_2.pop()
                cur_x, cur_y = self.map.nodes[cur_node_id].x, self.map.nodes[cur_node_id].y
                if len(self.map.get_valid_moves(cur_node_id)) != 2 or cur_node_id in [start1, goal1, start2, goal2]:
                    corridor_end_2 = cur_node_id
                    corridor_2.append(corridor_end_2)
                else:
                    for move in self.map.get_valid_moves(cur_node_id):
                        tmp_dir = cur_x - move.x, cur_y - move.y
                        # 方向和move2相同
                        if tmp_dir[0] == dir_2[0] and tmp_dir[1] == dir_2[1]:
                            tmp_open_2.append(move.id)
                        else:
                            continue
                    corridor_2.append(cur_node_id)
        if len(corridor_1) > 0 and len(corridor_2) > 0:
            if corridor_1[-1] == corridor_2[0]:
                corridor.extend(corridor_1[:-1])
                corridor.extend(corridor_2)
            else:
                corridor.extend(corridor_1)
                corridor.extend(corridor_2)
        elif len(corridor_1) > 0 and len(corridor_2) == 0:
            corridor = corridor_1
            corridor.append(move2.id1)
        elif len(corridor_1) == 0 and len(corridor_2) > 0:
            corridor = corridor_2
            corridor.insert(0, move1.id1)
        return corridor

    def get_corridor_constraint(self, conflict: Conflict, corridor: list) -> list:
        cor_len = len(corridor)
        if cor_len <= 1:
            return []
        agent1 = conflict.agent1
        agent2 = conflict.agent2
        move1 = conflict.move1
        move2 = conflict.move2
        move1_index = corridor.index(move1.id2)
        remaining_time1 = 0
        for i in range(move1_index, len(corridor)-1):
            remaining_time1 += self.map.get_dist_id(corridor[i], corridor[i+1])
        move2_index = corridor.index(move2.id2)
        remaining_time2 = 0
        for i in range(move2_index, 0, -1):
            remaining_time2 += self.map.get_dist_id(corridor[i], corridor[i-1])
        end_time_2 = move2.t2 + remaining_time2 + 1
        end_time_1 = move1.t2 + remaining_time1 + 1

        # end_time_2 = move2.t1 + move2_index + 1
        # end_time_1 = move1.t1 + (cor_len - move1_index)
        start_id = corridor[0]
        end_id = corridor[-1]
        # if len(corridor) < 2:
        #     print(corridor)
        #     breakpoint()

        return [Constraint(agent1, 0, end_time_2, start_id, corridor[1]), Constraint(agent2, 0, end_time_1, end_id, corridor[-2])]

    def get_target_constraint(self, conflict: Conflict, task: Task) -> list:
        """

        :param conflict:
        :return: 返回针对两个agent的约束
        """
        agent1 = conflict.agent1
        goal1 = task.get_agent(agent1).goal_id
        agent2 = conflict.agent2
        goal2 = task.get_agent(agent2).goal_id

        move1 = conflict.move1
        move2 = conflict.move2

        # agent1位于终点
        target_case_1 = move1.id1 == move1.id2 and move1.id2 == goal1 and move2.id1 != move2.id2 and move2.id2 == move1.id1
        # agent2位于终点
        target_case_2 = move2.id1 == move2.id2 and move2.id2 == goal2 and move1.id1 != move1.id2 and move1.id2 == move2.id1
        if target_case_1:
            return [Constraint(agent1, 0, move2.t2, move1.id1, move1.id2), Constraint(agent2, move2.t1, float('inf'), move2.id1, move2.id2), ]
        elif target_case_2:
            return [Constraint(agent1, move1.t1, float('inf'), move1.id1, move1.id2), Constraint(agent2, 0, move1.t2, move2.id1, move2.id2)]
        return []

    def get_wait_constraint(self, agent: int, move1: Move, move2: Move) -> Constraint:
        radius = 2 * self.config.agent_size
        x0, y0 = self.map.nodes[move2.id1].x, self.map.nodes[move2.id1].y
        x1, y1 = self.map.nodes[move2.id2].x, self.map.nodes[move2.id2].y  # move2 is move
        x2, y2 = self.map.nodes[move1.id1].x, self.map.nodes[move1.id1].y  # move1 is wait

        interval = None
        point = Point(x2, y2)
        p0, p1 = Point(x0, y0), Point(x1, y1)

        # Classify the position of the first agent relative to the line formed by the motion of the second agent
        cls = point.classify(p0, p1)
        # Calculate the perpendicular distance between the initial position of the first agent
        # and the line formed by the motion of the second agent
        # ToDo: division by zero?
        dist = abs((x0 - x1) * y2 + (y1 - y0) * x2 + (y0 * x1 - x0 * y1)) / (
                math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2) + self.CN_EPSILON)
        # Calculate squared distances from the initial positions of the first agent to the endpoints of the second agent's motion
        da = (x0 - x2) ** 2 + (y0 - y2) ** 2
        db = (x1 - x2) ** 2 + (y1 - y2) ** 2
        # Calculate the height of the triangle formed by the initial position of the first agent
        # and the line formed by the motion of the second agent
        ha = math.sqrt(max(da - dist ** 2, 0))
        # Calculate the size of the safety zone (a circle centered on the line of motion of the second agent)
        size = math.sqrt(radius ** 2 - dist ** 2)

        # Check for No Conflict
        if radius ** 2 - dist ** 2 < 0:
            print(f"No conflict at t={move1.t1} diff = {dist - radius}")
            return Constraint(agent, None, None, move1.id1, move1.id2)

        # Generate Wait Constraint
        if cls == 3:
            interval = (move2.t1, move2.t1 + (size - ha))
        elif cls == 4:
            interval = (move2.t2 - size + math.sqrt(db - dist ** 2), move2.t2)
        elif da < radius ** 2:
            if db < radius ** 2:
                interval = (move2.t1, move2.t2)
            else:
                hb = math.sqrt(db - dist ** 2)
                interval = (move2.t1, move2.t2 - hb + size)
        else:
            if db < radius ** 2:
                interval = (move2.t1 + ha - size, move2.t2)
            else:
                interval = (move2.t1 + ha - size, move2.t1 + ha + size)

        return Constraint(agent, interval[0], interval[1], move1.id1, move1.id2)

    def get_hl_heuristic(self, conflicts) -> float:
        if not conflicts or self.config.hlh_type == 0:
            return 0

        elif self.config.hlh_type == 1:  # solve lpp by simplex
            '''
            simplex = optimization.Simplex("simplex")
            colliding_agents = {}
            for c in conflicts:
                colliding_agents[c.agent1] = len(colliding_agents)
                colliding_agents[c.agent2] = len(colliding_agents)

            coefficients = pilal.Matrix(len(conflicts), len(colliding_agents), 0)
            overcosts = [0] * len(conflicts)
            i = 0
            for c in conflicts:
                coefficients.at(i, colliding_agents[c.agent1]) = 1
                coefficients.at(i, colliding_agents[c.agent2]) = 1
                overcosts[i] = c.overcost
                i += 1

            simplex.set_problem(coefficients, overcosts)
            simplex.solve()
            return simplex.get_solution()
            '''
            # ToDo
            return 0  # Not supported
        else:  # 2: greedly take disjoint conflicts
            h_value = 0
            values = [(c.overcost, c.agent1, c.agent2) for c in conflicts]
            values.sort(reverse=True, key=lambda x: x[0])
            used = set()
            for v in values:
                if v[1] in used or v[2] in used:
                    continue
                h_value += v[0]
                used.add(v[1])
                used.add(v[2])
            return h_value

    def get_constraint(self, agent: int, move1: Move, move2: Move) -> Constraint:
        if move1.id1 == move1.id2:
            return self.get_wait_constraint(agent, move1, move2)

        move1 = Move.fromMove(move1)
        move2 = Move.fromMove(move2)

        startTimeA, endTimeA = move1.t1, move1.t2
        # A = Vector2D(self.map.nodes[move1.id1].x, self.map.nodes[move1.id1].y)
        # A2 = Vector2D(self.map.nodes[move1.id2].x, self.map.nodes[move1.id2].y)
        # B = Vector2D(self.map.nodes[move2.id1].x, self.map.nodes[move2.id1].y)
        # B2 = Vector2D(self.map.nodes[move2.id2].x, self.map.nodes[move2.id2].y)

        if math.isinf(move2.t2):
            return Constraint(agent, move1.t1, float('inf'), move1.id1, move1.id2)

        delta = move2.t2 - move1.t1

        # ToDo: calculate the delay directly based on move1 and move2
        while delta > self.config.precision / 2.0:
            if self.check_conflict(move1, move2):
                move1.t1 += delta
                move1.t2 += delta
            else:
                move1.t1 -= delta
                move1.t2 -= delta

            if move1.t1 > move2.t2 + self.CN_EPSILON:
                move1.t1 = move2.t2
                move1.t2 = move1.t1 + endTimeA - startTimeA
                break

            delta /= 2.0

        if delta < self.config.precision / 2.0 + self.CN_EPSILON and self.check_conflict(move1, move2):
            move1.t1 = min(move1.t1 + delta * 2, move2.t2)
            move1.t2 = move1.t1 + endTimeA - startTimeA

        return Constraint(agent, startTimeA, move1.t1, move1.id1, move1.id2)

    def get_conflict(self, conflicts) -> Conflict:
        if len(conflicts) == 0:
            return None

        best_it = conflicts[0]
        for it in conflicts:
            if it.overcost > 0:
                if best_it.overcost < it.overcost or (
                        abs(best_it.overcost - it.overcost) < self.CN_EPSILON and best_it.t < it.t):
                    best_it = it
            elif best_it.t < it.t:
                best_it = it

        conflicts.remove(best_it)
        return best_it

    # 检查给定的约束在添加之前是否与已经存在的正约束冲突
    def check_positive_constraints(self, constraints, constraint: Constraint) -> bool:
        positives = [c for c in constraints if c.positive and c.agent == constraint.agent]

        for p in positives:
            if (
                    p.id1 == constraint.id1 and p.id2 == constraint.id2 and p.t1 - self.CN_EPSILON < constraint.t1 and p.t2 + self.CN_EPSILON > constraint.t2) or \
                    (
                            p.id1 == constraint.id1 and p.id2 == constraint.id2 and constraint.t1 - self.CN_EPSILON < p.t1 and constraint.t2 + self.CN_EPSILON > p.t2):
                return False

        return True

    def validate_constraints(self, constraints, agent_id: int) -> bool:
        positives = [c for c in constraints if c.positive and c.agent == agent_id]

        for p in positives:
            for c in constraints:
                if c.positive:
                    continue

                # Both positive and negative constraints for the same agent on the same interval
                if p.agent == c.agent and p.id1 == c.id1 and p.id2 == c.id2:
                    if p.t1 > c.t1 - self.CN_EPSILON and p.t2 < c.t2 + self.CN_EPSILON:
                        return False
        return True

    # Get all constraints for current node and its parents
    def get_constraints(self, node: CBS_Node, agent_id: int):
        cur_node = node
        constraints = []
        while cur_node.parent is not None:
            if agent_id < 0 or cur_node.constraint.agent == agent_id:
                constraints.append(cur_node.constraint)
            if cur_node.positive_constraint.agent == agent_id:
                constraints.append(cur_node.positive_constraint)
            cur_node = cur_node.parent
        return constraints

    # Check paths A and B
    def check_paths(self, pathA: sPath, pathB: sPath) -> Conflict:
        a, b = 0, 0
        nodesA, nodesB = pathA.nodes, pathB.nodes

        # print(f"Checking paths...\nA: {pathA}\nB: {pathB}")

        while a < len(nodesA) - 1 or b < len(nodesB) - 1:
            dist = self.map.get_dist_id(nodesA[a].id, nodesB[b].id)

            # Common nodes
            if a < len(nodesA) - 1 and b < len(nodesB) - 1:
                dist = min(dist, self.map.get_dist_id(nodesA[a + 1].id, nodesB[b + 1].id))

                if dist < (nodesA[a + 1].g - nodesA[a].g) + (nodesB[b + 1].g - nodesB[b].g):
                    if self.check_conflict(Move.fromNodes(nodesA[a], nodesA[a + 1]),
                                           Move.fromNodes(nodesB[b], nodesB[b + 1])):
                        # print(f"check_path - conflict at {a}-{a+1} {b}-{b+1}   {nodesA[a]}-{nodesA[a+1]} {nodesB[b]}-{nodesB[b+1]}")
                        return Conflict(pathA.agentID, pathB.agentID, Move.fromNodes(nodesA[a], nodesA[a + 1]),
                                        Move.fromNodes(nodesB[b], nodesB[b + 1]), min(nodesA[a].g, nodesB[b].g))

            # no more path A nodes
            elif a == len(nodesA) - 1:
                if dist < (nodesB[b + 1].g - nodesB[b].g):
                    if self.check_conflict(Move(nodesA[a].g, self.CN_INFINITY, nodesA[a].id, nodesA[a].id),
                                           Move.fromNodes(nodesB[b], nodesB[b + 1])):
                        # print(f"check_path - conflict at {a}-end  {b}-{b+1}")
                        return Conflict(pathA.agentID, pathB.agentID,
                                        Move(nodesA[a].g, self.CN_INFINITY, nodesA[a].id, nodesA[a].id),
                                        Move.fromNodes(nodesB[b], nodesB[b + 1]), min(nodesA[a].g, nodesB[b].g))

            # no more path B nodes
            elif b == len(nodesB) - 1:
                if dist < (nodesA[a + 1].g - nodesA[a].g):
                    if self.check_conflict(Move.fromNodes(nodesA[a], nodesA[a + 1]),
                                           Move(nodesB[b].g, self.CN_INFINITY, nodesB[b].id, nodesB[b].id)):
                        # print(f"check_path - conflict at {a}-{a+1} {b}-end")
                        return Conflict(pathA.agentID, pathB.agentID, Move.fromNodes(nodesA[a], nodesA[a + 1]),
                                        Move(nodesB[b].g, self.CN_INFINITY, nodesB[b].id, nodesB[b].id),
                                        min(nodesA[a].g, nodesB[b].g))

            if a == len(nodesA) - 1:
                b += 1
            elif b == len(nodesB) - 1:
                a += 1
            elif abs(nodesA[a + 1].g - nodesB[b + 1].g) < self.CN_EPSILON:
                a += 1
                b += 1
            elif nodesA[a + 1].g < nodesB[b + 1].g:
                a += 1
            elif nodesB[b + 1].g - self.CN_EPSILON < nodesA[a + 1].g:
                b += 1
        return Conflict()

    def cal_total_wait_time(self, paths):
        total_wait_time = 0
        for i in range(len(paths)):
            nodes= paths[i].nodes
            dist = 0
            for j in range(len(nodes)-1):
                dist += self.map.get_dist_id(nodes[j].id, nodes[j+1].id)
            total_wait_time += (paths[i].cost - dist)
            print(paths[i].cost - dist)
        return total_wait_time

    def cal_conflict_agent(self, paths):
        agent_set = set()
        conflicts = self.get_all_conflicts(paths, -1)
        for c in conflicts:
            agent_set.add(c.agent1)
            agent_set.add(c.agent2)
        return len(agent_set)

    def cal_conflict_time_std(self, paths):
        happen_time = []
        conflicts = self.get_all_conflicts(paths, -1)
        if len(conflicts) == 0:
            return 0
        for c in conflicts:
            happen_time.append(c.t)
        return statistics.variance(happen_time)

    def get_earlierst_conflict(self, paths):
        conflicts = self.get_all_conflicts(paths, -1)
        if len(conflicts) == 0:
            return -1
        EH = np.Inf
        for c in conflicts:
            if c.t < EH:
                EH = c.t
        return EH

    def get_all_conflicts(self, paths, agent_id: int):
        # print("Checking all paths for conflicts...")
        conflicts = []
        if agent_id < 0:
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    conflict = self.check_paths(paths[i], paths[j])
                    if conflict.agent1 >= 0:
                        # print(f"Conflict found: {conflict}")
                        conflicts.append(conflict)
        else:
            for i in range(len(paths)):
                if i == agent_id:
                    continue
                conflict = self.check_paths(paths[i], paths[agent_id])
                if conflict.agent1 >= 0:
                    # print(f"Conflict found: {conflict}")
                    conflicts.append(conflict)
        return conflicts

    def get_cost(self, node: CBS_Node, agent_id: int):
        while node.parent is not None:
            if node.paths[0].agentID == agent_id:
                return node.paths[0].cost
            node = node.parent
        return node.paths[agent_id].cost

    def get_paths(self, node: CBS_Node, agents_size: int):
        cur_node = node
        paths = [sPath() for _ in range(agents_size)]

        while cur_node.parent is not None:
            if paths[cur_node.paths[0].agentID].cost < 0:
                paths[cur_node.paths[0].agentID] = cur_node.paths[0]
            cur_node = cur_node.parent
        for i in range(agents_size):
            if paths[i].cost < 0:
                paths[i] = cur_node.paths[i]
        return paths

    def get_spath_cost(self, paths):
        cost_list = []
        for p in paths:
            cost_list.append(p.cost)
        return sum(cost_list)

    def find_new_conflicts(self, task: Task, node: CBS_Node, paths, path: sPath, conflicts, semicard_conflicts,
                           cardinal_conflicts, low_level_searches: int, low_level_expanded: int):
        old_path = paths[path.agentID]
        paths[path.agentID] = path
        new_conflicts = self.get_all_conflicts(paths, path.agentID)
        paths[path.agentID] = old_path

        conflictsA, semicard_conflictsA, cardinal_conflictsA = [], [], []

        # Filter / sort the conflicts 
        for c in conflicts:
            if c.agent1 != path.agentID and c.agent2 != path.agentID:
                conflictsA.append(c)

        for c in semicard_conflicts:
            if c.agent1 != path.agentID and c.agent2 != path.agentID:
                semicard_conflictsA.append(c)

        for c in cardinal_conflicts:
            if c.agent1 != path.agentID and c.agent2 != path.agentID:
                cardinal_conflictsA.append(c)

        # If not using cardinal / non cardinal differentiation
        if not self.config.use_cardinal:
            node.conflicts = conflictsA + new_conflicts
            node.cardinal_conflicts.clear()
            node.semicard_conflicts.clear()
            node.conflicts_num = len(node.conflicts)
            return [low_level_searches, low_level_expanded]

        for c in new_conflicts:
            constraintsA, constraintsB = [], []

            # if both agents are waiting....
            if c.agent1 == path.agentID:
                constraintsA = self.get_constraints(node, c.agent1)
                constraintsA.append(self.get_constraint(c.agent1, c.move1, c.move2))
                new_pathA = self.planner.find_path(task.get_agent(c.agent1), constraintsA)

                constraintsB = self.get_constraints(node, c.agent2)
                constraintsB.append(self.get_constraint(c.agent2, c.move2, c.move1))
                new_pathB = self.planner.find_path(task.get_agent(c.agent2), constraintsB)

                old_cost = self.get_cost(node, c.agent2)

                if new_pathA.cost < 0 and new_pathB.cost < 0:
                    node.cost = -1
                    return [low_level_searches, low_level_expanded]
                elif new_pathA.cost < 0:
                    c.overcost = new_pathB.cost - old_cost
                    cardinal_conflictsA.append(c)
                    # print("new_conflict =a a< 0 " + str(c.overcost))
                elif new_pathB.cost < 0:
                    c.overcost = new_pathA.cost - path.cost
                    cardinal_conflictsA.append(c)
                    # print("new_conflict =a b< 0 " + str(c.overcost))
                elif new_pathA.cost > path.cost and new_pathB.cost > old_cost:
                    c.overcost = min(new_pathA.cost - path.cost, new_pathB.cost - old_cost)
                    cardinal_conflictsA.append(c)
                    # print("new_conflict =a =? " + str(c.overcost))
                elif new_pathA.cost > path.cost or new_pathB.cost > old_cost:
                    semicard_conflictsA.append(c)
                else:
                    conflictsA.append(c)

                low_level_searches += 2
                low_level_expanded += (new_pathA.expanded + new_pathB.expanded)

            else:
                constraintsA = self.get_constraints(node, c.agent2)
                constraintsA.append(self.get_constraint(c.agent2, c.move2, c.move1))
                new_pathA = self.planner.find_path(task.get_agent(c.agent2), constraintsA)

                constraintsB = self.get_constraints(node, c.agent1)
                constraintsB.append(self.get_constraint(c.agent1, c.move1, c.move2))
                new_pathB = self.planner.find_path(task.get_agent(c.agent1), constraintsB)

                old_cost = self.get_cost(node, c.agent1)

                if new_pathA.cost < 0 and new_pathB.cost < 0:
                    node.cost = -1
                    return [low_level_searches, low_level_expanded]
                elif new_pathA.cost < 0:
                    c.overcost = new_pathB.cost - old_cost
                    cardinal_conflictsA.append(c)
                    # print("new_conflict !a a< 0 " + str(c.overcost))
                elif new_pathB.cost < 0:
                    c.overcost = new_pathA.cost - path.cost
                    cardinal_conflictsA.append(c)
                    # print("new_conflict !a b< 0 " + str(c.overcost))
                elif new_pathA.cost > path.cost and new_pathB.cost > old_cost:
                    c.overcost = min(new_pathA.cost - path.cost, new_pathB.cost - old_cost)
                    cardinal_conflictsA.append(c)
                    # print("new_conflict !a =? " + str(c.overcost))
                elif new_pathA.cost > path.cost or new_pathB.cost > old_cost:
                    semicard_conflictsA.append(c)
                else:
                    conflictsA.append(c)

                low_level_searches += 2
                low_level_expanded += (new_pathA.expanded + new_pathB.expanded)

        node.conflicts = conflictsA
        node.semicard_conflicts = semicard_conflictsA
        node.cardinal_conflicts = cardinal_conflictsA
        node.conflicts_num = len(conflictsA) + len(semicard_conflictsA) + len(cardinal_conflictsA)
        return [low_level_searches, low_level_expanded]

    def find_solution(self, task: Task) -> Solution:
        if self.config.use_precalculated_heuristic:
            # Initialize the heuristic based on reverse Dijkstra
            heuristic_start_time = time.time()
            self.map.init_heuristic(task.agents)
            print(f"heuristic process time {time.time() - heuristic_start_time}s")

        if self.verbose:
            print("CCBS find solution...")
        self.solution = Solution()
        start_time = time.time()

        cardinal_solved = 0
        semicardinal_solved = 0

        # 初始化根节点
        if not self.init_root(task):
            if self.verbose:
                print("No root solution possible, cannot continue")
            return self.solution

        self.solution.init_time = time.time() - start_time
        self.solution.found = True
        node = CBS_Node()

        expanded = 1
        time_elapsed = 0
        low_level_searches = 0
        low_level_expanded = 0
        agent_id = 2

        # 使用强化学习进行CBS分支
        if self.config.use_rl:
            from ccbsenv import CCBSEnv  # 延迟导入以避免循环依赖
            from stable_baselines3 import PPO

            if len(self.tree.container) == 0:
                print('No Solution')
                return self.solution
            else:
                parent = self.tree.get_front()  # Get frontal node from the tree
                if parent.conflicts_num == 0:
                    # Save solution results
                    self.solution.paths = self.get_paths(parent, len(task.agents))
                    self.solution.flowtime = parent.cost
                    self.solution.low_level_expansions = low_level_searches
                    self.solution.low_level_expanded = low_level_expanded / max(low_level_searches, 1)
                    self.solution.high_level_expanded = expanded
                    self.solution.high_level_generated = self.tree.get_size()

                    for path in self.solution.paths:
                        self.solution.makespan = max(self.solution.makespan, path.cost)

                    self.solution.time = time.time() - start_time
                    self.solution.check_time = time_elapsed
                    self.solution.cardinal_solved = cardinal_solved
                    self.solution.semicardinal_solved = semicardinal_solved

                    self.solution.task = task
                    return self.solution

                # 记录程序开始时间
                start_time = time.time()
                env = CCBSEnv(task, parent, self.map, expanded, time_elapsed, low_level_searches, low_level_expanded, self.tree)
                model_path = self.config.rl_model_path
                candidates = [model_path]
                base, ext = os.path.splitext(model_path)
                if ext == "":
                    candidates.append(f"{model_path}.zip")
                else:
                    candidates.append(base)

                resolved_model_path = None
                for candidate in candidates:
                    if candidate and os.path.exists(candidate):
                        resolved_model_path = candidate
                        break

                if resolved_model_path is None:
                    raise FileNotFoundError(f"RL model not found. Tried: {', '.join(candidate for candidate in candidates if candidate)}")

                model = PPO.load(resolved_model_path)
                start_solve_time = time.time()

                total_rewards = []
                state, _ = env.reset()
                done = False

                while not done:
                    action, _ = model.predict(state, deterministic=False)
                    state, reward, done, truncated, _ = env.step(action)
                    total_rewards.append(reward)
                    print(f"Action: {action}, Reward: {reward}, Done: {done}")

                print(f"optimal solution: {state}")
                print(f"optimal obj: {env.node.cost}")
                print(f"average reward: {np.mean(total_rewards)}")

                end_time = time.time()
                # 计算运行时间
                solve_time = end_time - start_solve_time
                print(f"求解时间: {solve_time:.4f} s")
                runtime = end_time - start_time
                print(f"程序运行时间: {runtime:.4f} s")

                if not env.final_res:
                    print('No Solution')
                    return self.solution

                # Save solution results
                self.solution.paths = self.get_paths(env.final_res, len(task.agents))
                self.solution.flowtime = env.final_res.cost
                self.solution.low_level_expansions = env.low_level_searches
                self.solution.low_level_expanded = env.low_level_expanded / max(env.low_level_searches, 1)
                self.solution.high_level_expanded = env.expanded
                self.solution.high_level_generated = env.high_level_generated

                for path in self.solution.paths:
                    self.solution.makespan = max(self.solution.makespan, path.cost)

                self.solution.time = solve_time
                self.solution.check_time = env.time_elapsed

                self.solution.task = task
                return self.solution

        # if not use rl
        while True:
            if len(self.tree.container) == 0:
                print('No Solution')
                break
            parent = self.tree.get_front()  # Get frontal node from the tree
            node = parent.create_node_move_conflicts()  # Create new node based on parent, move conflicts to new node
            node.cost -= node.h  # remove heuristics

            # Combine all existing paths in the tree from node to parent
            paths = self.get_paths(node, len(task.agents))

            time_now = time.time()

            # node.conflicts中为路径之间出现的第一次冲突
            if not node.conflicts and not node.semicard_conflicts and not node.cardinal_conflicts:
                # Done with search, no more conflicts
                if self.verbose:
                    print("No conflicts, solution found successfully")
                break  # No conflicts => solution found

            # Select the new conflict to be solved - prioritize cardinal over semi-cardinal over regular
            if node.cardinal_conflicts:
                conflict = self.get_conflict(node.cardinal_conflicts)
                cardinal_solved += 1
            elif node.semicard_conflicts:
                conflict = self.get_conflict(node.semicard_conflicts)
                semicardinal_solved += 1
            else:
                conflict = self.get_conflict(node.conflicts)

            time_spent = time.time() - time_now
            time_elapsed += time_spent
            expanded += 1

            # 使用corridor_symmetry策略
            corridor_constraints = []
            if self.config.use_corridor_symmetry:
                corridor = self.check_corridor_conflict(conflict, task)
                corridor_constraints = self.get_corridor_constraint(conflict, corridor)

            # 使用target_symmetry策略
            target_constraints = []
            if self.config.use_target_symmetry:
                target_constraints = self.get_target_constraint(conflict, task)

            # Join existing constraints, add new constraint from the selected conflict, then plan path - all for agent 1
            constraintsA = self.get_constraints(node, conflict.agent1)
            if len(target_constraints) > 0:
                constraintA = target_constraints[0]
                # print("target symmetry happens.")
                # print(constraintA)
            elif len(corridor_constraints) > 0:
                constraintA = corridor_constraints[0]
                # print("corridor symmetry happens.")
                # print(constraintA)
            else:
                constraintA = self.get_constraint(conflict.agent1, conflict.move1, conflict.move2)
            constraintsA.append(constraintA)
            pathA = self.planner.find_path(task.get_agent(conflict.agent1), constraintsA)
            low_level_searches += 1
            low_level_expanded += pathA.expanded

            # Do the same for the other agent
            constraintsB = self.get_constraints(node, conflict.agent2)
            if len(target_constraints) > 0:
                constraintB = target_constraints[1]
                # print("target symmetry happens.")
                # print(constraintB)
            elif len(corridor_constraints) > 0:
                constraintB = corridor_constraints[1]
                # print("corridor symmetry happens.")
                # print(constraintB)
            else:
                constraintB = self.get_constraint(conflict.agent2, conflict.move2, conflict.move1)
            constraintsB.append(constraintB)
            pathB = self.planner.find_path(task.get_agent(conflict.agent2), constraintsB)
            low_level_searches += 1
            low_level_expanded += pathB.expanded

            # Print the stats/current step            
            if self.verbose:
                confstr = f"{conflict.move1.id1}@{conflict.move1.t1} -> {conflict.move1.id2}@{conflict.move1.t2} and {conflict.move2.id1}@{conflict.move2.t1} -> {conflict.move2.id2}@{conflict.move2.t2}"
                print(
                    f"Tree node {node.id}/{node.id_str}: Conflict {conflict.agent1} and {conflict.agent2} - total constraints: {len(constraintsA)} / {len(constraintsB)}     - conflict: {confstr}")

            # Construct leaf nodes for the two solutions
            right = CBS_Node([pathA], parent, constraintA,
                             node.cost + pathA.cost - self.get_cost(node, conflict.agent1), 0, node.total_cons + 1)
            left = CBS_Node([pathB], parent, constraintB,
                            node.cost + pathB.cost - self.get_cost(node, conflict.agent2), 0, node.total_cons + 1)

            positive = None
            inserted = False
            left_ok = True  # ToDo: Why is this needed?
            right_ok = True  # ToDo: Why is this needed?

            # In case of disjoint splitting, add positive constraint to one of the agent's paths
            if self.config.use_disjoint_splitting:
                agent1_positives = sum(1 for c in constraintsA if c.positive)
                agent2_positives = sum(1 for c in constraintsB if c.positive)

                # Positive constraint for agent 1
                if conflict.move1.id1 != conflict.move1.id2 and agent2_positives > agent1_positives and pathA.cost > 0:
                    positive = Constraint(conflict.agent1, constraintA.t1, constraintA.t2, conflict.move1.id1,
                                          conflict.move1.id2, True)
                    if self.check_positive_constraints(constraintsA, positive):
                        left.positive_constraint = positive
                        left.total_cons += 1
                        constraintsB.append(left.positive_constraint)
                        inserted = True

                # Positive constraint for agent 2
                if conflict.move2.id1 != conflict.move2.id2 and not inserted and pathB.cost > 0:
                    positive = Constraint(conflict.agent2, constraintB.t1, constraintB.t2, conflict.move2.id1,
                                          conflict.move2.id2, True)
                    if self.check_positive_constraints(constraintsB, positive):
                        right.positive_constraint = positive
                        right.total_cons += 1
                        constraintsA.append(right.positive_constraint)
                        inserted = True

                # Positive constraint for agent 1
                if conflict.move1.id1 != conflict.move1.id2 and not inserted and pathA.cost > 0:
                    positive = Constraint(conflict.agent1, constraintA.t1, constraintA.t2, conflict.move1.id1,
                                          conflict.move1.id2, True)
                    if self.check_positive_constraints(constraintsA, positive):
                        left.positive_constraint = positive
                        left.total_cons += 1
                        constraintsB.append(left.positive_constraint)
                        inserted = True

            # Mark the tree nodes
            right.id_str = node.id_str + "0"
            left.id_str = node.id_str + "1"
            right.id = agent_id
            agent_id += 1
            left.id = agent_id
            agent_id += 1

            # If pathA exists and satisfies to the given constraints, add it to the tree
            if right_ok and pathA.cost > 0 and self.validate_constraints(constraintsA, pathA.agentID):
                time_now = time.time()
                low_level_searches, low_level_expanded = self.find_new_conflicts(task, right, paths, pathA,
                                                                                 node.conflicts,
                                                                                 node.semicard_conflicts,
                                                                                 node.cardinal_conflicts,
                                                                                 low_level_searches, low_level_expanded)
                time_spent = time.time() - time_now
                time_elapsed += time_spent

                if right.cost > 0:
                    right.h = self.get_hl_heuristic(right.cardinal_conflicts)
                    right.cost += right.h
                    self.tree.add_node(right)

            # If pathB exists and satisfies to the given constraints, add it to the tree
            if left_ok and pathB.cost > 0 and self.validate_constraints(constraintsB, pathB.agentID):
                time_now = time.time()
                low_level_searches, low_level_expanded = self.find_new_conflicts(task, left, paths, pathB,
                                                                                 node.conflicts,
                                                                                 node.semicard_conflicts,
                                                                                 node.cardinal_conflicts,
                                                                                 low_level_searches, low_level_expanded)
                time_spent = time.time() - time_now
                time_elapsed += time_spent

                if left.cost > 0:
                    left.h = self.get_hl_heuristic(left.cardinal_conflicts)
                    left.cost += left.h
                    self.tree.add_node(left)

            # Timeout handling
            time_spent = time.time() - start_time
            if time_spent > self.config.timelimit:
                print("Time limit reached, no solution found")
                self.solution.found = False
                break

        # Save solution results
        self.solution.paths = self.get_paths(node, len(task.agents))
        self.solution.flowtime = node.cost
        self.solution.low_level_expansions = low_level_searches
        self.solution.low_level_expanded = low_level_expanded / max(low_level_searches, 1)
        self.solution.high_level_expanded = expanded
        self.solution.high_level_generated = self.tree.get_size()

        for path in self.solution.paths:
            self.solution.makespan = max(self.solution.makespan, path.cost)

        self.solution.time = time.time() - start_time
        self.solution.check_time = time_elapsed
        self.solution.cardinal_solved = cardinal_solved
        self.solution.semicardinal_solved = semicardinal_solved

        self.solution.task = task

        return self.solution

    def write_to_log_path(self, file):
        root = ET.Element('root')

        # List all agents
        for ag in self.solution.task.agents:
            ags = ET.SubElement(root, 'agent')
            ags.set('start_id', str(ag.start_id))
            ags.set('goal_id', str(ag.goal_id))

        log = ET.SubElement(root, 'log')
        summary = ET.SubElement(log, 'summary')
        summary.set('time', str(self.solution.time))
        summary.set('flowtime', str(self.solution.flowtime))
        summary.set('makespan', str(self.solution.makespan))

        for i, path in enumerate(solution.paths):
            agent = ET.SubElement(log, 'agent')
            agent.set("number", str(i))

            path_elem = ET.SubElement(agent, 'path')
            path_elem.set('duration', str(path.cost))

            for i in range(len(path.nodes) - 1):
                n1, n2 = path.nodes[i], path.nodes[i + 1]
                part = ET.SubElement(path_elem, 'section')
                part.set('number', str(i))
                part.set('start_i', str(self.map.nodes[n1.id].x))
                part.set('start_j', str(self.map.nodes[n1.id].y))
                part.set('start_id', str(n1.id))
                part.set('goal_i', str(self.map.nodes[n2.id].x))
                part.set('goal_j', str(self.map.nodes[n2.id].y))
                part.set('goal_id', str(n2.id))
                part.set('duration', str(n2.g - n1.g))

        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ", level=0)
        tree.write(file, short_empty_elements=False)

    def evaluate_model(self, model, env, num_episodes=100):
        total_rewards = []

        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                action, _ = model.predict(state)
                state, reward, done, truncated, _ = env.step(action)
                total_reward += reward

            total_rewards.append(total_reward)
        print(f"optimal solution: {state}")
        print(f"optimal obj: {env.node.cost}")
        print(f"Average Reward over {num_episodes} episodes: {np.mean(total_rewards)}")
        return env.node


# Test the CCBS
if __name__ == "__main__":
    # Load the map
    print("Loading map...")
    map = Map("instances/room-64-64-8_random/map.xml")

    ccbs = CCBS(map)
    test_file = [
        os.path.join("instances", "room-64-64-8_random", "test", "30", fname)
        for fname in os.listdir(os.path.join("instances", "room-64-64-8_random", "test", "30"))
        if fname != ".DS_Store"
    ]
    result = {}

    for f in test_file:
        path_len = 0
        task = Task()
        task.load_from_file(f)
        print(task)

        solution = ccbs.find_solution(task)
        result[f] = solution
        print(solution)
        print(f"==========Paths result==========")
        for i in range(len(solution.paths)):
            print(f"Agent{i}: {solution.paths[i]}")
            path_len += solution.paths[i].cost
        # breakpoint()
        print(f"total_path_len: {path_len}")
        # animation = GraphRender(map, task, solution.paths)
        # animation.show()
    print(result)

    # task_list = []
    # for f in test_file:
    #     task = Task()
    #     task.load_from_file(f)
    #     task_list.append(task)
    # with mp.Pool(processes=10) as pool:
    #     results = []
    #     for i, solution in enumerate(pool.imap_unordered(ccbs.find_solution, task_list)):
    #         results.append(solution)
    # print(results)

    # Create task
    # task = Task()
    #
    # taskSet = 0
    # if taskSet is None:
    #     import random
    #
    #     # Specify the station nodes
    #     stations = range(1, 217, 2)
    #     Ntasks = 20
    #
    #     Na = 0
    #     while len(task.agents) < Ntasks:
    #         starts = [a.start_id for a in task.agents]
    #         goals = [a.goal_id for a in task.agents]
    #
    #         # Generate new random task
    #         sId = stations[int(random.random() * len(stations))]
    #         gId = stations[int(random.random() * len(stations))]
    #
    #         if sId in starts or gId in goals:
    #             continue
    #
    #         task.agents.append(Agent(sId, gId, Na))
    #         Na += 1
    #
    # elif taskSet == 0:
    #     task.load_from_file("./instances/empty-16-16-random/test/10/converted_empty-16-16-random-18-10.xml")
    #
    # elif taskSet == 1:
    #     task.agents.append(Agent(85, 35, 0))
    #     task.agents.append(Agent(161, 113, 1))
    #     task.agents.append(Agent(105, 19, 2))
    #     task.agents.append(Agent(10, 117, 3))
    #
    # elif taskSet == 2:
    #     # tasks = "0: 34->80, 1: 184->122, 2: 114->174, 3: 12->126, 4: 130->194, 5: 210->212, 6: 200->88, 7: 78->110, 8: 168->112, 9: 150->118"
    #     # tasks = "0: 156->94, 1: 16->172, 2: 66->20, 3: 64->62, 4: 150->208, 5: 8->2, 6: 14->196, 7: 178->10, 8: 32->150, 9: 106->26"
    #     tasks = "0: 14->174, 1: 136->180, 2: 32->176, 3: 156->86, 4: 86->30, 5: 158->142, 6: 96->62, 7: 214->74, 8: 56->114, 9: 34->44, 10: 148->212, 11: 76->156, 12: 150->26, 13: 84->162, 14: 208->36, 15: 60->206, 16: 88->78, 17: 174->96, 18: 114->152, 19: 82->130"
    #     for t in tasks.split(","):
    #         id_task = t.split(": ")
    #         nodes = id_task[1].split("->")
    #         task.agents.append(Agent(int(nodes[0]), int(nodes[1]), int(id_task[0])))
    #
    # print(task)
    #
    # solution = ccbs.find_solution(task)
    # print(solution)
    # print(f"==========Paths result==========")
    # for i in range(len(solution.paths)):
    #     print(f"Agent{i}: {solution.paths[i]}")
    #     # time_positions = [(n.id, n.g) for n in solution.paths[i].nodes]
    #     # print(time_positions)
    #
    # animation = GraphRender(map, task, solution.paths)
    # animation.show()

    # ccbs.write_to_log_path('test_out.xml')
3