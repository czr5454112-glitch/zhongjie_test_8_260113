# -*- coding: utf-8 -*-
# @author: Jay-Bling
# @email: gzj22@mails.tsinghua.edu.cn
# @date: 2024/9/10
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
import numpy as np


class GraphRender:
    def __init__(self, Map, task, paths):
        self.graph = Map.graph
        self.nodes = Map.nodes
        self.task = task
        self.paths = paths
        self.starts = []
        self.goals = []
        self.agents = dict()
        self.agent_names = dict()
        self.patches = []
        self.artists = []
        self.T = 0
        self.colors = ['orange', 'green', 'red', 'purple']
        self.fig = plt.figure(frameon=False)
        self.ax = self.fig.add_subplot(111, aspect='equal')
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)

        for edge in self.graph.edges(data=True):
            x1, y1 = map(float, self.graph.nodes[edge[0]]['coords'].split(','))
            x2, y2 = map(float, self.graph.nodes[edge[1]]['coords'].split(','))
            plt.plot([x1, x2], [y1, y2], 'white')

        plt.gca().set_facecolor('black')

        for t in self.task.agents:
            self.starts.append((self.nodes[t.start_id].x, self.nodes[t.start_id].y))
            self.goals.append((self.nodes[t.goal_id].x, self.nodes[t.goal_id].y))

        for i, goal in enumerate(self.goals):
            self.patches.append(
                Rectangle((goal[0] - 0.25, goal[1] - 0.25), 0.5, 0.5,
                          facecolor=self.colors[i % len(self.colors)],
                          edgecolor='black', alpha=0.5))

        for i, start in enumerate(self.starts):
            self.patches.append(
                Rectangle((start[0] - 0.25, start[1] - 0.25), 0.5, 0.5,
                          facecolor=self.colors[i % len(self.colors)],
                          edgecolor='black', alpha=0.5))

        for i in range(len(self.paths)):
            name = str(i + 1)
            self.agents[i] = Circle((self.starts[i][0], self.starts[i][1]), 0.25,
                                    facecolor=self.colors[i % len(self.colors)],
                                    edgecolor='black')
            self.agents[i].original_face_color = self.colors[i % len(self.colors)]
            self.patches.append(self.agents[i])
            self.T = max(self.T, self.paths[i].nodes[-1].g)
            self.agent_names[i] = self.ax.text(self.starts[i][0], self.starts[i][1], name, fontsize=7)
            self.agent_names[i].set_horizontalalignment('center')
            self.agent_names[i].set_verticalalignment('center')
            self.artists.append(self.agent_names[i])

        self.animation = animation.FuncAnimation(self.fig, self.animate_func, init_func=self.init_func,
                                                 frames=int(self.T + 1) * 10, interval=50, blit=True)

    def init_func(self):
        for p in self.patches:
            self.ax.add_patch(p)
        for a in self.artists:
            self.ax.add_artist(a)
        return self.patches + self.artists

    def animate_func(self, t):
        for k, path in enumerate(self.paths):
            pts = [(n.id, n.g) for n in path.nodes]
            pos = (0, 0)
            if t / 10 >= pts[-1][1]:
                pos = (self.nodes[pts[-1][0]].x, self.nodes[pts[-1][0]].y)
            elif t / 10 <= pts[0][1]:
                pos = (self.nodes[pts[0][0]].x, self.nodes[pts[0][0]].y)
            else:
                for i in range(len(pts) - 1):
                    p1, t1 = pts[i]
                    p2, t2 = pts[i + 1]
                    if t1 <= (t / 10) < t2:
                        ratio = (t / 10 - t1) / (t2 - t1)
                        pos = (self.nodes[p1].x + ratio * (self.nodes[p2].x - self.nodes[p1].x),
                               self.nodes[p1].y + ratio * (self.nodes[p2].y - self.nodes[p1].y))
                        break
                    else:
                        continue

            self.agents[k].center = (pos[0], pos[1])
            self.agent_names[k].set_position((pos[0], pos[1]))
            # print(f"agent_{k}:, {self.agents[k].center}")

        # reset all colors
        for _, agent in self.agents.items():
            agent.set_facecolor(agent.original_face_color)

        # check drive-drive collisions
        agents_array = [agent for _, agent in self.agents.items()]
        for i in range(len(agents_array)):
            for j in range(i + 1, len(agents_array)):
                d1 = agents_array[i]
                d2 = agents_array[j]
                pos1 = np.array(d1.center)
                pos2 = np.array(d2.center)
                if np.linalg.norm(pos1 - pos2) < 0.5:
                    d1.set_facecolor('red')
                    d2.set_facecolor('red')
                    print(f"COLLISION! (agent-agent) ({i}, {j}) at time {t / 10}")
        return self.patches + self.artists

    @staticmethod
    def show():
        plt.show()

    def save(self, file_name, speed):
        self.animation.save(
            file_name,
            fps=10 * speed,
            dpi=200,
            savefig_kwargs={"pad_inches": 0, "bbox_inches": "tight"})
