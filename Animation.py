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
        self.edge_lines = []  # 存储边的线条对象
        self.T = 0
        # 使用更鲜艳、对比度高的颜色（RGB格式，确保在黑色背景上清晰可见）
        # 这些颜色在黑色背景上有很高的对比度
        self.colors = [
            '#FF4500',  # 橙红色（非常鲜艳）
            '#00CED1',  # 深青色（高对比度）
            '#FFD700',  # 金黄色（非常亮）
            '#32CD32',  # 酸橙绿（高对比度）
            '#FF1493',  # 深粉红（非常鲜艳）
            '#00BFFF',  # 深天蓝（高对比度）
            '#FF6347',  # 番茄红（非常鲜艳）
            '#9370DB',  # 中紫色（高对比度）
            '#FF69B4',  # 热粉红（非常鲜艳）
            '#FFFF00',  # 纯黄色（非常亮）
            '#00FF7F',  # 春绿色（高对比度）
            '#1E90FF',  # 道奇蓝（高对比度）
            '#FF0000',  # 纯红色（非常鲜艳）
            '#FFA500',  # 橙色（高对比度）
            '#DA70D6',  # 兰花紫（高对比度）
            '#FF8C00',  # 深橙色（高对比度）
            '#20B2AA',  # 浅海绿（高对比度）
            '#DC143C',  # 深红色（非常鲜艳）
            '#00FA9A',  # 中春绿（高对比度）
            '#FF00FF'   # 洋红色（非常鲜艳）
        ]
        
        # 创建图形和坐标轴 - 使用更明确的设置
        plt.rcParams['figure.facecolor'] = 'black'
        plt.rcParams['axes.facecolor'] = 'black'
        self.fig = plt.figure(figsize=(14, 12), facecolor='black', edgecolor='black')
        self.ax = self.fig.add_subplot(111, aspect='equal')
        self.ax.set_facecolor('black')
        self.ax.axis('off')  # 隐藏坐标轴
        self.fig.patch.set_facecolor('black')
        
        # 收集所有坐标以设置合适的坐标轴范围
        all_x = []
        all_y = []
        
        # 绘制地图的边
        edge_count = 0
        for edge in self.graph.edges(data=True):
            try:
                node1_id = edge[0]  # 例如 "n0"
                node2_id = edge[1]  # 例如 "n1"
                
                # 从graph中获取节点坐标
                if 'coords' in self.graph.nodes[node1_id]:
                    coords1 = self.graph.nodes[node1_id]['coords'].split(',')
                    x1, y1 = float(coords1[0]), float(coords1[1])
                else:
                    # 如果graph中没有，从Map.nodes中获取
                    node1_num = int(node1_id[1:])  # 从"n0"提取0
                    x1, y1 = self.nodes[node1_num].x, self.nodes[node1_num].y
                
                if 'coords' in self.graph.nodes[node2_id]:
                    coords2 = self.graph.nodes[node2_id]['coords'].split(',')
                    x2, y2 = float(coords2[0]), float(coords2[1])
                else:
                    node2_num = int(node2_id[1:])
                    x2, y2 = self.nodes[node2_num].x, self.nodes[node2_num].y
                
                # 绘制边 - 使用更明显的白色线条
                line, = self.ax.plot([x1, x2], [y1, y2], 'w-', linewidth=2.0, alpha=0.8, zorder=1)
                self.edge_lines.append(line)
                all_x.extend([x1, x2])
                all_y.extend([y1, y2])
                edge_count += 1
            except (KeyError, ValueError, AttributeError, IndexError) as e:
                print(f"警告: 无法绘制边 {edge[0]} -> {edge[1]}: {e}")
                continue
        
        print(f"成功绘制 {edge_count} 条边")
        
        # 如果没有边，使用节点坐标
        if not all_x or not all_y:
            print("警告: 没有找到边的坐标，使用节点坐标")
            for node_id, node in self.nodes.items():
                all_x.append(node.x)
                all_y.append(node.y)
        
        # 设置坐标轴范围，留出边距
        if all_x and all_y:
            margin_x = max((max(all_x) - min(all_x)) * 0.15, 10)
            margin_y = max((max(all_y) - min(all_y)) * 0.15, 10)
            self.ax.set_xlim(min(all_x) - margin_x, max(all_x) + margin_x)
            self.ax.set_ylim(min(all_y) - margin_y, max(all_y) + margin_y)
            print(f"坐标轴范围: x=[{min(all_x) - margin_x:.1f}, {max(all_x) + margin_x:.1f}], "
                  f"y=[{min(all_y) - margin_y:.1f}, {max(all_y) + margin_y:.1f}]")
        else:
            print("错误: 无法确定坐标轴范围")
            return

        # 获取起点和终点
        for t in self.task.agents:
            try:
                start_node = self.nodes[t.start_id]
                goal_node = self.nodes[t.goal_id]
                self.starts.append((start_node.x, start_node.y))
                self.goals.append((goal_node.x, goal_node.y))
            except KeyError as e:
                print(f"警告: 无法找到节点 {e}")
                continue

        print(f"找到 {len(self.starts)} 个起点和 {len(self.goals)} 个终点")

        # 绘制终点（目标位置）- 使用更大的标记，并添加角标
        for i, goal in enumerate(self.goals):
            goal_color = self.colors[i % len(self.colors)]
            # 创建矩形，确保颜色正确设置
            rect = Rectangle((goal[0] - 0.8, goal[1] - 0.8), 1.6, 1.6,
                          facecolor=goal_color,
                          edgecolor='white', linewidth=3.0, alpha=1.0, zorder=8)
            # 强制设置颜色
            rect.set_facecolor(goal_color)
            rect.set_edgecolor('white')
            rect.set_alpha(1.0)
            self.patches.append(rect)
            # 添加"G"标签，带角标（G1, G2, ...）
            goal_label = f'G{i+1}'
            goal_text = self.ax.text(goal[0], goal[1], goal_label, fontsize=12, 
                                    color='black', weight='bold', 
                                    ha='center', va='center', zorder=9,
                                    bbox=dict(boxstyle='round,pad=0.3', 
                                             facecolor='white', 
                                             edgecolor='black', 
                                             linewidth=2.0,
                                             alpha=1.0))
            self.artists.append(goal_text)

        # 绘制起点，并添加角标
        for i, start in enumerate(self.starts):
            start_color = self.colors[i % len(self.colors)]
            # 创建矩形，确保颜色正确设置
            rect = Rectangle((start[0] - 0.8, start[1] - 0.8), 1.6, 1.6,
                          facecolor=start_color,
                          edgecolor='white', linewidth=3.0, alpha=0.9, zorder=8)
            # 强制设置颜色
            rect.set_facecolor(start_color)
            rect.set_edgecolor('white')
            rect.set_alpha(0.9)
            self.patches.append(rect)
            # 添加"S"标签，带角标（S1, S2, ...）
            start_label = f'S{i+1}'
            start_text = self.ax.text(start[0], start[1], start_label, fontsize=12,
                                     color='black', weight='bold',
                                     ha='center', va='center', zorder=9,
                                     bbox=dict(boxstyle='round,pad=0.3', 
                                              facecolor='white', 
                                              edgecolor='black', 
                                              linewidth=2.0,
                                              alpha=1.0))
            self.artists.append(start_text)

        # 验证路径数据
        if not self.paths or len(self.paths) == 0:
            print("错误: 路径数据为空，无法创建可视化")
            return

        # 创建智能体
        agent_count = 0
        for i in range(len(self.paths)):
            if not self.paths[i] or not hasattr(self.paths[i], 'nodes') or len(self.paths[i].nodes) == 0:
                print(f"警告: 智能体 {i} 的路径为空，跳过")
                continue
                
            name = str(i + 1)
            start_pos = self.starts[i] if i < len(self.starts) else (0, 0)
            agent_color = self.colors[i % len(self.colors)]
            
            # 创建智能体圆圈 - 使用更大的尺寸和更明显的颜色
            self.agents[i] = Circle(start_pos, 0.8,  # 增大半径到0.8
                                    facecolor=agent_color,
                                    edgecolor='white', 
                                    linewidth=3.0,  # 更粗的边框
                                    alpha=1.0,  # 完全不透明
                                    zorder=10)  # 确保在最上层
            # 强制设置颜色，确保正确显示
            self.agents[i].set_facecolor(agent_color)
            self.agents[i].set_edgecolor('white')
            self.agents[i].set_alpha(1.0)
            self.agents[i].original_face_color = agent_color
            self.patches.append(self.agents[i])
            self.T = max(self.T, self.paths[i].nodes[-1].g)
            
            # 智能体标签 - 使用黑色文字在彩色背景上更清晰
            self.agent_names[i] = self.ax.text(start_pos[0], start_pos[1], name, 
                                              fontsize=13,  # 更大的字体
                                              color='black',  # 黑色文字在彩色背景上更清晰
                                              weight='bold',
                                              ha='center', va='center', 
                                              zorder=11,  # 在智能体之上
                                              bbox=dict(boxstyle='circle,pad=0.3', 
                                                       facecolor='white', 
                                                       edgecolor='black', 
                                                       linewidth=1.5,
                                                       alpha=0.9))
            self.artists.append(self.agent_names[i])
            agent_count += 1
            
            # 打印调试信息
            print(f"  智能体 {i+1}: 颜色={agent_color}, 位置=({start_pos[0]:.1f}, {start_pos[1]:.1f})")

        print(f"创建了 {agent_count} 个智能体")

        if self.T <= 0:
            print("错误: 动画时长为0，无法创建动画")
            return

        # 创建动画
        self.animation = animation.FuncAnimation(
            self.fig, self.animate_func, init_func=self.init_func,
            frames=int(self.T + 1) * 10, interval=50, blit=True, repeat=True
        )

    def init_func(self):
        # 确保所有补丁和艺术家都被添加，并强制设置颜色
        for p in self.patches:
            self.ax.add_patch(p)
            # 如果是智能体或起点/终点，确保颜色正确
            if isinstance(p, Circle) and hasattr(p, 'original_face_color'):
                p.set_facecolor(p.original_face_color)
                p.set_alpha(1.0)
            elif isinstance(p, Rectangle):
                # 检查是否是起点或终点，确保颜色正确
                pass  # 颜色已在创建时设置
        for a in self.artists:
            self.ax.add_artist(a)
        # 返回所有需要更新的对象
        return self.patches + self.artists + self.edge_lines

    def animate_func(self, t):
        for k, path in enumerate(self.paths):
            if k not in self.agents:
                continue
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
                        ratio = (t / 10 - t1) / (t2 - t1) if (t2 - t1) > 0 else 0
                        pos = (self.nodes[p1].x + ratio * (self.nodes[p2].x - self.nodes[p1].x),
                               self.nodes[p1].y + ratio * (self.nodes[p2].y - self.nodes[p1].y))
                        break

            self.agents[k].center = (pos[0], pos[1])
            self.agent_names[k].set_position((pos[0], pos[1]))

        # reset all colors - 确保颜色正确恢复
        for agent_id, agent in self.agents.items():
            # 确保使用原始颜色，并设置alpha为1.0确保完全不透明
            if hasattr(agent, 'original_face_color'):
                agent.set_facecolor(agent.original_face_color)
                agent.set_alpha(1.0)
                agent.set_edgecolor('white')

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
                    print(f"COLLISION! (agent-agent) ({i}, {j}) at time {t / 10:.2f}")
        
        # 返回所有需要更新的对象
        return self.patches + self.artists + self.edge_lines

    def show(self):
        """显示动画"""
        try:
            self.fig.canvas.manager.set_window_title('CCBS 路径规划可视化')
        except:
            pass  # 某些后端可能不支持设置标题
        
        # 打印调试信息
        print(f"\n{'='*60}")
        print(f"可视化信息:")
        print(f"  智能体数量: {len(self.paths)}")
        print(f"  动画总时长: {self.T:.2f}")
        print(f"  动画帧数: {int(self.T + 1) * 10}")
        print(f"  地图节点数: {len(self.nodes)}")
        print(f"  地图边数: {len(self.edge_lines)}")
        print(f"  起点数: {len(self.starts)}")
        print(f"  终点数: {len(self.goals)}")
        for i, path in enumerate(self.paths):
            if path and hasattr(path, 'nodes') and len(path.nodes) > 0:
                print(f"  智能体 {i+1}: {len(path.nodes)} 个节点, 总时间: {path.nodes[-1].g:.2f}")
        print(f"{'='*60}\n")
        
        plt.tight_layout()
        plt.show()

    def save(self, file_name, speed):
        self.animation.save(
            file_name,
            fps=10 * speed,
            dpi=200,
            savefig_kwargs={"pad_inches": 0, "bbox_inches": "tight"})
