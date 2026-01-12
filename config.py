class Config:
    def __init__(self) -> None:
        self.agent_size = 0.5       # Agent size in world units: (0, 0.5]
        self.hlh_type = 2           # 0 - no hlh, 1 - solve lpp by simplex, 2 - greedly take disjoint conflicts
        self.precision = 0.1        # Precision for waiting time determination
        self.timelimit = 300        # Time limit in seconds

        self.use_precalculated_heuristic = False  # True: reverse Dijskstra, False: Euclidean distance
        self.use_disjoint_splitting = True
        self.use_cardinal = True  # Prioritize cardinal over semi-cardinal over regular conflicts
        self.use_corridor_symmetry = False
        self.use_target_symmetry = False
        self.use_rl = True  # True: use PPO to apply branching strategy in CCBS
        self.rl_model_path = "ppo_empty-16-16-random.zip"  # 可根据需求切换 PPO 模型
