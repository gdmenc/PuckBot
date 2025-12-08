import numpy as np
from typing import List, Dict, Optional
from scripts.kinematics.game_controller import GameController
from scripts.kinematics.states import PuckState

class GameEvaluator:
    def __init__(self, env, controller: GameController, robot_id: int = 1):
        self.env = env
        self.controller = controller
        self.robot_id = robot_id

        if robot_id == 1:
            self.own_goal_x = -1.06
            self.opponent_goal_x = 1.06
            self.attack_direction = 1.0
        else:
            self.own_goal_x = 1.06
            self.opponent_goal_x = -1.06
            self.attack_direction = -1.0

        self.goal_width_y = 0.20
        self.stats = {
            "total_shots": 0,
            "shots_on_target": 0,
            "saves": 0,
            "goals_conceded": 0,
            "goals_scored": 0,
            "current_rally_hits": 0,
            "hits_per_life": [],
            "game_length": 0.0
        }
        self.prev_puck_state: Optional[PuckState] = None
        self.last_hit_time = -1.0
        self.hit_cooldown = 0.5
        self.puck_radius = 0.04
        self.paddle_radius = 0.05
        self.hit_threshold = self.puck_radius + self.paddle_radius + 0.02
        self.was_threat = False

        self.goal_cooldown_time = 2.0
        self.last_goal_time = -1.0

    def reset(self):
        self.stats = {
            k: 0 if isinstance(v, int) else ([] if isinstance(v, list) else 0.0) for k, v in self.stats.items()
        }
        self.prev_puck_state = None
        self.last_hit_time = -1.0
        self.was_threat = False

    def update(self, dt: float):
        current_time = self.env.simulator.get_context().get_time()
        puck_state = self.controller.get_puck_state()
        paddle_pose = self.env.get_paddle_pose()
        paddle_pos = paddle_pos.translation()

        self._check_for_goals(puck_state)
        hit_occurred = self._check_for_hit(puck_state, paddle_pos, current_time)

        if hit_occurred:
            self.stats["current_rally_hits"] += 1
            self._analyze_hit(puck_state)
        

    def _check_for_goals(self, puck: PuckState):
        pos = puck.position
        current_time = puck.t
        
        if current_time - self.last_goal_time < self.goal_cooldown_time:
            return

        if self.robot_id == 1:
            own_goal = pos[0] < self.own_goal_x and abs(pos[1]) < self.goal_width_y
            opp_goal = pos[0] > self.opponent_goal_x and abs(pos[1]) < self.goal_width_y
        else:
            own_goal = pos[0] > self.own_goal_x and abs(pos[1]) < self.goal_width_y
            opp_goal = pos[0] < self.opponent_goal_x and abs(pos[1]) < self.goal_width_y

        if own_goal:
            self.stats["goals_conceded"] += 1
            self.stats["hits_per_life"].append(self.stats["current_rally_hits"])
            self.stats["current_rally_hits"] = 0
            self.last_goal_time = current_time
            print(f"Goal Conceded! (t={current_time:.2f}s)")

        elif opp_goal:
            self.stats["goals_scored"] += 1
            self.last_goal_time = current_time
            print(f"Goal Scored! (t={current_time:.2f}s)")

    def _check_for_hit(self, puck: PuckState, paddle_pos: np.ndarray, time: float) -> bool:
        if time - self.last_hit_time < self.hit_cooldown:
            return False

        dist = np.linalg.norm(puck.position[:2] - paddle_pos[:2])
        if dist < self.hit_threshold:
            self.last_hit_time = time
            self.stats["paddle_hits"] += 1
            return True
        return False

    def _analyze_hit(self, puck: PuckState):
        vel_x = puck.velocity[0]
        is_attacking = (vel_x > 0) if self.robot_id == 1 else (vel_x < 0)

        if is_attacking:
            self.stats["total_shots"] += 1
            if self._is_on_target(puck):
                self.stats["shots_on_target"] += 1

            self.was_threat = False

        else:
            if self.was_threat:
                self.stats["saves"] += 1
                self.was_threat = False

    def _is_on_target(self, puck: PuckState) -> bool:
        x, y = puck.position_2d
        vx, vy = puck.velocity_2d

        if abs(vx) < 0.01:
            return False

        t_impact = (self.opponent_goal_x - x) / vx
        if t_impact < 0:
            return False
        y_impact = y + vy * t_impact

        return abs(y_impact) < self.goal_width_y

    def _update_threat_status(self, puck: PuckState):
        x, y = puck.position_2d
        vx, vy = puck.velocity_2d
        is_incoming = (vx < 0) if self.robot_id == 1 else (vx > 0)

        if not is_incoming or abs(vx) < 0.01:
            return

        t_impact = (self.own_goal_x - x) / vx
        if t_impact > 0:
            y_impact = y + vy * t_impact
            if abs(y_impact) < self.goal_width_y:
                self.was_threat = True
            else:
                self.was_threat = False
        else:
            self.was_threat = False

    def print_summary(self):
        print("\n=== Evaluation Summary ===")
        print(f"Game Length: {self.stats['game_length']:.2f} s")
        print(f"Goals Scored: {self.stats['goals_scored']}")
        print(f"Goals Conceded: {self.stats['goals_conceded']}")
        
        acc = 0
        if self.stats['total_shots'] > 0:
            acc = (self.stats['shots_on_target'] / self.stats['total_shots']) * 100
        print(f"Hitting Accuracy: {acc:.1f}% ({self.stats['shots_on_target']}/{self.stats['total_shots']} on target)")
        
        save_pct = 0
        total_def_events = self.stats['saves'] + self.stats['goals_conceded']
        if total_def_events > 0:
            save_pct = (self.stats['saves'] / total_def_events) * 100
        print(f"Save Percentage: {save_pct:.1f}% ({self.stats['saves']} saves)")
        
        print(f"Total Paddle Hits: {self.stats['paddle_hits']}")
        
        avg_hits = 0
        if len(self.stats['hits_per_life']) > 0:
            avg_hits = sum(self.stats['hits_per_life']) / len(self.stats['hits_per_life'])
        elif self.stats['goals_conceded'] == 0:
             avg_hits = self.stats['current_rally_hits']
             
        print(f"Avg Hits Before Concede: {avg_hits:.1f}")