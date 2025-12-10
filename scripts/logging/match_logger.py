"""
Match Logger - Comprehensive performance tracking for air hockey robots.

Tracks detailed metrics for each robot during a match including:
- Successful interceptions
- Saves (blocks when puck heading toward goal)
- Goals scored (on-target strikes)
- Accuracy statistics
- Game duration and score
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Tuple


class RobotStats:
    """Statistics tracker for a single robot."""
    
    def __init__(self, robot_name: str):
        self.robot_name = robot_name
        
        # Strike metrics
        self.strikes_attempted = 0
        self.strikes_contacted = 0  # Actually made contact with puck
        self.successful_interceptions = 0  # Contact made as planned at intercept point
        
        # Defensive metrics
        self.saves = 0  # Blocks when puck heading toward goal
        self.blocks = 0  # Any defensive contact
        
        # Offensive metrics
        self.goals_scored = 0
        self.shots_on_target = 0  # Strikes that went toward opponent goal
        
        # Position tracking
        self.total_distance_traveled = 0.0  # meters
        self.last_position: Optional[np.ndarray] = None
        
        # Timing
        self.reaction_times: List[float] = []  # Time from puck detection to action
        self.average_reaction_time = 0.0
        
        # Puck possession
        self.time_with_puck = 0.0  # seconds
        self.puck_touches = 0
        
    def record_strike_attempt(self):
        """Record that robot attempted a strike."""
        self.strikes_attempted += 1
    
    def record_puck_contact(self, was_planned: bool = False):
        """
        Record contact with puck.
        
        Args:
            was_planned: True if contact was at a planned intercept point
        """
        self.strikes_contacted += 1
        self.puck_touches += 1
        if was_planned:
            self.successful_interceptions += 1
    
    def record_save(self):
        """Record a successful save (blocked shot heading to goal)."""
        self.saves += 1
        self.blocks += 1
    
    def record_block(self):
        """Record a defensive block."""
        self.blocks += 1
    
    def record_goal(self, on_target: bool = True):
        """Record a goal scored."""
        self.goals_scored += 1
        if on_target:
            self.shots_on_target += 1
    
    def record_shot_on_target(self):
        """Record a shot that went toward opponent goal (but didn't score)."""
        self.shots_on_target += 1
    
    def update_position(self, position: np.ndarray):
        """Update robot position and calculate distance traveled."""
        if self.last_position is not None:
            distance = np.linalg.norm(position[:2] - self.last_position[:2])
            self.total_distance_traveled += distance
        self.last_position = position.copy()
    
    def record_reaction_time(self, reaction_time: float):
        """Record reaction time from puck detection to action."""
        self.reaction_times.append(reaction_time)
        self.average_reaction_time = np.mean(self.reaction_times)
    
    def update_puck_time(self, dt: float, has_puck: bool):
        """Update time with puck possession."""
        if has_puck:
            self.time_with_puck += dt
    
    @property
    def accuracy(self) -> float:
        """Calculate strike accuracy (contacts / attempts)."""
        if self.strikes_attempted == 0:
            return 0.0
        return self.strikes_contacted / self.strikes_attempted
    
    @property
    def interception_rate(self) -> float:
        """Calculate successful interception rate."""
        if self.strikes_attempted == 0:
            return 0.0
        return self.successful_interceptions / self.strikes_attempted
    
    @property
    def goal_conversion_rate(self) -> float:
        """Calculate goal conversion rate (goals / shots on target)."""
        if self.shots_on_target == 0:
            return 0.0
        return self.goals_scored / self.shots_on_target
    
    def to_dict(self) -> Dict:
        """Convert stats to dictionary for JSON serialization."""
        return {
            "robot_name": self.robot_name,
            "strikes": {
                "attempted": self.strikes_attempted,
                "contacted": self.strikes_contacted,
                "successful_interceptions": self.successful_interceptions,
                "accuracy": round(self.accuracy, 3),
                "interception_rate": round(self.interception_rate, 3),
            },
            "defense": {
                "saves": self.saves,
                "blocks": self.blocks,
            },
            "offense": {
                "goals_scored": self.goals_scored,
                "shots_on_target": self.shots_on_target,
                "goal_conversion_rate": round(self.goal_conversion_rate, 3),
            },
            "movement": {
                "total_distance_m": round(self.total_distance_traveled, 2),
                "average_reaction_time_ms": round(self.average_reaction_time * 1000, 1) if self.reaction_times else 0.0,
            },
            "possession": {
                "time_with_puck_s": round(self.time_with_puck, 2),
                "puck_touches": self.puck_touches,
            }
        }


class MatchLogger:
    """
    Comprehensive match logging system.
    
    Tracks performance metrics for all robots in a match and saves to JSON.
    """
    
    def __init__(self, log_dir: str = "logs", mode: str = "tournament"):
        # Create mode-specific subdirectory
        mode_dir_name = f"{mode.lower()}_mode"
        self.log_dir = Path(log_dir) / mode_dir_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.mode = mode
        self.start_time = datetime.now()
        self.match_id = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # Robot statistics
        self.robot_stats: Dict[str, RobotStats] = {}
        
        # Match-level statistics
        self.game_duration = 0.0
        self.final_score = {"right": 0, "left": 0}
        self.total_goals = 0
        
        # Event log
        self.events: List[Dict] = []
        
        # Puck tracking
        self.last_puck_pos: Optional[np.ndarray] = None
        self.puck_speed_history: List[float] = []
        
        print(f"[LOGGER] Match logger initialized")
        print(f"[LOGGER] Mode: {mode}, ID: {self.match_id}")
        print(f"[LOGGER] Log directory: {self.log_dir}")
    
    def add_robot(self, robot_name: str):
        """Add a robot to track."""
        if robot_name not in self.robot_stats:
            self.robot_stats[robot_name] = RobotStats(robot_name)
            print(f"[LOGGER] Tracking robot: {robot_name}")
    
    def log_event(self, event_type: str, robot: str, details: Dict = None):
        """
        Log a match event.
        
        Args:
            event_type: Type of event (e.g., "goal", "save", "strike")
            robot: Robot name
            details: Additional event details
        """
        event = {
            "time": self.game_duration,
            "type": event_type,
            "robot": robot,
            "details": details or {}
        }
        self.events.append(event)
    
    def update(self, current_time: float, puck_pos: np.ndarray, puck_vel: np.ndarray,
               robot_positions: Dict[str, np.ndarray] = None):
        """
        Update match statistics.
        
        Args:
            current_time: Current simulation time
            puck_pos: Puck position [x, y, z]
            puck_vel: Puck velocity [vx, vy, vz]
            robot_positions: Dict of robot_name -> position
        """
        self.game_duration = current_time
        
        # Track puck speed
        puck_speed = np.linalg.norm(puck_vel[:2])
        self.puck_speed_history.append(puck_speed)
        
        # Update robot positions
        if robot_positions:
            for robot_name, position in robot_positions.items():
                if robot_name in self.robot_stats:
                    self.robot_stats[robot_name].update_position(position)
                    
                    # Check if robot has puck (within 10cm)
                    if np.linalg.norm(position[:2] - puck_pos[:2]) < 0.10:
                        dt = 0.01  # Assuming 100Hz update
                        self.robot_stats[robot_name].update_puck_time(dt, True)
        
        self.last_puck_pos = puck_pos.copy()
    
    def record_goal(self, scoring_robot: str, opponent_robot: str):
        """Record a goal."""
        self.total_goals += 1
        
        # Update scores
        if scoring_robot in self.robot_stats:
            self.robot_stats[scoring_robot].record_goal()
        
        # Update final score
        side = "right" if "right" in scoring_robot else "left"
        self.final_score[side] += 1
        
        self.log_event("goal", scoring_robot, {
            "score": self.final_score.copy()
        })
        
        print(f"[LOGGER] GOAL! {scoring_robot} scored - Score: {self.final_score}")
    
    def save_match_log(self) -> str:
        """
        Save complete match log to JSON file.
        
        Returns:
            Path to saved log file
        """
        # Compile full match data
        match_data = {
            "match_info": {
                "match_id": self.match_id,
                "mode": self.mode,
                "start_time": self.start_time.isoformat(),
                "duration_seconds": round(self.game_duration, 2),
            },
            "final_score": self.final_score,
            "robot_stats": {
                name: stats.to_dict() 
                for name, stats in self.robot_stats.items()
            },
            "match_summary": {
                "total_goals": self.total_goals,
                "average_puck_speed_ms": round(np.mean(self.puck_speed_history) if self.puck_speed_history else 0.0, 2),
                "max_puck_speed_ms": round(np.max(self.puck_speed_history) if self.puck_speed_history else 0.0, 2),
            },
            "events": self.events,
        }
        
        # Save to file
        log_file = self.log_dir / f"match_{self.match_id}.json"
        with open(log_file, 'w') as f:
            json.dump(match_data, f, indent=2)
        
        print(f"\n[LOGGER] Match log saved: {log_file}")
        return str(log_file)
    
    def print_summary(self):
        """Print match summary to console."""
        print("\n" + "="*70)
        print("MATCH SUMMARY")
        print("="*70)
        print(f"Match ID: {self.match_id}")
        print(f"Mode: {self.mode}")
        print(f"Duration: {self.game_duration:.1f}s")
        print(f"Final Score: Right {self.final_score['right']} - {self.final_score['left']} Left")
        print("="*70)
        
        for robot_name, stats in self.robot_stats.items():
            print(f"\n{robot_name.upper()} STATISTICS:")
            print(f"  Strikes: {stats.strikes_attempted} attempted, {stats.strikes_contacted} contacted ({stats.accuracy:.1%} accuracy)")
            print(f"  Interceptions: {stats.successful_interceptions} successful ({stats.interception_rate:.1%} rate)")
            print(f"  Defense: {stats.saves} saves, {stats.blocks} blocks")
            print(f"  Offense: {stats.goals_scored} goals, {stats.shots_on_target} shots on target")
            print(f"  Movement: {stats.total_distance_traveled:.1f}m traveled")
            if stats.reaction_times:
                print(f"  Reaction: {stats.average_reaction_time*1000:.1f}ms average")
        
        print("="*70 + "\n")
