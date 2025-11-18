PuckBot/
├── simulation.py                     # Main entry point & game loop
│
├── scripts/                          # Core systems
│   ├── drake_implementation.py       # Physics environment (EXISTING)
│   ├── simple_motion_controller.py   # Motion planning (EXISTING)
│   ├── demo_basic_motion.py          # Testing demos (EXISTING)
│   └── trajectory_executor.py        # NEW: Real-time trajectory execution
│
├── strategy/                         # AI/Planning layer (TO CREATE)
│   ├── __init__.py
│   ├── puck_predictor.py            # Puck trajectory prediction
│   ├── defensive_planner.py         # Defensive intercept selection
│   ├── offensive_planner.py         # Offensive shot planning
│   ├── zone_strategy.py             # Zone-based interception (Fig 2)
│   └── strategy_coordinator.py      # High-level decision making
│
├── utils/                            # Shared utilities (TO CREATE)
│   ├── __init__.py
│   ├── constants.py                 # Table dims, zones, robot limits
│   ├── geometry.py                  # Collision, intersection math
│   └── visualization.py             # Plotting/debugging tools
│
├── tests/                            # Unit tests (TO CREATE)
│   ├── test_puck_prediction.py
│   ├── test_motion_controller.py
│   └── test_strategy.py
│
├── config/                           # Configuration (OPTIONAL)
│   └── game_config.yaml             # Tunable parameters
│
├── Dockerfile                        # Container setup (EXISTING)
├── requirements.txt                  # Dependencies (EXISTING)
├── README.md                         # Documentation
└── proposal.tex                      # Project proposal (EXISTING)


┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                            │
│  simulation.py - Main game loop, orchestration, scoring         │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STRATEGY LAYER                               │
│  strategy/                                                       │
│    - puck_prediction.py    (trajectory forecasting)             │
│    - defensive_strategy.py (blocking logic)                     │
│    - offensive_strategy.py (shot selection)                     │
│    - zone_based_strategy.py (Fig 2 from proposal)              │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CONTROL LAYER                                │
│  scripts/                                                        │
│    - simple_motion_controller.py (IK + trajectories)            │
│    - trajectory_executor.py      (real-time execution)          │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHYSICS LAYER                                │
│  scripts/                                                        │
│    - drake_implementation.py (environment, robots, puck)        │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    UTILITY LAYER                                │
│  utils/                                                          │
│    - geometry.py    (collision detection, line intersections)   │
│    - kinematics.py  (FK/IK helpers)                            │
│    - constants.py   (table dimensions, zones, limits)          │
└─────────────────────────────────────────────────────────────────┘