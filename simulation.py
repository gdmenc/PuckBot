class GameState:
    """
    Manages game state and scoring.
    """
    def __init__(self):
        self.player1_score = 0
        self.player2_score = 0
        self.game_over = False
        self.winner = None
        self.goal_scored = False
        self.goal_side = None
        self.reset_pending = False