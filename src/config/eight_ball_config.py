# 8-Ball Pool Game Configuration

# Ball Classifications for 8-Ball Pool
EIGHT_BALL_CLASSES = {
    # Existing general detection
    0: "ball",           # Generic ball (legacy)
    1: "table",          # Table surface
    
    # Specific ball types for 8-ball pool
    12: "cue_ball",      # White ball (cue ball)
    13: "eight_ball",    # Black ball (8-ball)
    14: "solid_ball",    # Solid color balls (1-7)
    15: "stripe_ball",   # Striped balls (9-15)
    
    # Pocket types (existing)
    2: "BottomLeft",
    3: "BottomRight", 
    4: "IntersectionLeft",
    5: "IntersectionRight",
    6: "MediumLeft",
    7: "MediumRight",
    8: "SemicircleLeft",
    9: "SemicircleRight",
    10: "TopLeft",
    11: "TopRight"
}

# Colors for 8-ball specific visualization
EIGHT_BALL_COLORS = {
    12: (255, 255, 255), # White for cue ball
    13: (0, 0, 0),       # Black for 8-ball
    14: (0, 255, 255),   # Yellow for solids
    15: (255, 0, 255),   # Magenta for stripes
    
    # Table and pockets
    1: (255, 0, 0),      # Blue for table
    2: (255, 255, 0), 3: (255, 255, 0), 4: (255, 255, 0),
    5: (255, 255, 0), 6: (255, 255, 0), 7: (255, 255, 0),
    8: (255, 255, 0), 9: (255, 255, 0), 10: (255, 255, 0), 11: (255, 255, 0)
}

# Game States
class GameState:
    WAITING = "waiting"           # Waiting for game to start
    BREAK = "break"              # Break shot
    OPEN_TABLE = "open_table"    # Table is open, no assignment yet
    ASSIGNED = "assigned"        # Players assigned to solids/stripes
    SHOOTING_8 = "shooting_8"    # Player can legally shoot 8-ball
    GAME_OVER = "game_over"      # Game finished

# Player Types
class PlayerType:
    NONE = "none"
    SOLIDS = "solids"    # Player shooting balls 1-7
    STRIPES = "stripes"  # Player shooting balls 9-15

# Shot Types
class ShotType:
    BREAK = "break"
    NORMAL = "normal"
    EIGHT_BALL = "eight_ball"

# Foul Types
class FoulType:
    CUE_BALL_POCKETED = "cue_ball_pocketed"
    WRONG_BALL_FIRST = "wrong_ball_first"
    NO_BALL_HIT = "no_ball_hit"
    NO_RAIL_CONTACT = "no_rail_contact"
    EIGHT_BALL_EARLY = "eight_ball_early"
    EIGHT_BALL_WRONG_POCKET = "eight_ball_wrong_pocket"
    
# Win Conditions
class WinCondition:
    EIGHT_BALL_POCKETED = "eight_ball_pocketed"
    OPPONENT_FOUL_ON_EIGHT = "opponent_foul_on_eight"

# 8-Ball Pool Rules Configuration
EIGHT_BALL_RULES = {
    "break_rules": {
        "must_hit_rack": True,
        "minimum_balls_to_rail": 4,
        "cue_ball_pocket_rerack": False,  # Player continues but with ball in hand
        "eight_ball_pocket_loses": True
    },
    
    "assignment_rules": {
        "first_ball_pocketed_determines": True,
        "must_call_eight_ball_pocket": True
    },
    
    "legal_shot_rules": {
        "must_hit_target_group_first": True,
        "must_pocket_ball_or_hit_rail": True,
        "cue_ball_must_not_pocket": True
    },
    
    "eight_ball_rules": {
        "must_clear_group_first": True,
        "must_call_pocket": True,
        "wrong_pocket_loses": True,
        "early_eight_ball_loses": True
    }
}

# Ball tracking for game logic
BALL_NUMBERS = {
    # Solid balls (1-7)
    1: {"type": "solid", "color": "yellow", "class_id": 14},
    2: {"type": "solid", "color": "blue", "class_id": 14},
    3: {"type": "solid", "color": "red", "class_id": 14},
    4: {"type": "solid", "color": "purple", "class_id": 14},
    5: {"type": "solid", "color": "orange", "class_id": 14},
    6: {"type": "solid", "color": "green", "class_id": 14},
    7: {"type": "solid", "color": "brown", "class_id": 14},
    
    # Eight ball
    8: {"type": "eight", "color": "black", "class_id": 13},
    
    # Striped balls (9-15)
    9: {"type": "stripe", "color": "yellow_stripe", "class_id": 15},
    10: {"type": "stripe", "color": "blue_stripe", "class_id": 15},
    11: {"type": "stripe", "color": "red_stripe", "class_id": 15},
    12: {"type": "stripe", "color": "purple_stripe", "class_id": 15},
    13: {"type": "stripe", "color": "orange_stripe", "class_id": 15},
    14: {"type": "stripe", "color": "green_stripe", "class_id": 15},
    15: {"type": "stripe", "color": "brown_stripe", "class_id": 15},
    
    # Cue ball
    0: {"type": "cue", "color": "white", "class_id": 12}
}

# Game scoring
GAME_SCORING = {
    "win_points": 1,
    "loss_points": 0,
    "foul_penalty": 0  # No points penalty, just loss of turn
}

# Detection thresholds for 8-ball specific
EIGHT_BALL_DETECTION = {
    "confidence_threshold": 0.4,
    "cue_ball_confidence": 0.5,    # Higher confidence for cue ball
    "eight_ball_confidence": 0.6,  # Highest confidence for 8-ball
    "min_ball_distance": 20,       # Minimum distance between balls
    "pocket_distance_threshold": 30 # Distance to consider ball pocketed
}

# Pocket mapping for called shots
POCKET_MAPPING = {
    "TopLeft": "top_left",
    "TopRight": "top_right", 
    "MediumLeft": "side_left",
    "MediumRight": "side_right",
    "BottomLeft": "bottom_left",
    "BottomRight": "bottom_right"
}

# Game timing
GAME_TIMING = {
    "shot_timeout": 60,      # Seconds per shot
    "game_timeout": 1800,    # 30 minutes per game
    "break_timeout": 120     # 2 minutes for break
} 