BOARD_SIZE = 4
NUM_CELLS = BOARD_SIZE * BOARD_SIZE
# action encoding: 0..15 square, 16..31 circle, 32..95 arrow (dir = (a-32)%4)
NUM_ACTIONS = NUM_CELLS * (1 + 1 + 4)

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left

# MCTS / training defaults
C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPS = 0.25
MCTS_SIMS = 200
TEMPERATURE = 1.0

# network
POLICY_CHANNELS = 64
RES_BLOCKS = 6

# self-play
MAX_MOVES = 200  # safety cap
