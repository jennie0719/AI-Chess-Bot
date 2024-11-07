# Hachessware Chess Bot

**Author:** Jennie Lin

## Overview
Hachessware Chess Bot is a chess engine built in Python that utilizes the Universal Chess Interface (UCI) protocol to communicate moves. It features a minimax algorithm with alpha-beta pruning for efficient decision-making, allowing it to evaluate potential moves deeply while optimizing for computational efficiency. The bot includes a visualization tool to illustrate the decision tree of moves, offering insights into the minimax algorithm’s functioning.

## Features
- **UCI Protocol Support:** Communicate with chess engines using UCI commands such as `position`, `go`, and `isready`.
- **Minimax Algorithm with Alpha-Beta Pruning:** Explores the best possible moves by simulating several levels of gameplay.
- **Board Visualization:** Graphs the minimax tree, highlighting pruned branches and optimal paths.
- **Evaluation Function:** Scores board positions based on material, king safety, and other heuristic factors.

## Installation
Ensure the following Python packages are installed:

```bash
pip install chess networkx matplotlib
```

## Usage
To run Hachessware Chess Bot, simply execute:

```bash
python hachessware_bot.py
```
It will then listen for UCI commands to process moves.

To run tournament, simply execute:
```bash
python tournament.py
```
It will then run 5 rounds of games between Hachessware chess bot against the random move bot.

## Commands
The bot supports the following UCI commands:

- ```uci``` - Initiates UCI mode, providing basic bot information.
- ```isready``` - Confirms readiness to receive commands.
- ```position startpos moves ...``` - Sets up a board from the start position with a sequence of moves.
- ```position fen ...``` - Sets up the board with a specific FEN position.
- ```go``` - Finds and returns the best move based on current board position.
- ```draw``` - Visualizes the minimax tree of possible moves.
- ```quit``` - Exits the program.
## Code Structure
- ```evaluate``` function: Calculates a score for the board position based on material and positional factors.
- ```minimax``` function: Implements the minimax algorithm with alpha-beta pruning, exploring potential moves.
- ```draw_minimax``` function: Builds a NetworkX graph to visualize the decision tree of moves and pruning points.
- ```find_best_move``` function: Selects the best move for the bot, calling minimax with a set depth.
- **UCI command handling:** Processes commands from external chess GUIs that support UCI, allowing full integration.
##Visualization
To visualize the decision tree, call the ```draw``` command after starting the bot. The graph highlights:

- **Pruned branches:** Indicating nodes skipped due to alpha-beta pruning.
- **Optimal path:** Representing the moves the bot considers the strongest.
## Future Work
Potential improvements could include:

- Enhanced evaluation function with additional heuristics.
- Deeper exploration of move sequences using quiet search or iterative deepening.
- Improved visualization with more extensive and detailed branching.
## Acknowledgements
This project uses the ```python-chess```, ```networkx```, and ```matplotlib``` libraries to handle chess logic, graph structures, and visualizations.


This format provides structure, code details, and installation/usage steps that make it easy for users to get started with your bot.
