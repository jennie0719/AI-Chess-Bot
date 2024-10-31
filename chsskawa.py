#!/usr/bin/env python
import chess
import random
import sys
import matplotlib.pyplot as plt
import networkx as nx
# import pydot
# from networkx.drawing.nx_pydot import graphviz_layout

board = chess.Board()
OPENING_SEQUENCES = {
    "Queen's Gambit":"rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR",
    "Spanish Opening":"r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R",
    "Sicilian Defense":"r1bqkb1r/pp1p1ppp/2n1pn2/8/3NP3/2N5/PPP2PPP/R1BQKB1R",
    "Vienna Game":"rnbqkb1r/pppp1ppp/8/4p3/2B1n3/2N5/PPPP1PPP/R1BQK1NR"
}
# PIECE_VALUE = { # TODO: update piece value
#     'king': 200,
#     'pawn': 1,
#     'knight': 3,
#     'bishop': 3,
#     'rook': 5,
#     'queen': 9
# }
pawntable = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, -20, -20, 10, 10, 5,
    5, -5, -10, 0, 0, -10, -5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, 5, 10, 25, 25, 10, 5, 5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 0, 0, 0, 0, 0, 0]

knightstable = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50]
bishopstable = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20]
rookstable = [
    0, 0, 0, 5, 5, 0, 0, 0,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0]
queenstable = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 5, 5, 5, 5, 5, 0, -10,
    0, 0, 5, 5, 5, 5, 0, -5,
    -5, 0, 5, 5, 5, 5, 0, -5,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20]
kingstable = [
    20, 30, 10, 0, 0, 10, 30, 20,
    20, 20, 0, 0, 0, 0, 20, 20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30]
DRAW_VALUE = 0
MAX_LEVEL = 7
RESTRICT_MOVES = 10
NODES_SHOWING = 3
HEIGHT_SHOWING = 4

def evaluate(b): # TODO: update evaluation
    if b.is_insufficient_material():
        return DRAW_VALUE

    wk = len(b.pieces(chess.KING, chess.WHITE))
    bk = len(b.pieces(chess.KING, chess.BLACK))
    
    wp = len(b.pieces(chess.PAWN, chess.WHITE))
    bp = len(b.pieces(chess.PAWN, chess.BLACK))

    wn = len(b.pieces(chess.KNIGHT, chess.WHITE))
    bn = len(b.pieces(chess.KNIGHT, chess.BLACK))

    wb = len(b.pieces(chess.BISHOP, chess.WHITE))
    bb = len(b.pieces(chess.BISHOP, chess.BLACK))

    wr = len(b.pieces(chess.ROOK, chess.WHITE))
    br = len(b.pieces(chess.ROOK, chess.BLACK))

    wq = len(b.pieces(chess.QUEEN, chess.WHITE))
    bq = len(b.pieces(chess.QUEEN, chess.BLACK))

    material = 100 * (wp - bp) + 320 * (wn - bn) + 330 * (wb - bb) + 500 * (wr - br) + 900 * (wq - bq)
    pawnsq = sum([pawntable[i] for i in board.pieces(chess.PAWN, chess.WHITE)])
    pawnsq = pawnsq + sum([-pawntable[chess.square_mirror(i)]
                        for i in board.pieces(chess.PAWN, chess.BLACK)])
    knightsq = sum([knightstable[i] for i in board.pieces(chess.KNIGHT, chess.WHITE)])
    knightsq = knightsq + sum([-knightstable[chess.square_mirror(i)]
                            for i in board.pieces(chess.KNIGHT, chess.BLACK)])
    bishopsq = sum([bishopstable[i] for i in board.pieces(chess.BISHOP, chess.WHITE)])
    bishopsq = bishopsq + sum([-bishopstable[chess.square_mirror(i)]
                            for i in board.pieces(chess.BISHOP, chess.BLACK)])
    rooksq = sum([rookstable[i] for i in board.pieces(chess.ROOK, chess.WHITE)])
    rooksq = rooksq + sum([-rookstable[chess.square_mirror(i)]
                        for i in board.pieces(chess.ROOK, chess.BLACK)])
    queensq = sum([queenstable[i] for i in board.pieces(chess.QUEEN, chess.WHITE)])
    queensq = queensq + sum([-queenstable[chess.square_mirror(i)]
                            for i in board.pieces(chess.QUEEN, chess.BLACK)])
    kingsq = sum([kingstable[i] for i in board.pieces(chess.KING, chess.WHITE)])
    kingsq = kingsq + sum([-kingstable[chess.square_mirror(i)]
                        for i in board.pieces(chess.KING, chess.BLACK)])
    eval = material + pawnsq + knightsq + bishopsq + rooksq + queensq + kingsq
    
    if board.turn:
        return eval
    else:
        return -eval
    
def bestMoves(board):
    '''retrns a list of RESTRICT_MOVES best next move according to evaluation'''
    legal_moves = list(board.legal_moves)
    index = [i for i in range(len(legal_moves))]
    evaluation = []
    for move in legal_moves:
        board.push(move)
        evaluation.append(evaluate(board))
        board.pop()
    best = sorted(zip(evaluation, index), reverse=board.turn)[:RESTRICT_MOVES]
    return [legal_moves[i[1]] for i in best]

def max_value(board, depth, alpha, beta):
    """
    Returns the maximum value for the current player on the board 
    using alpha-beta pruning.
    """
    if depth == 0 or board.outcome():
        return evaluate(board)
    v = float('-inf')
    for move in bestMoves(board):
        board.push(move)
        v = max(v, min_value(board, depth-1, alpha, beta))
        alpha = max(alpha, v)
        board.pop()
        if alpha >= beta:
            break
    return v

def min_value(board, depth, alpha, beta):
    """
    Returns the minimum value for the current player on the board 
    using alpha-beta pruning.
    """
    if depth == 0 or board.outcome():
        return evaluate(board)
    v = float('inf')
    for move in bestMoves(board):
        board.push(move)
        v = min(v, max_value(board, depth-1, alpha, beta))
        beta = min(beta, v)
        board.pop()
        if alpha >= beta:
            break
    return v

def minimax(board,depth):
    '''return best next move'''
    if depth == 0 or board.outcome():
        return None
    next_moves = bestMoves(board)
    if board.turn == chess.WHITE:
        v = float('-inf')
        opt_move = None
        for move in next_moves:
            board.push(move)
            new_value = min_value(board, depth-1, float('-inf'), float('inf'))
            if new_value > v:
                v = new_value
                opt_move = move
            board.pop()
        return opt_move
    else:
        v = float('inf')
        opt_move = None
        for move in next_moves:
            board.push(move)
            new_value = max_value(board, depth-1, float('-inf'), float('inf'))
            if new_value < v:
                v = new_value
                opt_move = move
            board.pop()
        return opt_move

def draw_keepLeaves(G, fr, maxmin):
    '''a funciton to keep at most NODES_SHOWING number of leaves'''
    name = dict(nx.bfs_successors(G, fr,1))[fr]
    score = [G.nodes[node]['value'] for node in name]
    if len(name) <= NODES_SHOWING:
        return
    s = sorted(zip(score, name), reverse=maxmin)[NODES_SHOWING:]
    remove = [n[1] for n in s]
    for node in s:
        remove += list(nx.descendants(G,node[1]))
    G.remove_nodes_from(remove)

def draw_minimax(G, board, depth, parent):
    '''a duplicate of minimax, but performs only minimax and record it into a graph'''
    
    if depth == 0 or board.outcome():
          value = evaluate(board)
          G.add_node(parent, value = value, ran = False)
          return value, None
    next_moves = bestMoves(board)
    G.add_node(parent)
    count = 1
    if board.turn:
        value = float('-inf')
        for move in next_moves:
            board.push(move)
            tmp = draw_minimax(G, board,depth-1,parent+f"{count:02}")[0]
            if tmp > value:
                value = tmp
                best_move = move
            G.add_edge(parent, parent+f"{count:02}", label=move)
            board.pop()
            count +=1
        draw_keepLeaves(G, parent,True)
    else:
        value = float('inf')
        for move in next_moves:
            board.push(move)
            tmp = draw_minimax(G, board,depth-1,parent+f"{count:02}")[0]
            if tmp < value:
                value = tmp
                best_move = move
            G.add_edge(parent, parent+f"{count:02}", label=move)
            board.pop()
            count +=1
        draw_keepLeaves(G, parent, False)
    G.nodes[parent]['value'] = value
    G.nodes[parent]['ran'] = False
    return value, best_move

def draw_alphabetaprune(G, alpha, beta, root, maximizingPlayer):
    '''an all-in-one alpha-beta pruning function and record it into a graph'''
    G.nodes[root]['ran'] = True
    children = dict(nx.bfs_successors(G, root, 1))[root]
    if not children:
          return G.nodes[root]['value']
    
    if maximizingPlayer:
        maxEval = float('-inf')
        for child in children:
            eval = draw_alphabetaprune(G, alpha, beta, child, False)
            maxEval = max(maxEval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cut-off
        return maxEval
    else:
        minEval = float('inf')
        for child in children:
            eval = draw_alphabetaprune(G, alpha, beta, child, False)
            minEval = min(minEval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cut-off
        return minEval

def trimTreeTo(G, root):
    '''helper function for removing nodes over height of HEIGHT_SHOWING'''
    nodes = sum(dict(nx.bfs_successors(G, root, HEIGHT_SHOWING)).values(),[])
    nodes.append(root) #prevent root to be removed
    remove = []
    for node in list(G.nodes()):
        if node not in nodes:
            remove.append(node)
    G.remove_nodes_from(remove)

def draw():
    '''Prints out the minimax visualization'''
    draw_board = chess.Board()
    draw_board.set_fen(OPENING_SEQUENCES["Queen's Gambit"])
    H = nx.DiGraph()
    best_first = draw_minimax(H, draw_board, MAX_LEVEL, 'A')[1]
    trimTreeTo(H,'A')
    draw_alphabetaprune(H, float('-inf'), float('inf'), 'A', True)
    
    # pos = graphviz_layout(H, prog="dot")
    pos = nx.spring_layout(H)
    edge_labels = {(u, v): f'{d["label"]}' for u, v, d in H.edges(data=True)}
    colors = ['green' if H.nodes[k]['ran'] else 'lightblue' for k in H.nodes()]
    nx.draw(H, pos, with_labels=True, node_color=colors, font_size=2, edge_color='blue')
    nx.draw_networkx_edge_labels(H, pos=pos, edge_labels=edge_labels)
    labels = nx.get_node_attributes(H, 'value')
    nx.draw_networkx_labels(H, pos, labels, font_size = 4)
    plt.title(best_first)
    plt.grid(True)
    plt.show() 

def move_bestOfMinimax():
    '''Returns the best next move according to minimax and alpha-beta pruning'''
    legal_moves = list(board.legal_moves)
    if legal_moves:
        return minimax(board, MAX_LEVEL)
    return None

def uci(msg: str):
    '''Returns result of UCI protocol given passed message'''
    if msg == "uci":
        print("id name chsskawa Chess Bot")
        print("id author Jennie Lin")
        print("uciok")
    elif msg == "isready":
        print("readyok")
    elif msg.startswith("position startpos moves"):
        board.clear()
        # board.set_fen(OPENING_SEQUENCES["Queen's Gambit"])
        board.set_fen(chess.STARTING_FEN)
        moves = msg.split()[3:]
        for move in moves:
            board.push(chess.Move.from_uci(move))
    elif msg.startswith("position fen"):
        fen = msg.removeprefix("position fen ")
        board.set_fen(fen)
    elif msg.startswith("go"):
        move = move_bestOfMinimax() 
        print(f"bestmove {move}")
    elif msg.startswith("draw"):
        draw()
    elif msg == "quit":
        sys.exit(0)
    return
    
def main():
    '''Expects to forever be passed UCI messages'''
    try:
        while True:
            uci(input())
    except Exception:
        print("Fatal Error")
    
if __name__ == "__main__":
    # print(sys.argv)
    main()