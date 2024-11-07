#!/usr/bin/env python
import chess
import random
import sys
import networkx as nx
import matplotlib.pyplot as plt
# import pydot
# from networkx.drawing.nx_pydot import graphviz_layout

board = chess.Board()

piece_values = {
    chess.KING: 200,
    chess.QUEEN: 9,
    chess.ROOK: 5,
    chess.BISHOP: 3,
    chess.KNIGHT: 3,
    chess.PAWN: 1
}

def evaluate(board):
    if board.is_game_over():
        if board.is_checkmate():
            return -1000 if board.turn == chess.WHITE else 1000
        else:
            return 0
    total_score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values.get(piece.piece_type, 0)
            adjusted_value = value if piece.color == chess.WHITE else -value
            total_score += adjusted_value
    white_king_safety = 0 if board.has_kingside_castling_rights(chess.WHITE) else -5
    black_king_safety = 0 if board.has_kingside_castling_rights(chess.BLACK) else 5
    return total_score + white_king_safety + black_king_safety

def evaluate_move(board, move):
    if board.is_capture(move):
        return 10
    return 0
    # score = 0
    #
    # # 1. Capture Value: Reward captures more based on the piece being captured.
    # if board.is_capture(move):
    #     captured_piece = board.piece_at(move.to_square)
    #     if captured_piece:
    #         score += piece_values.get(captured_piece.piece_type, 0) * 10
    #
    # # 2. Promotion: Reward moves that result in promotion.
    # if move.promotion:
    #     score += piece_values.get(move.promotion, 0) * 10
    #
    # # 3. Center Control: Reward moves that move to or control center squares.
    # center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    # if move.to_square in center_squares:
    #     score += 5
    #
    # # 4. Piece Activity: Reward moves that develop pieces from their original squares.
    # piece = board.piece_at(move.from_square)
    # if piece and not board.is_check():
    #     if piece.piece_type in [chess.KNIGHT, chess.BISHOP, chess.QUEEN]:
    #         score += 2
    #
    # # 5. Piece Safety: Penalize moves that result in an undefended piece.
    # board.push(move)
    # if board.is_attacked_by(board.turn, move.to_square):
    #     score -= 5
    # board.pop()
    #
    # return score

def minimax(board, depth, alpha, beta, is_maximizing, quiet_search=False):
    if depth == 0 or board.is_game_over():
        return evaluate(board)
    if is_maximizing:
        max_eval = -float('inf')
        moves = list(board.legal_moves)
        # Prioritize captures and other tactical moves
        moves.sort(key=lambda move: -evaluate_move(board, move), reverse=True)
        moves = moves[:10]  # Limit to top 10 moves

        for move in moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False, quiet_search or board.is_capture(move))
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Alpha-beta pruning
        return max_eval
    else:
        min_eval = float('inf')

        moves = list(board.legal_moves)
        moves.sort(key=lambda move: evaluate_move(board, move))
        moves = moves[:10]  # Limit to top 10 moves

        for move in moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True, quiet_search or board.is_capture(move))
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha-beta pruning
        return min_eval

def draw_minimax(G, board, depth, alpha, beta, is_maximizing, notPruned, quiet_search=False):
    if depth == 0 or board.is_game_over():
        return evaluate(board)
    root = board.fen()
    track = notPruned
    G.add_node(root, ran=track)
    if is_maximizing:
        max_eval = -float('inf')
        moves = list(board.legal_moves)
        moves.sort(key=lambda move: -evaluate_move(board, move), reverse=True)
        moves = moves[:10]
        for move in moves:
            board.push(move)
            eval = draw_minimax(G, board, depth - 1, alpha, beta, False, track and notPruned,
                                quiet_search or board.is_capture(move))  # Track pruning here
            draw_addNode(G, board, root, move, eval, track and notPruned)  # Update track based on pruning
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:  # Prune if beta <= alpha
                track = False  # Set track to False after pruning
        G.nodes[root]["utility"] = max_eval
        return max_eval
    else:
        min_eval = float('inf')
        moves = list(board.legal_moves)
        moves.sort(key=lambda move: evaluate_move(board, move))
        moves = moves[:10]
        for move in moves:
            board.push(move)
            eval = draw_minimax(G, board, depth - 1, alpha, beta, True, track and notPruned,
                                quiet_search or board.is_capture(move))  # Track pruning here
            draw_addNode(G, board, root, move, eval, track and notPruned)  # Update track based on pruning
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:  # Prune if beta <= alpha
                track = False  # Set track to False after pruning
        G.nodes[root]["utility"] = min_eval
        return min_eval

def draw_addNode(G, board, root, move, val, notprune):
    fen = board.fen()
    G.add_node(fen, utility=val, ran=notprune)
    G.add_edge(root, fen, move=move)

def draw_visualize():
    draw_board = chess.Board()
    draw_board.set_fen("rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR")
    best_move = None
    best_value = -float('inf')
    G = nx.DiGraph()
    root = draw_board.fen()
    G.add_node(root, utility=0, ran=True)
    moves = list(draw_board.legal_moves)
    moves.sort(key=lambda move: -evaluate_move(draw_board, move), reverse=True)
    alpha = -float('inf')
    beta = float('inf')
    track = True
    for move in moves:

        draw_board.push(move)
        move_value = draw_minimax(G, draw_board, 3, alpha, beta, False, track)
        draw_addNode(G, draw_board, root, move, move_value, track)
        draw_board.pop()
        if move_value > best_value:
            best_move = move
            best_value = move_value
        alpha = max(alpha, move_value)
        if beta <= alpha:
            track = False
    G.nodes[root]["utility"] = best_value
    level = 4
    queue = [draw_board.fen()]
    top_nodes = [draw_board.fen()]
    while level > 0:
        nodes_in_this_level = []
        while queue:
            tmp = {}
            cur = queue.pop(0)
            for child in G.successors(cur):
                tmp[child] = G.nodes[child]['utility']
            top_3 = sorted(tmp.items(), key=lambda x: x[1], reverse=(level % 2 == 0))[:3]
            for node, score in top_3:
                top_nodes.append(node)
                nodes_in_this_level.append(node)
        queue = nodes_in_this_level
        level -= 1

    # Create a subgraph with only the top nodes at each level
    subgraph = G.subgraph(top_nodes)

    # Set up node sizes and labels based on scores.
    labels = {node: subgraph.nodes[node]["utility"] for node in subgraph}

    # Plot the subgraph.
    # plt.figure(figsize=(8, 6))
    # pos = nx.spring_layout(subgraph)  # Use spring layout for better visualization
    # nx.draw(subgraph, pos, with_labels=False, node_size=100, node_color='red', font_weight='bold')
    # nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=12, font_color="black")

    # pos = graphviz_layout(subgraph, prog="dot")
    pos = nx.spring_layout(subgraph)
    edge_labels = {(u, v): f'{d["move"]}' for u, v, d in subgraph.edges(data=True)}
    colors = ['green' if subgraph.nodes[k]['ran'] else 'lightblue' for k in subgraph.nodes()]
    nx.draw(subgraph, pos, with_labels=False, node_color=colors, font_size=2, edge_color='blue')
    nx.draw_networkx_edge_labels(subgraph, pos=pos, edge_labels=edge_labels)
    labels = nx.get_node_attributes(subgraph, 'utility')
    nx.draw_networkx_labels(subgraph, pos, labels, font_size=4)
    plt.figtext(0.5, 0.01, str(best_move) + ": " + str(best_value), ha="center", fontsize=8)
    plt.grid(True)
    plt.show()

def find_best_move(board, depth):
    best_move = None
    if board.turn:
        best_value = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            move_value = minimax(board, depth - 1, -float('inf'), float('inf'), board.turn)
            board.pop()
            if move_value > best_value:
                best_move = move
                best_value = move_value
        return best_move
    else:
        best_value = float('inf')
        for move in board.legal_moves:
            board.push(move)
            move_value = minimax(board, depth - 1, -float('inf'), float('inf'), board.turn)
            board.pop()
            if move_value < best_value:
                best_move = move
                best_value = move_value
        return best_move

def uci(msg: str):
    '''Returns result of UCI protocol given passed message'''
    if msg == "uci":
        print("id name Hachessware Chess Bot")
        print("id author Jennie Lin")
        print("uciok")
    elif msg == "isready":
        print("readyok")
    elif msg.startswith("position startpos moves"):
        board.clear()
        board.set_fen(chess.STARTING_FEN)
        moves = msg.split()[3:]
        for move in moves:
            board.push(chess.Move.from_uci(move))
    elif msg.startswith("position fen"):
        fen = msg.removeprefix("position fen ")
        board.set_fen(fen)
    elif msg.startswith("go"):
        move = find_best_move(board, 5)
        print(f"bestmove {move}")
    elif msg.startswith("draw"):
        draw_visualize()
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
