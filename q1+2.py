import numpy as np
import random
import math
import multiprocessing
import time
import sys


class Node:
    # a simple node class for the mission.
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0
        self.action = ""
        self.actions_considered = []
        self.boards_considered = []

    def add_action(self, action):
        self.actions_considered.append(action)

    def set_boards_considered(self, boards_considered):
        self.boards_considered = boards_considered

    def __eq__(self, other):
        return (self.position == other.position).sum() == 36


def find_path(starting_board, goal_board, search_method, detail_output):
    if search_method not in np.arange(6):
        print("No path found")
        return
    starting_board = np.asarray(starting_board)                    # convert to a numpy array
    goal_board = np.asarray(goal_board)
    if (goal_board == 2).sum() > (starting_board == 2).sum():      # assure there are more agents in the starting board than in the goal board.
        print("No path found")
        return
    if not assure_valid_boards(board1=starting_board, board2=goal_board):
        print("No path found")
        return

    # gen open and closed list
    open_list = []
    closed_list = []

    # gen a start and end nodes
    end_node = Node(None, goal_board)
    end_node.g = end_node.h = end_node.f = 0
    start_node = Node(None, starting_board)

    start_node.g = 0
    start_node.h = heuristic_calculation(goal_board=goal_board, current_board=starting_board)
    start_node.f = start_node.g + start_node.f
    # Add the start node to the open list
    open_list.append(start_node)

    if search_method == 1:
        if not a_star_search(end_node, detail_output, open_list, closed_list, limit=5000):
            print("No path found")
    elif search_method == 2:
        if not hill_climbing_search(end_node, detail_output, open_list, closed_list, limit=2000):  # I used a smaller limit than in a star, in order to let you see the random restart
            print("No path found")
    elif search_method == 3:
        if not simulated_annealing_search(end_node, open_list, closed_list, limit=5000):
            print("No path found")
    elif search_method == 4:
        local_beam_search(end_node, open_list, closed_list, limit=3000)
        return
    elif search_method == 5:
        if not genetic_search(start_node, end_node, detail_output, open_list):
            print("No path found")
        return


def a_star_search(end_node, detail_output, open_list, closed_list, limit):
    # Loop until you find the path, or after 5000 iterations
    counter = 0
    while len(open_list) > 0 and counter <= limit:
        counter += 1
        # Get the current node
        current_node = open_list[0]
        current_index = 0

        for index, item in enumerate(open_list):  # choose the best node in the open list to explore
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Remove (pop) current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # check if the current node is the goal board
        if current_node == end_node:
            final_print(current_node, detail_output, end_node)
            return True

        # Gen children:
        children = []
        new_boards = gen_new_boards(current_node.position)
        for new_position in new_boards:
            node_position = np.copy(new_position)        # Get node position
            new_node = Node(current_node, node_position) # Create new node
            children.append(new_node)                    # add to the children list

        # Loop through children
        for child in children:
            # check if the node was already explored
            for closed_child in closed_list:
                if child == closed_child:
                    break

            # calculate the f, g, and h values
            child.g = current_node.g + 1
            child.h = heuristic_calculation(goal_board=end_node.position, current_board=child.position)
            child.f = child.g + child.h
            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue
            # Add the child to the open list
            open_list.append(child)
    return False


def simulated_annealing_search(end_node, open_list, closed_list, limit):
    cooling_rate = 0.98
    temperature = 1000
    current_node = open_list[0]

    counter = 0
    while len(open_list) > 0 and counter <= limit and current_node.g <= 100 and temperature > 0.1:
        counter += 1
        # update the temperature and the index
        temperature = temperature * cooling_rate
        current_index = 0
        improved = False
        # choose the best node in the open list to explore:
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                improved = True
                current_node = item
                current_index = index
        if improved:
            current_node.add_action("action:" + current_node.action + " ; probability:1.0")
        else:
            current_node.actions_considered = []

        i = 0
        random.shuffle(open_list)
        while not improved and len(open_list) > 0 and i < len(open_list) - 1:
            random_child = open_list[i]
            i += 1
            loss = random_child.f - current_node.f
            probability = math.exp(- loss / temperature)
            a = f"action:{random_child.action}; probability:{round(probability,3)}"
            current_node.add_action(action=a)
            if random.random() < probability:
                improved = True
                current_node = random_child
                current_index = i

        # Remove (pop) current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # check if the current node is the goal board
        if current_node == end_node:
            final_print_annealing(current_node)
            return True

        # Gen children:
        children = []
        new_boards_and_actions = gen_new_boards_and_actions(current_node.position)
        for new_position_and_action in new_boards_and_actions:
            new_node = Node(current_node, new_position_and_action['board'])  # Create new node
            new_node.action = new_position_and_action['action']
            children.append(new_node)  # add to the children list

        # Loop through children
        for child in children:
            explored = False
            # check if the node was already explored
            # do not insert the child to the open list if it is already in the closed list.
            for closed_child in closed_list:
                if child == closed_child:
                    explored = True
                    break
            # calculate the f, g, and h values
            if not explored:
                child.g = current_node.g + 1
                child.h = heuristic_calculation(goal_board=end_node.position, current_board=child.position)
                child.f = child.g + child.h
                # Child is already in the open list
                for open_node in open_list:
                    if child == open_node and child.g > open_node.g:
                        explored = True
                        break
            if not explored:
                open_list.append(child)
    return False


def final_print_annealing(current_node):
    print("we found a path")
    path = []
    current = current_node
    while current is not None:
        path.append(current)
        current = current.parent
    path = path[::-1]
    print_path_annealing(path)
    return


def print_path_annealing(path):
    for i in range(len(path)):
        print(f"Board {i+1}:")
        print_board(path[i].position)
        if i < len(path) - 1:
            for action in path[i + 1].actions_considered:
                print(action)
        print('-----')


def hill_climbing_search(end_node, detail_output, open_list, closed_list, limit):
    # Loop until you find the path, or after 5000 iterations
    counter = 0
    restarts = 0
    if a_star_search(end_node, detail_output, open_list, closed_list, limit=limit):
        return True

    current_node = open_list[0]

    # Gen children:
    children = []
    new_boards = gen_new_boards(current_node.position)
    for new_position in new_boards:
        node_position = np.copy(new_position)        # Get node position
        new_node = Node(current_node, node_position)  # Create new node
        children.append(new_node)                    # add to the children list

    for child in children:
        # calculate the f, g, and h values
        child.g = current_node.g + 1
        child.h = heuristic_calculation(goal_board=end_node.position, current_board=child.position)
        child.f = child.g + child.h
        # Child is already in the open list
        for open_node in open_list:
            if child == open_node and child.g > open_node.g:
                continue
            # Add the child to the open list
        open_list.append(child)

    get_top_n(open_list, n=len(open_list))         # returns a sorted by f list
    open_list = open_list[1:]                      # remove the best one because it is the one that was already explored
    random_5 = np.random.choice(open_list, 5, replace=False)  # chose random 5 child, for 5 random restarts

    for node in random_5:
        open_list = [node]
        if a_star_search(end_node, detail_output, open_list, closed_list, limit=limit):
            return True


def local_beam_search(end_node, open_list, closed_list, limit):
    # Loop until you find the path, or after 5000 iterations
    counter = 0
    while len(open_list) > 0 and counter <= limit:
        counter += 1
        # Get the current node
        current_node = open_list[0]
        current_index = 0

        for index, item in enumerate(open_list):  # choose the best node in the open list to explore
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Remove (pop) current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # check if the current node is the goal board
        if current_node == end_node:
            final_print_local_beam(current_node)
            return

        # Gen children:
        children = []
        new_boards = gen_new_boards(current_node.position)
        for new_position in new_boards:
            node_position = np.copy(new_position)        # Get node position
            new_node = Node(current_node, node_position) # Create new node
            children.append(new_node)                    # add to the children list

        # Loop through children
        temp_list = []
        for child in children:
            # check if the node was already explored
            for closed_child in closed_list:
                if child == closed_child:
                    break

            # calculate the f, g, and h values
            child.g = current_node.g + 1
            child.h = heuristic_calculation(goal_board=end_node.position, current_board=child.position)
            child.f = child.g + child.h
            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue
            # Add the child to the open list
            temp_list.append(child)
        temp_list = get_top_n(population=temp_list, n=3)
        current_node.set_boards_considered(temp_list)
        for child in temp_list:
            open_list.append(child)
    print("No path found")


def final_print_local_beam(current_node):
    print("we found a path")
    path = []
    current = current_node
    while current is not None:
        path.append(current)
        current = current.parent
    path = path[::-1]
    options = ['a', 'b', 'c']
    for i in range(len(path)):
        print(f"Board {i + 1}:")
        print_board(path[i].position)
        for j in range(len(path[i].boards_considered)):
            print(f"Board{i+2}{options[j]}:")
            print_board(path[i].boards_considered[j].position)
            print('-----')
    return


def final_print(current_node, detail_output, end_node):
    print("we found a path")
    path = []
    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    path = path[::-1]
    print_path(path, detail_output, end_node.position)
    return


def get_top_n(population, n):
    """
    :param population: a list of nodes.
    :param n: number of nodes to return.
    :return: a list in length n, populated with the nodes with the smallest f.
    """
    population = [a for a in sorted(population, key=lambda x: x.f, reverse=False)]  # sort by f
    if n < len(population):
        return population[:n]
    else:
        return population


def combine_two_boards(board1, board2):
    """
    :param board1: a board.
    :param board2: a board.
    :return: a board from the upper half of board 1, and the lower half of board 2
    """
    # vertical combination
    if random.random() < 0.5:
        row_cut = random.randint(1, 5)
        return np.concatenate((board1[:row_cut], board2[row_cut:]), axis=0)
    # horizontal combination
    else:
        board1 = np.rot90(board1)
        board2 = np.rot90(board2)
        row_cut = random.randint(1, 5)
        return np.rot90(np.concatenate((board1[:row_cut], board2[row_cut:]), axis=0))


def genetic_selection(population):
    weights = []
    score_sum = 0
    # choose the ten nodes with the highest f score
    population = get_top_n(population, n=10)
    # calculate the probability of each node:
    # we have min problem, so I changed the score to "score-1000" and turned it to a max problem, I could have used something nicer, but this is an efficient one.
    for node in population:
        score_sum += node.f
        weights.append(node.f)
    np.seterr(divide='ignore', invalid='ignore')
    weights = (1 - (np.asarray(weights)/score_sum))
    weights = weights/sum(weights)
    # pick two nodes randomly according to the weighted probability
    choices = random.choices(population=population, weights=weights, k=2)
    # combine the two nodes boards to one and return it
    combined_board = combine_two_boards(choices[0].position, choices[1].position)
    if random.random() < 0.9:
        mutation_happened = "yes"
        combined_board = gen_random_mutation(combined_board)
    else:
        mutation_happened = "no"
    probability_board1 = round((1 - (choices[0].f / score_sum))/sum(weights), 3)
    probability_board2 = round((1 - (choices[1].f / score_sum))/sum(weights), 3)
    return {"new_board": combined_board, "board1": choices[0], "board2": choices[1],
            "mutation_happened": mutation_happened, "probability_board1":probability_board1, "probability_board2":probability_board2}


def is_in_board(location):
    return location[0] in range(0, 6) and location[1] in range(0, 6)


def find_agents(board):
    agents = []
    for i in range(6):
        for j in range(6):
            if board[i][j] == 2:
                agents.append((i, j))
    return agents


def gen_random_mutation(board):
    agents_locations = find_agents(board)
    random.shuffle(agents_locations)
    possible_moves = []
    for agent in agents_locations:
        possible_moves = []
        # move up
        if is_in_board((agent[0]-1, agent[1])):
            if board[agent[0]-1, agent[1]] == 0:
                possible_moves.append("up")
        # move right
        if is_in_board((agent[0], agent[1] + 1)):
            if board[agent[0], agent[1] + 1] == 0:
                possible_moves.append("right")
        # move down
        if is_in_board((agent[0]+1, agent[1])):
            if board[agent[0]+1, agent[1]] == 0:
                possible_moves.append("down")
        # step out
        if agent[0] == 5:
            possible_moves.append("out")
        # move left
        if is_in_board((agent[0], agent[1] - 1)):
            if board[agent[0], agent[1] - 1] == 0:
                possible_moves.append("left")
        if len(possible_moves) > 0:
            board[agent[0]][agent[1]] = 0
            random.shuffle(possible_moves)
            move = possible_moves[0]
            if move == 'out':
                continue
            elif move == 'up':
                board[agent[0] - 1][agent[1]] = 2
            elif move == 'right':
                board[agent[0]][agent[1] + 1] = 2
            elif move == 'down':
                board[agent[0]+1][agent[1]] = 2
            elif move == 'left':
                board[agent[0]][agent[1]-1] = 2
            return board
    return board


def assure_valid_boards(board1, board2):
    """
    assure the boards are 6X6 and have force fields in the same locations.
    :param board1: an array
    :param board2: an array
    :return: True/False.
    """
    board1 = np.asarray(board1)
    board2 = np.asarray(board2)
    if board1.shape != (6, 6):
        return False
    if board2.shape != (6, 6):
        return False
    for i in range(6):
        for j in range(6):
            if board1[i][j] == 1 and board2[i][j] != 1:
                return False
            if board1[i][j] != 1 and board2[i][j] == 1:
                return False
    return True


def gen_new_boards(current_board):
    new_boards = []
    for i in range(6):
        for j in range(6):
            if current_board[i][j] == 2:
                # move up
                board = np.copy(current_board)
                if i != 0:
                    if board[i - 1][j] == 0:
                        board[i][j] = 0
                        board[i - 1][j] = 2
                        new_boards.append(board)
                # move right
                board = np.copy(current_board)
                if j != 5:  # we are in the right column
                    if board[i][j + 1] == 0:
                        board[i][j] = 0
                        board[i][j + 1] = 2
                        new_boards.append(board)
                # move down
                board = np.copy(current_board)
                if i == 5:  # we are in the last row
                    board[i][j] = 0
                    new_boards.append(board)
                elif board[i + 1][j] == 0:
                    board[i][j] = 0
                    board[i + 1][j] = 2
                    new_boards.append(board)
                # move left
                board = np.copy(current_board)
                if j != 0:  # we are in the left column
                    if board[i][j - 1] == 0:
                        board[i][j] = 0
                        board[i][j - 1] = 2
                        new_boards.append(board)
    return new_boards


def gen_new_boards_and_actions(current_board):
    boards_and_actions = []
    for i in range(6):
        for j in range(6):
            if current_board[i][j] == 2:
                # move up
                if i != 0:
                    if current_board[i - 1][j] == 0:
                        board = np.copy(current_board)
                        board[i][j] = 0
                        board[i - 1][j] = 2
                        boards_and_actions.append({'board': board, 'action': f"({i+1},{j+1})->({i},{j+1})"})
                # move right
                if j != 5:  # we are in the right column
                    if current_board[i][j + 1] == 0:
                        board = np.copy(current_board)
                        board[i][j] = 0
                        board[i][j + 1] = 2
                        boards_and_actions.append({'board': board, 'action': f"({i+1},{j+1})->({i+1},{j+2})"})
                # move down
                if i == 5:  # we are in the last row
                    board = np.copy(current_board)
                    board[i][j] = 0
                    boards_and_actions.append({'board':board, 'action': f"({i+1},{j+1})->(out)"})
                elif current_board[i + 1][j] == 0:
                    board = np.copy(current_board)
                    board[i][j] = 0
                    board[i + 1][j] = 2
                    boards_and_actions.append({'board': board, 'action': f"({i+1},{j+1})->({i+2},{j+1})"})
                # move left
                if j != 0:  # we are in the left column
                    if current_board[i][j - 1] == 0:
                        board = np.copy(current_board)
                        board[i][j] = 0
                        board[i][j - 1] = 2
                        boards_and_actions.append({'board': board, 'action': f"({i+1},{j+1})->({i+1},{j})"})
    return boards_and_actions


def print_board(board):
    dic = {2: "*", 1: "@", 0: " "}
    print("  1 2 3 4 5 6")
    for i in range(0, 6):
        print(f"{i+1}:", end="")
        for j in range(6):
            if j < 5:
                print(dic[board[i][j]], end=" ")
            else:
                print(dic[board[i][j]])


def print_path(path, detail_output, goal_board):
    for i in range(len(path)):
        print(f"Board {i+1}:")
        print_board(path[i])
        if i == 1 and detail_output:
            print(f"Heuristic: {heuristic_calculation(goal_board=goal_board, current_board=path[i])}")
        print('-----')


def gen_start_and_goal_boards():
    """
    :return: two new random boards, only for checking myself.
    """
    start_board = np.zeros(shape=(6, 6), dtype=int)
    max_force_fields = 8
    max_agents_on_start = 10
    max_agents_on_goal = 7
    # place the force_fields on the board
    num_of_force_fields = np.random.randint(0, max_force_fields)
    for i in range(num_of_force_fields):
        start_board[np.random.randint(0, 6), np.random.randint(0, 6)] = 1
    # copy the board
    goal_board = np.copy(start_board)
    # place the agents on the start board
    num_of_agents_start = np.random.randint(0, max_agents_on_start)
    for i in range(num_of_agents_start):
        start_board[np.random.randint(0, 6), np.random.randint(0, 6)] = 2
    # place the agents on the end board
    num_of_agents_goal = np.random.randint(0, max_agents_on_goal)
    for i in range(num_of_agents_goal):
        goal_board[np.random.randint(0, 6), np.random.randint(0, 6)] = 2
    return {'start_board': start_board, 'goal_board': goal_board}


def manhattan_distance(location_1, location_2):
    x = abs(location_1[0] - location_2[0])
    x += abs(location_1[1] - location_2[1])
    return x


def score_agents_num(goal_board, current_board):
    return ((current_board == 2).sum() - (goal_board == 2).sum()) * 3
    # 1 for each redundant agent on the board


def heuristic_calculation(goal_board, current_board):
    score = 0
    score += score_agents_num(goal_board=goal_board, current_board=current_board)
    # gen all possible moves
    moves = []
    current = np.copy(current_board)
    for i in range(6):           # 6 for all the moves
        for j in range(6):
            moves.append((i, j))
            moves.append((-i, j))
            moves.append((i, -j))
            moves.append((-i, -j))
    moves = list(dict.fromkeys(moves))   # removes duplicates
    moves = [a for a in sorted(moves, key=lambda x: abs(x[1]) + abs(x[0]), reverse=False)] ## sort moves
    # iterate over the goal board and check each agent
    for i in range(6):
        for j in range(6):
            if goal_board[i][j] == 2:
                found_agent = False
                for move in moves:
                    new_location = [a + b for a, b in zip(move, [i, j])]
                    if is_in_board(new_location):
                        if current[new_location[0]][new_location[1]] == 2:  # meaning we have an agent in the location we checked
                            score += manhattan_distance(new_location, [i, j])
                            ### current[new_location[0]][new_location[1]] = 0
                            found_agent = True
                            break
                if not found_agent:
                    score += 100
    return score


def print_genetic_creation(genetic_creation):
    print(f"Starting board 1 (probability of selection from population::{genetic_creation['probability_board1']}):")
    print_board(genetic_creation['board1'].position)
    print("-----")
    print(f"Starting board 2 (probability of selection from population::{genetic_creation['probability_board2']}):")
    print_board(genetic_creation['board2'].position)
    print("-----")
    print(f"Result board (mutation happened::{genetic_creation['mutation_happened']}):")
    print_board(genetic_creation['new_board'])


def final_print_genetic(current_node):
    print("we found a path")
    path = []
    current = current_node
    is_valid_path = True
    first = True
    while current is not None and is_valid_path:
        if not first:
            is_valid_path = is_move_away(current.parent.position, current.position)
            first = False
        path.append(current.position)
        current = current.parent
    path = path[::-1]
    if is_valid_path:
        for i in range(len(path)):
            print(f"Board {i + 1}:")
            print_board(path[i])
            print('-----')
    else:
        print("We could not find the path, because with genetic algorithm we can not always find the path")
    return


def genetic_search(current_node, end_node, detail_output, open_list):
    is_first_genetic_creation = True
    if current_node == end_node:
        final_print(current_node, detail_output, end_node)
        return True
    # create the initial population
    population = []
    new_boards = gen_new_boards(current_node.position)
    for new_position in new_boards:
        node_position = np.copy(new_position)  # Get node position
        new_node = Node(current_node, node_position)  # Create new node
        new_node.g = current_node.g + 1
        new_node.h = heuristic_calculation(goal_board=end_node.position, current_board=new_node.position)
        new_node.f = new_node.h + new_node.g
        population.append(new_node)  # add to the children list

    counter = 0
    while len(open_list) > 0 and counter <= 20000:
        counter += 1
        if is_first_genetic_creation:
            first_genetic_creation = genetic_selection(population=population)
        # check if the current node is the goal board
        for node in population:
            if node == end_node:
                final_print_genetic(node)
                print_genetic_creation(first_genetic_creation)
                return True

        new_population = []
        for i in range(10):
            if is_first_genetic_creation:
                new_node = Node(current_node, first_genetic_creation["new_board"])
                is_first_genetic_creation = False
            else:
                new_node = Node(current_node, genetic_selection(population=population)["new_board"])
            new_node.g = current_node.g + 1
            new_node.h = heuristic_calculation(goal_board=end_node.position, current_board=new_node.position)
            new_node.f = new_node.h + new_node.g
            new_population.append(new_node)
        population = np.copy(new_population)
    return False


def is_move_away(board1, board2):
    board1_agents = find_agents(board1)
    board2_agents = find_agents(board2)
    if len(board2_agents) > len(board1_agents) or len(board2_agents) < len(board1_agents) - 1:
        return False
    board1_agents_temp = board1_agents[:]
    for agent in board1_agents_temp:
        if agent in board2_agents:
            board1_agents.pop(board1_agents.index(agent))
            board2_agents.pop(board2_agents.index(agent))
    if len(board1_agents) > 1 or len(board2_agents) > 1:
        return False
    if len(board1_agents) == 0 and len(board2_agents) == 0:
        return True
    row_diff = 0
    col_diff = 0
    if board1_agents and board2_agents:
        row_diff = abs(board1_agents[0][0] - board2_agents[0][0])
        col_diff = abs(board1_agents[0][1] - board2_agents[0][1])
    if row_diff + col_diff > 1:
        return False
    return True


if __name__ == '__main__':
    # Start foo as a process
    p = multiprocessing.Process(target=find_path, name="Foo", args=(10,))
    p.start()
    # Wait 5 min for find_path
    time.sleep(60*5)
    # Terminate foo
    if p.is_alive():
        p.terminate()
        # Cleanup
        p.join()
        print("No path found")
    sys.exit()


