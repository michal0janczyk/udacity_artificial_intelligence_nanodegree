from sample_players import DataPlayer


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE:
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        import random

        self.queue.put(random.choice(state.actions()))

        depth = 1
        while 1:
            self.queue.put(self.principal_variation_search(state, depth))
            depth += 1

    def alpha_beta_pruning(self, state, depth):
        alpha = float("-inf")
        beta = float("inf")
        actions_list = state.actions()
        best_move = actions_list[0] if actions_list else None
        max_player = True
        for single_action in actions_list:
            new_state = state.result(single_action)
            next_move = self.search_alpha_beta(new_state, depth - 1, alpha, beta, max_player)
            if next_move > alpha:
                alpha = next_move
                best_move = single_action
        return best_move

    def search_alpha_beta(self, state, depth, alpha, beta, max_player):
        if max_player:
            next_move = -float("inf")
            for single_action in state.actions():
                new_state = state.result(single_action)
                next_move = max(
                    next_move,
                    self.search_alpha_beta(new_state, depth - 1, alpha, beta, False),
                )
                alpha = max(alpha, next_move)
                if alpha >= beta:
                    break
        else:
            next_move = float("inf")
            for single_action in state.actions():
                new_state = state.result(single_action)
                next_move = min(
                    next_move, self.search_alpha_beta(new_state, depth - 1, alpha, beta, True)
                )
                beta = min(beta, next_move)
                if alpha >= beta:
                    break

        return next_move

    def principal_variation_search(self, state, depth):
        alpha = float("-inf")
        beta = float("inf")
        single_action = state.actions()
        best_move = single_action[0] if single_action else None
        max_player = True
        next_move = -float("inf")
        for i, action in enumerate(single_action):
            current_state = state.result(action)
            if i == 0:
                next_move = max(
                    next_move,
                    self.search_pvs(
                        current_state, depth - 1, alpha, beta, max_player
                    ),
                )
            else:
                next_move = max(
                    next_move,
                    self.search_pvs(
                        current_state, depth - 1, alpha, alpha + 1, max_player
                    ),
                )
                if next_move > alpha:
                    next_move = max(
                        next_move,
                        self.search_pvs(
                            current_state, depth - 1, alpha, beta, max_player
                        ),
                    )
            if next_move > alpha:
                alpha = next_move
                best_move = action
        return best_move

    def search_pvs(self, state, depth, alpha, beta, max_player):
        if state.terminal_test():
            return state.utility(self.player_id)
        if depth <= 0:
            return self.score(state)
        if max_player:
            next_move = -float("inf")
            for i, single_action in enumerate(state.actions()):
                new_state = state.result(single_action)
                if i == 0:
                    next_move = max(
                        next_move,
                        self.search_pvs(
                            new_state, depth - 1, alpha, beta, False
                        ),
                    )
                else:
                    next_move = max(
                        next_move,
                        self.search_pvs(
                            new_state, depth - 1, alpha, alpha + 1, False
                        ),
                    )
                    if next_move > alpha:
                        next_move = max(
                            next_move,
                            self.search_pvs(
                                new_state, depth - 1, alpha, beta, False
                            ),
                        )
                alpha = max(alpha, next_move)
                if alpha >= beta:
                    break
        else:
            next_move = float("inf")
            for i, single_action in enumerate(state.actions()):
                new_state = state.result(single_action)
                if i == 0:
                    next_move = min(
                        next_move,
                        self.search_pvs(
                            new_state, depth - 1, alpha, beta, True
                        ),
                    )
                else:
                    next_move = min(
                        next_move,
                        self.search_pvs(
                            new_state, depth - 1, beta - 1, beta, True
                        ),
                    )
                    if next_move < beta:
                        next_move = min(
                            next_move,
                            self.search_pvs(
                                new_state, depth - 1, alpha, beta, True
                            ),
                        )
                beta = min(beta, next_move)
                if alpha >= beta:
                    break
        return next_move

    def score(self, state):
        current_location = state.locs[self.player_id]
        opponent_location = state.locs[1 - self.player_id]
        current_liberties = state.liberties(current_location)
        opponent_liberties = state.liberties(opponent_location)
        return len(current_liberties) - len(opponent_liberties)
