class BluffPlayer:

    def __init__(self, player_id, np_random):
        ''' Initilize a player.

        Args:
            player_id (int): The id of the player
        '''
        self.np_random = np_random
        self.player_id = player_id
        self.status = 'alive'
        self.hand = None

        # The chips that this player has put in until now
        self.in_chips = 0

    def get_state(self, all_chips, legal_actions):
        ''' Encode the state for the player

        Args:
            all_chips (int): The chips that all players have put in

        Returns:
            (dict): The state of the player
        '''
        # removed public state from get_state method
        state = {}
        state['hand'] = self.hand.get_index()
        state['all_chips'] = all_chips
        state['my_chips'] = self.in_chips
        state['legal_actions'] = legal_actions
        return state

    def get_player_id(self):
        ''' Return the id of the player
        '''
        return self.player_id
