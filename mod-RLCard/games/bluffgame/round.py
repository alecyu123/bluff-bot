# -*- coding: utf-8 -*-
''' Implement Bluff-Game Round class
'''

from rlcard.games.limitholdem import Round

class BluffRound(Round):
    ''' Round can call other Classes' functions to keep the game running
    '''

    def __init__(self, raise_amount, allowed_raise_num, num_players, np_random):
        ''' Initilize the round class

        Args:
            raise_amount (int): the raise amount for each raise
            allowed_raise_num (int): The number of allowed raise num
            num_players (int): The number of players
        '''
        super(BluffRound, self).__init__(raise_amount, allowed_raise_num, num_players, np_random=np_random)
