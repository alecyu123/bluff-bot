from rlcard.games.base import Card
from rlcard.games.limitholdem import Dealer

class BluffDealer(Dealer):

    def __init__(self, np_random):
        '''
        Initialize a bluff-game dealer class
        Modified deck card to only include numebrs from 1-10.
        Suit is simply used to represent multiple cards with the same value 
        '''
        self.np_random = np_random
        self.deck = [Card('S', 'A'), Card('S', '2'), Card('S', '3'), Card('S', '4'), Card('S', '5'), Card('S', '6'), Card('S', '7'), Card('S', '8'), Card('S', '9'), Card('S', 'T'),
                     Card('H', 'A'), Card('H', '2'), Card('H', '3'), Card('H', '4'), Card('H', '5'), Card('H', '6'), Card('H', '7'), Card('H', '8'), Card('H', '9'), Card('H', 'T')] 
        self.shuffle()
        self.pot = 0

# Updated the deck