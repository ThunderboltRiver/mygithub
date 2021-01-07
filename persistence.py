import zipfile as zf
import dionysus as d
import numpy as np
from sgfmill import sgf
from scipy.spatial.distance import pdist

class GO_persistence:
	def __init__(self,zip_file):
	    self.winners = []
	    self.board_size = []
	    self.kihus = []
	    self.b_players = []
	    sefl.w_players = []
	    with zf.ZipFile(zip_file) as zip_data:
	        infos = [info for info in zip_data.infolist() if '.sgf' in info.filename]
	        for info in infos:
	            with zip_data.open(info.filename, 'r') as f:
	            	game = sgf.Sgf_game.from_bytes(f.read())
	            self.winners.append(game.get_winner())
	            self.board_size.append(game.get_size())
	            root_node = game.get_root()
	            self.b_players.append(root_node.get('PB'))
	            self.w_players.append(root_node.get('PW'))
	            self.kihus.append([node.get_move() for node in game.get_main_sequence() if node.get_move() != (None, None) ])

	def B_or_W(self):
	    Black_list = []
	    White_list = []
	    for kihu in self.kihus:
	    	Black = np.array([goishi[1] for goishi in kihu if goishi[0] == 'b'])
	    	White = np.array([goishi[1] for goishi in kihu if goishi[0] == 'w'])
	    	Black_list.append(Black)
	    	White_list.append(White)
	
	
	    return Black_kihus, White_kihus
	
	def B_or_W_fill(self,dim, radius):
		B, W = self.B_or_W()
		B_fill_list = [d.fill_rips(b, dim, radius) for b in B]
		W_fill_list = [d.fill_rips(w, dim, radius) for w in W]
		
		return B_fill_list, W_fill_list
		