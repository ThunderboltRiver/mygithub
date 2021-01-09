import zipfile as zf
import dionysus as d
import numpy as np
from sgfmill import sgf
from scipy.spatial.distance import pdist

class kihu_persistence:
	def __init__(self, Win_or_Lose_kihus, step = 1, dim, radius):
	##initializing parameter
		self.radius = radius
		self.dim = dim
		self.step = step
		self.kihus = Win_or_Lose_kihus
		##making filltration
		self.two_dim_fill_list = []
		for kihu in self.kihus:
			remainder = len(kihu) % step
			turns = np.arange(0, len(kihu),step)
			if remainder >= step / 2:
				turns = np.append(turns, len(kihu))
			else:
				turns[-1] = len(kihu)
			self.two_dim_fill_list.append([d.fill_rips(pdist(kihu[:t]), dim, radius) for t in turns])
				
		
	
	def random_choice_homology(self, choice_size = 1, replace_bool = False , root = None, plot = None, show_bool = True):
		colors = ['r','g','b','c','m','y','k']
		if root == None:
			choice_fills = np.random.choice(self.two_dim_fill_list, size = choice_size, replace = replace_bool)
			for fill in choice_fills:
				p = d.homology_persistence(fill)
				dgms = d.init_diagrams(p, fill)
				for i in len(dgms):
					if plot = 'bards':
						try:
							d.plot.plot_bards(dgms[i],show = show_bool, **bards_style = colors[i])
						finally:
							continue
							
					elif plot == 'density':
						try:
							d.plot.plot_diagram_density(dgms[i], show = show_bool)
							
						finally:
							continue
							
					else:
						try:
							d.plot.plot_diagram(dgms[i])
						finally:
							continue
						
							
					
			
		
		
		
		


class GO_data:
	def __init__(self,zip_file):
		self.winners = []
		self.board_size = []
		self.kihus = []
		self.b_players = []
		self.w_players = []
		self.Win_kihus = []
		self.Lose_kihus = []
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
		for i, kihu in self.kihus:
	    	win_kihu = np.array([goishi[1] for goishi in kihu if goishi[0] == self.winners[i]])
	    	lose_kihu = np.array([goishi[1] for goishi in kihu if goishi[0] != self.winners[i]])
	    	self.Win_kihus.append(win_kihu)
	    	self.Lose_kihus.append(lose_kihu)
	
	def kihu_random_plot(self,show_num = 5):
		kihus_len = len(self.Win_kihus)
		show_index = np.random.choice(np.arange(kihus_len), size = show_num, replace = False)
		show_Win, show_Lose = self.Win_kihus[show_index], self.Lose_kihus[show_index]
		l = len(show_Win)
		fig = plt.figure()
			for i in range(l):
				ax = fig.add_subplot(l, 2, i)
				ax.sccater(show_Win[i], color = 'r')
				ax = fig.add_subplot(l, 2, i + 1)
				ax.accater(show_Lose[i], color = 'b')
				
		plt.show()
		
