import zipfile as zf
import dionysus as d
import numpy as np
import matplotlib.pyplot as plt
from sgfmill import sgf
from scipy.spatial.distance import pdist

class kihu_persistence:
    def __init__(self, Win_or_Lose_kihus, dim = 2, radius = 2 * np.sqrt(2), step = 10):
        ##initializing parameter
        self.radius = radius
        self.dim = dim
        self.step = step
        self.kihus = Win_or_Lose_kihus
        ##making filltration
        self.dim2_fill_list = []
        for kihu in self.kihus:
            remainder = len(kihu) % step
            turns = np.arange(step, len(kihu),step)
            if remainder >= step / 2:
                turns = np.append(turns, len(kihu))
            else:
                turns[-1] = len(kihu)
            self.dim2_fill_list.append([d.fill_rips(pdist(kihu[: t]), dim, radius) for t in turns])

    def random_choice_homology(self, choice_size = 1, replace_bool = False , root = None, plot = None, show = True):
        colors = ['r','g','b','c','m','y','k']
        if root == None:
            choice_index = np.random.choice(len(self.dim2_fill_list), size = choice_size, replace = replace_bool)
            dim2fill_list = [self.dim2_fill_list[i]  for i in choice_index]
            dim2fill_dgms_lists = []
            for count, dim2fill in enumerate(dim2fill_list):
                print(f'loading dim2_fill {count}')
                dim2fill_dgms = [d.init_diagrams(d.homology_persistence(fill), fill) for fill in dim2fill]
                dim2fill_dgms_lists.append(dim2fill_dgms)
            if show == True:
               for dim2fill_dgms in dim2fill_dgms_lists:
                   fig, axes = plt.subplots(4, 4, figsize = (12, 20))
                   one_dim_axes = axes.ravel()
                   for i, dgms in enumerate(dim2fill_dgms):
                       for k, dgm in enumerate(dgms):
                           style = {'color':colors[k]}
                           if dgms == dim2fill_dgms[-1]:
                               style['label'] = 'dim = ' + str(k)

                           try:
                               d.plot.plot_diagram(dgm,ax = one_dim_axes[i], pt_style = style)

                           except ValueError:
                               continue
                   fig.legend()
               plt.show()

        return dim2fill_dgms_lists


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
                for i, kihu in enumerate(self.kihus):
                    win_kihu = np.array([goishi[1] for goishi in kihu if goishi[0] == self.winners[i]])
                    lose_kihu = np.array([goishi[1] for goishi in kihu if goishi[0] != self.winners[i]])
                    self.Win_kihus.append(win_kihu)
                    self.Lose_kihus.append(lose_kihu)

    def kihu_random_plot(self,show_num = 1):
        kihus_len = len(self.Win_kihus)
        show_index = np.random.choice(np.arange(kihus_len), size = show_num)
        show_Win = [self.Win_kihus[i] for i in show_index]
        show_Lose = [self.Lose_kihus[i] for i in show_index]
        fig, axes = plt.subplots(show_num, 2, figsize = (12, 20))
        axes = axes.ravel()
        for i in range(show_num):
            point = show_Win[i].T
            axes[2 * i].scatter(point[0], point[1], color = 'r')
            point = show_Lose[i].T
            axes[2 * i + 1].scatter(point[0], point[1], color = 'b')
        plt.show()

def test():
    go = GO_data('../Desktop/NHK2006.zip')
    pers_win = kihu_persistence(go.Win_kihus)
    pers_win.random_choice_homology(choice_size = 10)

test()
