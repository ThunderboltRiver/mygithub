import zipfile as zf
import dionysus as d
import numpy as np
import matplotlib.pyplot as plt
from sgfmill import sgf
from scipy.spatial.distance import pdist

class kihu_persistence:
    def __init__(self, GO_data, dim = 2, radius = 2 * np.sqrt(2), step = 10):
        ##initializing parameter
        self.radius = radius
        self.dim = dim
        self.step = step
        self.Win_kihus = GO_data.Win_kihus
        self.Lose_kihus = GO_data.Lose_kihus
        self.num_games = GO_data.num_games
        self.dim2fill_dgms_lists = []
        ##making filltration
        self.winners_dim2fill = []
        self.losers_dim2fill = []
        for kihu in self.Win_kihus:
            turns = step_turns(kihu, step)
            self.winners_dim2fill.append([d.fill_rips(pdist(kihu[: t]), dim, radius) for t in turns])

        for kihu in self.Lose_kihus:
            turns = step_turns(kihu, step)
            self.losers_dim2fill.append([d.fill_rips(pdist(kihu[: t]), dim, radius) for t in turns])

    def random_choice_homology(self, choice_size = 1, replace = False ,compare = False, show = True):
        random_indexes = np.random.choice(self.num_games, choice_size, replace = replace)
        return self.choice_homology(random_indexes, compare = compare, show = show)


    def choice_homology(self, indexes, compare = False, show = True):
        out = []
        persons_dim2fill = [self.winners_dim2fill]
        if compare == True:
            persons_dim2fill.append(self.losers_dim2fill)
        for dim2fill_list in persons_dim2fill:
            dim2fill_dgms_lists = []
            dim2fill_list = [dim2fill_list[i]  for i in indexes]
            for count, dim2fill in enumerate(dim2fill_list):
                print(f'loading dim2_fill {count}')
                dim2fill_dgms = [d.init_diagrams(d.homology_persistence(fill), fill) for fill in dim2fill]
                dim2fill_dgms_lists.append(dim2fill_dgms)
            out.append(dim2fill_dgms_lists)

        if show == True:
           people_list = ['winner', 'loser']
           for i, dim2fill_dgms_lists in enumerate(out):
               plot_dim2fill_dgms(dim2fill_dgms_lists, title = f'{people_list[i]} step = {self.step} ', show = False)
           plt.show()

        return out


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

        self.num_games = len(self.b_players)

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


def step_turns(kihu, step):
    remainder = len(kihu) % step
    turns = np.arange(step, len(kihu),step)
    if remainder >= step / 2:
        turns = np.append(turns, len(kihu))
    else:
        turns[-1] = len(kihu)

    return turns

def plot_dim2fill_dgms(dim2fill_dgms_lists, title = None, show = True):
    colors = ['r','g','b','c','m','y','k']
    for number, dim2fill_dgms in enumerate(dim2fill_dgms_lists):
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
        fig.suptitle(f'number = {number} {title}')
    if show == True:
        plt.show()

def test(zip_path, choice_size, compare, show):
    go = GO_data(zip_path)
    kp = kihu_persistence(go)
    kp.random_choice_homology(choice_size = choice_size, compare = compare, show)
