import zipfile as zf
import dionysus as d
from alpha import *
import numpy as np
import matplotlib.pyplot as plt
from sgfmill import sgf
from scipy.spatial.distance import pdist

class kihu_persistence:
    def __init__(self, GO_data, Maxdim = 2, Maxradius = 3 * np.sqrt(2), step = 'middle'):
        ##initializing parameter
        self.Maxradius = Maxradius
        self.Maxdim = Maxdim
        self.step = step
        self.Win_kihus = GO_data.Win_kihus
        self.Lose_kihus = GO_data.Lose_kihus
        self.num_games = GO_data.num_games
        self.dim2fill_dgms_lists = []
        self.homology_count = 0
        ##making filltration
        self.winners_dim2fill = []
        self.losers_dim2fill = []


    def random_choice_homology(self, choice_size = 1, replace = False ,compare = False, show = True, save = False, figname_head = None):
        random_indexes = np.random.choice(self.num_games, choice_size, replace = replace)
        return self.choice_homology(random_indexes, compare = compare, show = show, save = save, figname_head = figname_head)


    def choice_homology(self, indexes, compare = False, show = True, save = False, figname_head = None):
        persons_dim2filldgms_list = []
        persons = ['winner', 'loser']
        
        Win_kihus = [self.Win_kihus[i] for i in indexes]
        for kihu in Win_kihus:
            turns = step_turns(kihu, self.step)
            self.winners_dim2fill.append([fill_alpha(kihu[: t], self.Maxdim, self.Maxradius) for t in turns])
        persons_dim2fill = [self.winners_dim2fill]
        
        if compare:
            Lose_kihus = [self.Lose_kihus[i] for i in indexes]
            for kihu in Lose_kihus:
                turns = step_turns(kihu, self.step)
                self.losers_dim2fill.append([fill_alpha(kihu[: t], self.Maxdim, self.Maxradius) for t in turns])
            persons_dim2fill.append(self.losers_dim2fill)

        for person, dim2fill_list in enumerate(persons_dim2fill):
            dim2filldgms_list = []
            for number, dim2fill in enumerate(dim2fill_list):
                print(f'computing dim2 filtration diagrams of {persons[person]} {number}')
                dim2filldgms = [d.init_diagrams(d.homology_persistence(fill), fill) for fill in dim2fill]
                dim2filldgms_list.append(dim2filldgms)
            persons_dim2filldgms_list.append(dim2filldgms_list)

        if show or save:
            for person, dim2filldgms_list in enumerate(persons_dim2filldgms_list):
                title = f'{persons[person]} step:{self.step} count:{self.homology_count}'
                figname = None
                if save:
                    figname = f'{figname_head}{self.homology_count}_{persons[person]}'
                    print(f'saving {figname}')
                plot_dim2filldgms(dim2filldgms_list, title = title, show = (not compare) and show, save = save, figname = figname)
            if compare and show:
                plt.show()
            self.homology_count += 1

        return persons_dim2filldgms_list


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
            win_kihu = []
            lose_kihu = []
            for goishi in kihu:
                if win_kihu == None:
                    print('None')
                if (goishi[0] == self.winners[i]) and (goishi[1] not in win_kihu) :
                    win_kihu.append(goishi[1])
                    
                elif (goishi[0] != self.winners[i]) and (goishi[1] not in lose_kihu):
                    lose_kihu.append(goishi[1])
                
            self.Win_kihus.append(np.array(win_kihu, dtype = float))
            self.Lose_kihus.append(np.array(lose_kihu, dtype = float))

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
    if step == 'middle':
        turns = [len(kihu) // 2, len(kihu)]
        
    else:
        remainder = len(kihu) % step
        turns = np.arange(step, len(kihu),step)
        if remainder >= step / 2:
            turns = np.append(turns, len(kihu))
        else:
            turns[-1] = len(kihu)

    return turns

def plot_dim2filldgms(dim2filldgms_list, title = None, show = True, save = False, figname = None):
    colors = ['r','g','b','c','m','y','k']
    for number, dim2filldgms in enumerate(dim2filldgms_list):
        fig, axes = plt.subplots(4, 4, figsize = (12, 20))
        one_dim_axes = axes.ravel()
        for i, dgms in enumerate(dim2filldgms):
            for k, dgm in enumerate(dgms):
                color = colors[k]
                label = None
                style = {'color':color}
                if dgms == dim2filldgms[-1]:
                    style['label'] = f'dim = {k}'
                    label = f'hist:dim = {k}, death = inf'
                ax = one_dim_axes[i]
                try:
                    d.plot.plot_diagram(dgm,ax = ax, pt_style = style)

                except ValueError:
                    if len(dgm):
                       birth_list = [hole.birth for hole in dgm if hole.death == float('inf')]
                       bins_list = []
                       for birth in birth_list:
                           if birth not in bins_list:
                               bins_list.append(birth)
                       ax.hist(birth_list, bins = len(bins_list), density = True, color = color, label = label)
                    continue

        fig.legend()
        fig.suptitle(f'{title} dim2 filtraion diagrams{number}')
        if save:
            plt.savefig(f'{figname}{number}')
    if show:
        plt.show()

def main(zip_path, figname_head = None, choice_size = 5, compare = True, show = False, save = True, repeat = 1):
    go = GO_data(zip_path)
    print(type(go.Win_kihus[1]))

