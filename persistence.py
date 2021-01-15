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
        self.homology_count = 0
        ##making filltration
        self.winners_dim2fill = []
        self.losers_dim2fill = []
        for kihu in self.Win_kihus:
            turns = step_turns(kihu, step)
            self.winners_dim2fill.append([d.fill_rips(pdist(kihu[: t]), dim, radius) for t in turns])

        for kihu in self.Lose_kihus:
            turns = step_turns(kihu, step)
            self.losers_dim2fill.append([d.fill_rips(pdist(kihu[: t]), dim, radius) for t in turns])

    def random_choice_homology(self, choice_size = 1, replace = False ,compare = False, show = True, save = False, figname_head = None):
        random_indexes = np.random.choice(self.num_games, choice_size, replace = replace)
        return self.choice_homology(random_indexes, compare = compare, show = show, save = save, figname_head = figname_head)


    def choice_homology(self, indexes, compare = False, show = True, save = False, figname_head = None):
        persons_dim2filldgms_list = []
        persons = ['winner', 'loser']
        persons_dim2fill = [self.winners_dim2fill]
        if compare:
            persons_dim2fill.append(self.losers_dim2fill)
        for person, dim2fill_list in enumerate(persons_dim2fill):
            dim2filldgms_list = []
            compute_dim2fill_list = [dim2fill_list[i]  for i in indexes]
            for number, dim2fill in enumerate(compute_dim2fill_list):
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

def test(zip_path, choice_size, compare, show, save, figname_head, repeat):
    go = GO_data(zip_path)
    kp = kihu_persistence(go)
    for i in range(repeat):
        kp.random_choice_homology(choice_size = choice_size, compare = compare, show = show, save = save, figname_head = figname_head)
