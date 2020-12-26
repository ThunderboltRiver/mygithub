import zipfile as zf
import dionysus as d
import numpy as np
from sgfmill import sgf
from scipy.spatial.distance import pdist

def read_sgf_from_zip(zip_file, number):
    winners = []
    board_size = []
    root_nodes = []
    dic_nodes = {}
    b_players = []
    w_players = []
    with zf.ZipFile(zip_file) as zip_data:
        infos = [info for info in zip_data.infolist() if '.sgf' in info.filename]
        for info in infos:
            with zip_data.open(info.filename, 'r') as f:
            	game = sgf.Sgf_game.from_bytes(f.read())
            winners.append(game.get_winner())
            board_size.append(game.get_size())
            root_nodes.append(game.get_root())
            b_players.append(root_nodes[-1].get('PB'))
            w_players.append(root_nodes[-1].get('PW'))
            file_name = info.filename.encode('cp437').decode('utf-8')
            dic_nodes[file_name] = [node.get_move() for node in game.get_main_sequence() if node.get_move() != (None, None) ]
        if number == None:
            return {'win':winners,'size':board_size,'kihu':dic_nodes,'b':b_players,'w':w_players}

        else:
            return {'win':winners[number],'size':board_size[number],'kihu':dic_nodes[file_list[i]],'b':b_players[number],'w':w_players[number]}

