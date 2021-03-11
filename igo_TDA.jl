module kihu_data_geometry
using PyCall
using LinearAlgebra
using Distributed
using ZipFile
@pyimport sgfmill.sgf as sgf

struct data_set
    winners
    board_size
    kihus
    function data_set(dir::String)
    winners = []
    board_size = []
    kihus = []
    function read_sgf(sgf_file)
        game = sgf.Sgf_game.from_bytes(read(sgf_file))
        push!(winners, game.get_winner())
        push!(board_size, game.get_size())
        root_node = game.get_root
        push!(kihus, [node.get_move() for node in game.get_main_sequence() if node.get_move() != (nothing, nothing)])
    end
    
    function read_zip(zip_file::String)
        zip_file = joinpath(dir, zip_file)
        zip_data = ZipFile.Reader(zip_file);
        for f in zip_data.files
            if occursin("sgf", f.name)
                read_sgf(f)
            end
        end
        close(zip_data)
    end
    
    function read_dir(dir_path::String)
        elements = readdir(dir_path)
        for child in elements
            child = joinpath(dir_path, child)
            if occursin(".zip", child)
                read_zip(child)
                
            elseif occursin(".sgf", child)
                read_sgf(child)
                
            elseif isdir(child)
                read_dir(child)
            end
        end
    end
    read_dir(dir)
    new(winners, board_size, kihus)
    end
end

function two_combination(list_like)
    generater = ((list_like[i], list_like[j]) for i in 1: length(list_like) - 1 for j in i + 1: length(list_like))
end

struct geometry
    _kihu
    _N
    _goishi_colors
    _winner
    _dist_matrix

    function geometry(kihu, winner)
       
        function kihu_sort()
            sorted_kihu = []
            goishi_colors = []
            for (i, goishi) in enumerate(kihu)
                push!(goishi_colors, goishi[1])
                Goishi = [i]
                for j in goishi[2]
                    push!(Goishi, j)
                end
                push!(sorted_kihu, Goishi)
            end

            return (sorted_kihu, goishi_colors)
        end
        
        (_kihu, _goishi_colors) = kihu_sort()

        function dist(goishi1, goishi2)
            LinearAlgebra.norm(goishi1 - goishi2)
        end

        function dist_matrix()
            _dist_matrix = [dist(g1, g2) for g1 in _kihu, g2 in _kihu]
        end


        function normalize_dist_matirx()
            matrix = dist_matrix();
            return matrix / maximum(matrix)
        end
        
        
        _N = length(_kihu)
        _winner = winner
        _dist_matrix = normalize_dist_matirx()
        new(_kihu, _N, _goishi_colors, _winner, _dist_matrix);

    end
end
end

module kihu_homology
    using Plots
    using Ripserer
    using Distributed
    using PersistenceDiagrams
    using ..kihu_data_geometry
    players = ["w", "b"]

    function network_diagrams(kihu_geometry::kihu_data_geometry.geometry, param, dim)
        kc = kihu_geometry
        kihu = kc._kihu
        winner = kc._winner
        loser = setdiff(players, [winner])[1]

        function color(goishi)
            kc._goishi_colors[goishi[1]]
        end

        function coef(player, goishi1, goishi2)
            (c1, c2) = (color(goishi1), color(goishi2))
            if c1 == c2 == player
                param

            elseif c1 != c2
                1.0 - param

            else
                0.0

            end
        end

        function coef_matrix(player)
            matrix = [coef(player, g1, g2) for g1 in kihu, g2 in kihu]
        end
                
        function adjacency(player)
            coef_mat = coef_matrix(player);
            dist_mat = kc._dist_matrix;
            adj_mat = 1 .- coef_mat + (coef_mat .* dist_mat)
            adj_mat -= LinearAlgebra.Diagonal(adj_mat)
            return adj_mat
        end

        function persistent_diagrams(player)
            adj_mat = adjacency(player);
            dgms = Ripserer.ripserer(adj_mat, dim_max = dim, threshold = 0.99999);
            return dgms
        end
                
        dgms_winner = persistent_diagrams(winner);
        dgms_loser = persistent_diagrams(loser);
        dgms_dict = Dict("winner" => dgms_winner, "loser" => dgms_loser, "lambda" => param)
    end
    
    function alpha_diagrams(kihu_geometry)
        kc = kihu_geometry
        
        kihu = kc._kihu
        winner = kc._winner
        loser = setdiff(players, [winner])[1]

        function color(goishi)
            kc._goishi_colors[goishi[1]]
        end

        function player_kihu(player)
            out = []
            for goishi in kihu
                if color(goishi) == player
                    if goishi[1] in [1, 2]
                        push!(out, (1, goishi[2], goishi[3]))

                    else
                        push!(out, (goishi[1] - 1, goishi[2], goishi[3]))

                    end
                end
            end
            return out
        end

        winner_alpha = Ripserer.Alpha(player_kihu(winner));
        loser_alpha = Ripserer.Alpha(player_kihu(loser));
        winner_dgms = Ripserer.ripserer(winner_alpha, dim_max = 2);
        loser_dgms = Ripserer.ripserer(loser_alpha, dim_max = 2);
        dgms_dict = Dict("winner" => winner_dgms, "loser" => loser_dgms)
    end

    function dgms_distance(dgms1, dgms2)
        PersistenceDiagrams.Bottleneck()(dgms1[2], dgms2[2])
    end

    function dgms_plot(dgms)
        Plots.plot(Plots.plot(dgms), Ripserer.barcode(dgms))
    end
    
    function empty_dgms(len)
        PD = PersistenceDiagrams
        out = [PD.PersistenceDiagram(PD.PersistenceInterval[], dim = i - 1) for i in 1:len]
    end


    function summary_alpha_diagrams(data_set::kihu_data_geometry.data_set)
        summary_dict = Dict("winner" => kihu_homology.empty_dgms(3), "loser" => kihu_homology.empty_dgms(3))
        function sum_dgms(dgms_dict, player)
            player_sum = summary_dict[player]
            player_dgms = dgms_dict[player]
            for (i, dgm) in enumerate(player_sum)
                for interval in player_dgms[i].intervals
                    push!(dgm.intervals, interval)
                    sort(dgm; by = persistence)
                end
            end
        end
    
        for (i, kihu) in enumerate(data_set.kihus)
            winner = data_set.winners[i]
            kihu_geometry = kihu_data_geometry.geometry(kihu, winner)
            dgms_dict = kihu_homology.alpha_diagrams(kihu_geometry)
            sum_dgms(dgms_dict, "winner")
            sum_dgms(dgms_dict, "loser")
        end
        return summary_dict
    end
end



