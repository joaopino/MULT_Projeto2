import numpy as np
import os.path
import librosa
import librosa.feature.rhythm as rhythm
import scipy as sp

def extract_features():
    data = np.genfromtxt("Features/top100_features.csv", delimiter = ",", skip_header = 1)
    return data[:, 1:-1]

def normalize_features(data):
    max = data.max(axis = 0)
    min = data.min(axis = 0)
    return ((data - min)/(max - min))

def save_to_file(path, data):
    np.savetxt(path, data, delimiter=",", fmt="%.6f")
    
def extract_music_features(music):
    data, sampling_rate = librosa.load("Musica/" + music)

    mfcc = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc = 13)
    centroid = librosa.feature.spectral_centroid(y=data, sr=sampling_rate)
    bandwidth = librosa.feature.spectral_bandwidth(y=data, sr=sampling_rate)
    contrast = librosa.feature.spectral_contrast(y=data, sr=sampling_rate)
    flatness = librosa.feature.spectral_flatness(y=data)
    rolloff = librosa.feature.spectral_rolloff(y = data, sr = sampling_rate)
    f0 = librosa.yin(y=data, fmin=20, fmax=11025)
    f0[f0 == 11025] = 0
    rms = librosa.feature.rms(y=data)
    crossing_rate = librosa.feature.zero_crossing_rate(y=data)
    tempo = rhythm.tempo(y=data, sr=sampling_rate)
        
    return mfcc, centroid, bandwidth, contrast, flatness, rolloff, f0, rms, crossing_rate, tempo

def calculate_statistics(feature, axis_):
    mean = feature.mean(axis = axis_)
    std = feature.std(axis = axis_)
    skewness = sp.stats.skew(feature, axis = axis_)
    kurtosis = sp.stats.kurtosis(feature, axis = axis_)
    median = np.median(feature, axis = axis_)
    max = feature.max(axis = axis_)
    min = feature.min(axis = axis_)
    
    if(axis_ == 1):
        result = np.array([])
        for i in range(len(mean)):
            result = np.hstack((result, mean[i], std[i], skewness[i], kurtosis[i], median[i], max[i], min[i]))
        return result
    else:
        return np.hstack(([mean], [std], [skewness], [kurtosis], [median], [max], [min]))

def euclidean(a,b):
    distance = np.linalg.norm(a-b)
    return distance, distance

def manhattan(a, b):
    distance = sp.spatial.distance.cityblock(a, b)
    return distance, distance

def cosine(a, b):
    distance = sp.spatial.distance.cosine(a, b)
    return distance, distance

def receive_distances(filename, data, type):
    if not os.path.exists(filename):
        data[data != data] = 0
        data = np.empty((len(data), len(data)))
        for i in range (0, len(data)):
            for j in range (i, len(data)):
                if i == j:
                    data[i][j] = 0
                elif type == "Euclidean":
                    data[i][j], data[j][i] = euclidean(data[i], data[j])
                elif type == "Manhattan":
                    data[i][j], data[j][i] = manhattan(data[i], data[j])
                elif type == "Cosine":
                    data[i][j], data[j][i] = cosine(data[i], data[j])
        save_to_file(filename, data)
        return data
    else:
        return np.genfromtxt(filename, delimiter = ",")  
    
def print_best_rankings(data, indexes):
    rankings = np.array([])
    for i in range(1,21):
        print(f"\t{i}º - {data[indexes[i]]}")
        rankings = np.concatenate((rankings, data[indexes[i]]), axis = None)
    return rankings 

def metadata_extraction():
    data = np.genfromtxt("Dataset/panda_dataset_taffc_metadata.csv", delimiter = ",", dtype = str, skip_header = 1)
    dic = {}
    for line in data:
        genres = line[11][1:-1].split("; ")
        emotions = line[9][1:-1].split("; ")
        
        dic[line[0][1:-1]] = [line[1][1:-1], genres, line[3][1:-1], emotions]
    return dic

def calculate_music_quality(queries, dic, path):
    qualities = {}
    queries_qualities = np.empty((4, 900))
    i = 0
    for querie in queries:
        querie = querie.replace(".mp3", "")    
        j = 0
        for key, value in dic.items():  
            sum_ = 0
            if (key == querie):
                sum_ = -1
            else:
                sum_ += (dic[querie][0] == value[0])
                sum_ += len(list(set(dic[querie][1]).intersection(value[1])))                
                sum_ += (dic[querie][2] == value[2])                
                sum_ += len(list(set(dic[querie][3]).intersection(value[3])))
            queries_qualities[i][j] = sum_
            j += 1
        qualities[querie] = np.argsort(queries_qualities[i])[::-1]
        i += 1
    save_to_file(path, queries_qualities)
    return qualities

def print_quality_rankings(number_of_musics, music_folder, qualities_dic):
    array = np.empty((number_of_musics), dtype = str)
    print("\n--------------- Exercice 4.1.2. ------------------")
    for key, index in qualities_dic.items():
        print(f"Ranking best {number_of_musics} songs for {key}")
        
        a = np.array([])
        for i in range(number_of_musics):
            print(f"{i+1}º - {music_folder[index[i]]}")  
            a = np.concatenate((a, [music_folder[index[i]]]))          
        array = np.vstack((array, a))
        print()
    return array[1:]

def calculate_precision(number_of_musics, queries_folder, rankings_distance, rankings_metadata):
    print("--------------- Exercice 4.1.3. ------------------")
    for i in range(len(rankings_metadata)):
        precision_euclidean_data = len(list(set(rankings_metadata[i]).intersection(rankings_distance[i*6]))) / number_of_musics
        precision_euclidean_librosa = len(list(set(rankings_metadata[i]).intersection(rankings_distance[i*6 + 1]))) / number_of_musics
        precision_manhattan_data = len(list(set(rankings_metadata[i]).intersection(rankings_distance[i*6 + 2]))) / number_of_musics
        precision_manhattan_librosa = len(list(set(rankings_metadata[i]).intersection(rankings_distance[i*6 + 3]))) / number_of_musics
        precision_cosine_data = len(list(set(rankings_metadata[i]).intersection(rankings_distance[i*6 + 4]))) / number_of_musics
        precision_cosine_librosa = len(list(set(rankings_metadata[i]).intersection(rankings_distance[i*6 + 5]))) / number_of_musics
        
        print(f"--- PRECISIONS ({queries_folder[i]}) ---")
        print(f"Euclidean data: {precision_euclidean_data}")
        print(f"Euclidean librosa: {precision_euclidean_librosa}")
        print(f"Manhattan data: {precision_manhattan_data}")
        print(f"Manhattan librosa: {precision_manhattan_librosa}")
        print(f"Cosine data: {precision_cosine_data}")
        print(f"Cosine librosa: {precision_cosine_librosa}")
        print("----------------------------\n")

if __name__ == "__main__" :
    saved_features = "top100_features_normalized.csv"
    librosa_features = "librosa_features_normalized.csv"
    music_folder = os.listdir("Musica")
    queries_folder = os.listdir("Queries")
    
    # Ex 2.1
    if os.path.exists(saved_features):
        data_normalized = np.genfromtxt(saved_features, delimiter = ",")
    else:
        # Ex 2.1.1
        data_extracted = extract_features()
        # Ex 2.1.2
        data_normalized = normalize_features(data_extracted)
        # Ex 2.1.3
        save_to_file(saved_features, data_normalized)
    
    # Ex 2.2    
    if os.path.exists(librosa_features):
        librosa_data_normalized = np.genfromtxt(librosa_features, delimiter = ",")
    else:    
        statistics = np.empty((0, 190))
        for music in music_folder:
            mfcc, spectral_centroid, spectral_bandwidth, spectral_contrast, spectral_flatness, spectral_rolloff, f0, rms, zero_crossing_rate, tempo = extract_music_features(music)
            mfcc_stats = calculate_statistics(mfcc, 1)
            spectral_centroid_stats = calculate_statistics(spectral_centroid, 1)
            spectral_bandwidth_stats = calculate_statistics(spectral_bandwidth, 1)
            spectral_contrast_stats = calculate_statistics(spectral_contrast, 1)
            spectral_flatness_stats = calculate_statistics(spectral_flatness, 1)
            spectral_rolloff_stats = calculate_statistics(spectral_rolloff, 1)
            f0_stats = calculate_statistics(f0, 0)
            rms_stats = calculate_statistics(rms, 1)
            crossing_rate_stats = calculate_statistics(zero_crossing_rate, 1)
            music_stats = np.concatenate((mfcc_stats.flatten(), spectral_centroid_stats, spectral_bandwidth_stats, spectral_contrast_stats.flatten(), spectral_flatness_stats, spectral_rolloff_stats, f0_stats,  rms_stats, crossing_rate_stats, tempo))
            stats = np.vstack((statistics, music_stats))
    
        # Ex 2.2.3
        librosa_data_normalized = normalize_features(statistics)
        # Ex 2.2.4
        save_to_file(librosa_features, librosa_data_normalized)
    
    fich_names = ["data_euclidean_distance.csv", "librosa_euclidean_distance.csv","data_manhattan_distance.csv", "librosa_manhattan_distance.csv", "data_cosine_distance.csv", "librosa_cosine_distance.csv"]

    # Ex 3.1.1
    data_euclidean_distance = receive_distances(fich_names[0], data_normalized, "Euclidean")
    librosa_euclidean_distance = receive_distances(fich_names[1], librosa_data_normalized, "Euclidean")
    
    # Ex 3.1.2
    data_manhattan_distance = receive_distances(fich_names[2], data_normalized, "Manhattan")
    librosa_manhattan_distance = receive_distances(fich_names[3], librosa_data_normalized, "Manhattan")
    
    # Ex 3.1.3
    data_cosine_distance = receive_distances(fich_names[4], data_normalized, "Cosine")
    librosa_cosine_distance = receive_distances(fich_names[5], librosa_data_normalized, "Cosine")


    # Ex 3.3
    
    rankings_distance = np.empty((20), dtype = str)
    print("--------------- Exercice 3.3. ------------------")
    for i in range(len(queries_folder)):
        print(f">>>>> Song {queries_folder[i]} <<<<<")
        print("Ranking with data euclidean distance")
        indice = music_folder.index(queries_folder[i])

        indexes = np.argsort(data_euclidean_distance[indice])
        rankings_distance = np.vstack((rankings_distance, print_best_rankings(music_folder, indexes)))
        
        print("\nRanking with librosa euclidean distance")
        indexes = np.argsort(librosa_euclidean_distance[indice])
        rankings_distance = np.vstack((rankings_distance, print_best_rankings(music_folder, indexes)))
        
        print("\nRanking with data manhattan distance")
        indexes = np.argsort(data_manhattan_distance[indice])
        rankings_distance = np.vstack((rankings_distance, print_best_rankings(music_folder, indexes)))
        
        print("\nRanking with librosa manhattan distance")
        indexes = np.argsort(librosa_manhattan_distance[indice])
        rankings_distance = np.vstack((rankings_distance, print_best_rankings(music_folder, indexes)))
        
        print("\nRanking with data cosine distance")
        indexes = np.argsort(data_cosine_distance[indice])
        rankings_distance = np.vstack((rankings_distance, print_best_rankings(music_folder, indexes)))
                
        print("\nRanking with librosa cosine distance")
        indexes = np.argsort(librosa_cosine_distance[indice])
        rankings_distance = np.vstack((rankings_distance, print_best_rankings(music_folder, indexes)))
        print()

    rankings_distance = rankings_distance[1:]
    print(rankings_distance)

    # Ex 4.1.1
    qualities_dic = {}
    os.path.exists("metadataSimilarity.csv")
    data = np.genfromtxt("metadataSimilarity.csv", delimiter = ",")    
    for i in range(len(queries_folder)):
        qualities_dic[queries_folder[i]] = np.argsort(data[i])[::-1]
    dic = metadata_extraction()
    qualities_dic = calculate_music_quality(queries_folder, dic, "metadataSimilarity.csv")
        
    # Ex 4.1.2
    rankings_metadata = print_quality_rankings(20, music_folder, qualities_dic)

    # Ex 4.1.3
    calculate_precision(20, queries_folder, rankings_distance, rankings_metadata)