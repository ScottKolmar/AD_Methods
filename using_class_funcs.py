from src.class_funcs import *
import pprint as pp

g298atom_csv = r'C:\Users\skolmar\PycharmProjects\Modeling\AD_Methods\Datasets\g298atom_desc.csv'
dataset = DataSet(g298atom_csv, 1000)
dataset.split(test_size=0.2, shuffle=True, random_state=11)
dataset.get_neighbors(n_neighbors=5, metric='euclidean')
dataset.calc_average_cosine_similarity()
dataset.calc_kNN_error_corr(n_neighbors=5, metric='euclidean')
dataset.calculate_stats(algorithm = 'kNN', ad_measure = 'average_cosine_similarities')
out_path = r'C:\Users\skolmar\PycharmProjects\Modeling\AD_Methods\PNG\G298atom'
dataset.plot_correlations(algorithm='kNN', ad_measure = 'average_cosine_similarities', out_path=out_path)
