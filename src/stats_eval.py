import pstats
from pstats import SortKey

p = pstats.Stats('C:\\Users\\Kuba\\Desktop\\TEMP_Studia\\BIAI\\neural-network-backpropagation\\timing.txt')
p.strip_dirs().sort_stats(SortKey.TIME).print_stats(100)