import sys
import os

import numpy as np
list_paths = []
minlen = np.inf
for root, dirs, files in os.walk('F:\\workspace\\cacc_rl\\adversary_old_reward\\rear\\'):
	for file in files:
		filepath = os.path.join(root, file)
		with open(filepath, 'rb') as fp:
			data = pickle.load(fp)
			minlen = min(minlen, len(data))


			




