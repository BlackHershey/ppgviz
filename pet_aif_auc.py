import csv
import numpy as np
import os.path
import pandas as pd
import scipy.integrate

def calc_auc(filename, integration_bounds):
	data = []
	with open(filename) as f:
		 reader = csv.reader(f)
		 next(reader)
		 for line in reader:
			 data.append(line)

	data = [ l for l in data if float(l[0]) >= integration_bounds[0] and float(l[0]) <= integration_bounds[1] ]

	x, y = zip(*data)
	x = [ float(z) for z in x ]
	y = [ float(z) for z in y ]
	return scipy.integrate.trapz(y, x)


def pet_aif_auc(project_dir, subject_map):
	results = []
	for subject, mapping in subject_map.items():
		for condition, visit_id in mapping.items():
			aiffile = '{0}/{1}/Vall/mlsiemens_Herscovitch1985_plotCaprac_fdgv{2}r1_fdgv{2}r1.csv'.format(project_dir, subject, visit_id)
			aif_auc = calc_auc(aiffile, (0,3000)) if os.path.exists(aiffile) else np.nan

			petfile = '{0}/{1}/Vall/mlsiemens_Herscovitch1985_plotScannerWholebrain_fdgv{2}r1_fdgv{2}r1.csv'.format(project_dir, subject, visit_id)
			pet_auc = calc_auc(petfile, (2400, 3600)) if os.path.exists(petfile) else np.nan

			results.append([subject, condition, aif_auc, pet_auc])

	with open('static/ppg_auc.csv', 'w', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['subject', 'condition', 'x', 'y'])
		writer.writerows(results)
