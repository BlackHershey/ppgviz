import json
import matplotlib
matplotlib.use('Agg') # helps with webserver issues (https://stackoverflow.com/a/29172195)
import nibabel as nb
import numpy as np
import os
import os.path
import pandas as pd
import re
import sys

from datetime import datetime
from enum import Enum
from flask import Flask, render_template, request, send_file, send_from_directory
from glob import glob
from io import BytesIO
from matplotlib import pyplot as plt
from pet_aif_auc import pet_aif_auc

class DataType(Enum):
	FIG = 1
	TAC = 2
	AIF = 3
	TAC3D = 4
	TAC4D = 5

SUBJECT_ID = 'PET.Subject.Id'
VISIT_ID = 'PET.Visit.Id'
SESSION_DATE = 'Session.Date'

app = Flask(__name__)
app.config['PROJECT_FOLDER'] = '/data/nil-bluearc/raichle/PPGdata/jjlee2'

df = pd.read_excel('static/ppg_id_map.xlsx', sheet_name='id_map', usecols=[0,7,8,9])
df[SESSION_DATE] = pd.to_datetime(df[SESSION_DATE])
df = df.dropna().drop_duplicates()
df = df.loc[df.groupby([SUBJECT_ID, 'Condition'])[VISIT_ID].idxmin()]

subject_map = {}
for index, row in df.iterrows():
	if row[SUBJECT_ID] not in subject_map:
		subject_map[row[SUBJECT_ID]] = {}
	subject_map[row[SUBJECT_ID]][row['Condition']] = int(row[VISIT_ID])
pet_aif_auc(app.config['PROJECT_FOLDER'], subject_map) # recalculate auc on program start (in case any underlying CSVs have changed)


def dim_color(rgb_str):
	return rgb_str.replace(')', ', .3)')


# regenerate plotdata on each fig_report page load (needed to maintain "selected" subject styling)
def get_plotdata(subject_id=None):
	color_map = {
		'red': 'rgb(255, 99, 132)',
		'green': 'rgb(60, 179, 113)',
		'blue': 'rgb(30, 144, 255)'
	}
	auc_df = pd.read_csv('static/ppg_auc.csv')
	plotData = { 'datasets': [] }
	for condition, color in [('basal', 'blue'), ('hypergly', 'red'), ('hyperins', 'green')]:
		c_df = auc_df[auc_df.condition == condition]
		c_df = c_df.rename(columns={'aif': 'x', 'pet': 'y'})
		# hack: convert to json to get correct int serialization from df (to_dict will have numpy types that JS can't read)
		#   but then, we need to read the json back to dict so JS can properly deserialize it
		datapoints = json.loads(c_df.to_json(orient='records'))
		dataset = {
			'label': condition,
			'backgroundColor': color_map[color], # set overall backgroundColor to set up legend colors
			'pointBackgroundColor': [ color_map[color] if not subject_id or point['subject'] == subject_id else dim_color(color_map[color]) for point in datapoints ], #  dim points for non-selected subjects
			'fill': 'false',
			'data': datapoints
		}
		plotData['datasets'].append(dataset)
	return plotData


def get_filename(data_type, subject, condition):
	visit_id = subject_map[subject][condition]

	data_type = data_type.upper()
	if data_type == DataType.FIG.name:
		filename_template = '{}/Vall/fig_mlsiemens_Herscovitch1985_constructTracerState_fdgv{}r1/001.png'
	elif data_type == DataType.TAC.name:
		filename_template = '{0}/Vall/mlsiemens_Herscovitch1985_plotScannerWholebrain_fdgv{1}r1_fdgv{1}r1.csv'
	elif data_type == DataType.AIF.name:
		filename_template = '{0}/Vall/mlsiemens_Herscovitch1985_plotCaprac_fdgv{1}r1_fdgv{1}r1.csv'
	elif data_type == DataType.TAC3D.name:
		filename_template = '{0}/Vall/fdgv{1}r1_sumt.4dfp.img'
	elif data_type == DataType.TAC4D.name:
		filename_template = '{0}/Vall/fdgv{1}r1.4dfp.img'

	return filename_template.format(subject, visit_id) if filename_template else ''


def get_data_for_filetype(filename):
	full_path = os.path.join(app.config['PROJECT_FOLDER'], filename)
	if filename.endswith('.csv'):
		data = {}
		if os.path.exists(full_path):
			table_html = pd.read_csv(full_path).to_html(index=False, justify='left').replace('<table border="1" class="dataframe">','<table class="table table-striped table-scrollable">')
			data = { 'table_html': table_html, 'source': filename }
		return data
	else:
		return filename if os.path.exists(full_path) else None

def get_subject_data(data_type):
	rows = []
	for subject, condition_map in subject_map.items():
		row = {}
		row['subject'] = subject
		row['data'] = {}
		for condition in condition_map.keys():
			filename = get_filename(data_type, subject, condition)
			file_data = get_data_for_filetype(filename)
			row['data'][condition] = file_data
		row['data'] = dict(sorted(row['data'].items(), key=lambda x: x[0])) # sort conditions alphabetically ('basal', 'hygly', 'hyins')
		if any(row['data'].values()):
			rows.append(row)
	return rows


@app.route('/')
@app.route('/<subject_id>')
def fig_report(subject_id=None):
	rows = get_subject_data(DataType.FIG.name)
	return render_template('fig001.html', subject_id=subject_id, plotData=get_plotdata(subject_id), data=rows)


@app.route('/figdata')
@app.route('/<subject_id>/figdata')
def plot_report(subject_id=None):
	data = {}
	for metric in [ 'TAC', 'AIF' ]:
		data[metric] = get_subject_data(DataType[metric].name)
	return render_template('plot-tables.html', subject_id=subject_id, data=data)


@app.route('/figdata/tac', methods = ['POST', 'GET'])
@app.route('/<subject_id>/figdata/tac', methods = ['POST', 'GET'])
def tac_report(subject_id=None):
	slice = request.args.get('slice') if request.method == 'GET' else request.form.get('slice')
	frame = request.args.get('frame') if request.method == 'GET' else request.form.get('frame')
	data = {}
	for tab, img_type in [ ('time-integral', 'TAC3D'), ('time-series', 'TAC4D') ]: #'.gif') ]:
		data[tab] = get_subject_data(DataType[img_type].name)
	return render_template('view-images.html', subject_id=subject_id, data=data, slice=slice, frame=frame)


@app.route('/figdata/aif')
@app.route('/<subject_id>/figdata/aif')
def aif_report(subject_id=None):
	return render_template('view-pdfs.html', subject_id=subject_id, subject_list=sorted(list(subject_map.keys())))


@app.route('/file/<path:filename>')
def access_file(filename):
	return send_from_directory(app.config['PROJECT_FOLDER'], filename, as_attachment=True)


@app.route('/imshow/<path:filename>')
def access_slice(filename):
	imgdata = nb.load(os.path.join(app.config['PROJECT_FOLDER'], filename)).get_data()
	slice = int(request.args.get('slice')) if request.args.get('slice') else imgdata.shape[2] // 2
	frame = int(request.args.get('frame')) if request.args.get('frame') and imgdata.shape[3] > 1 else 0

	# cap values at max values to prevent indexing errors
	slice = min(slice, imgdata.shape[2])-1
	frame = min(frame, imgdata.shape[3])-1

	outimg = np.expand_dims(imgdata[:, :, slice, frame], axis=2)
	outimg = np.flip(outimg[:, :, -1], 1).T

	fig, ax = plt.subplots()
	ax.imshow(outimg, cmap='gray')
	ax.set_axis_off()
	img = BytesIO()
	fig.savefig(img)
	img.seek(0)
	plt.close()
	return send_file(img, mimetype='image/png')


@app.route('/pdfshow', methods=['GET'])
def show_pdf():
	subject = request.args.get('subject')
	condition = request.args.get('condition')
	visit_id = float(subject_map[subject][condition])

	visit_date = df.loc[(df[SUBJECT_ID] == subject) & (df[VISIT_ID] == visit_id)].iloc[0]['Session.Date']

	rad_files = glob(os.path.join(app.config['PROJECT_FOLDER'], 'CCIRRadMeasurements', 'CCIRRadMeasurements*.pdf'))
	filemap = { datetime.strptime(re.search('CCIRRadMeasurements (\w*).pdf', filename).group(1), '%Y%b%d'): filename for filename in rad_files }


	match = [ v for k,v in filemap.items() if pd.Timestamp(k) == visit_date ][0]
	return send_file(match, mimetype='application/pdf')


@app.route('/figdata/plot', methods=['GET'])
def gen_plot():
	data_type = request.args.get('data_type')
	subject = request.args.get('subject')
	condition = request.args.get('condition')

	fig, ax = plt.subplots()
	conditions = list(subject_map[subject].keys()) if condition == 'all' else [ condition ]
	conditions.sort()
	for condition in conditions:
		filename = os.path.join(app.config['PROJECT_FOLDER'], get_filename(data_type, subject, condition))
		if not os.path.exists(filename):
			continue
		x, y = np.genfromtxt(filename, delimiter=',', skip_header=1, unpack=True)
		ax.plot(x,y)

	ax.set_xlabel('Time (s)')
	ax.set_ylabel('Specific activity')
	plt.legend(conditions)

	plot = BytesIO()
	fig.savefig(plot)
	plot.seek(0)
	plt.close()
	return send_file(plot, mimetype='image/png')


if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0')
