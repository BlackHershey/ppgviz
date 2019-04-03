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

DATATYPES = {
	"FIG": ("Combined Plots", None),
	"TAC": ("Data tables", "TAC"),
	"AIF": ("Data tables", "AIF"),
	"TAC3D": ("PET Images", "Time integral"),
	"TAC4D": ("PET Images", "Time series")
}

SUBJECT_ID = 'PET.Subject.Id'
VISIT_ID = 'PET.Visit.Id'
SESSION_DATE = 'Session.Date'

with open('data/ppg_gen.json') as f:
	data_config = json.load(f)

app = Flask(__name__)
app.config['PROJECT_FOLDER'] = data_config['project_folder']

df = pd.read_excel('static/ppg_id_map.xlsx', sheet_name='id_map', usecols=[0,7,8,9])
df[SESSION_DATE] = pd.to_datetime(df[SESSION_DATE])
df = df.dropna().drop_duplicates()
df = df.loc[df.groupby([SUBJECT_ID, 'Condition'])[VISIT_ID].idxmin()]

subject_map = {}
for index, row in df.iterrows():
	if row[SUBJECT_ID] not in subject_map:
		subject_map[row[SUBJECT_ID]] = {}
	subject_map[row[SUBJECT_ID]][row['Condition']] = int(row[VISIT_ID])


def dim_color(rgb_str):
	return rgb_str.replace(')', ', .3)')


# regenerate plotdata on each fig_report page load (needed to maintain "selected" subject styling)
def get_plotdata(tracer, subject_id=None):
	pet_aif_auc(app.config['PROJECT_FOLDER'], subject_map, tracer) # recalculate auc on program start (in case any underlying CSVs have changed)

	color_map = {
		'red': 'rgb(255, 99, 132)',
		'green': 'rgb(60, 179, 113)',
		'blue': 'rgb(30, 144, 255)'
	}
	auc_df = pd.read_csv('static/{}_auc.csv'.format(tracer))
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


def get_filename(data_type, subject, condition, tracer):
	visit_id = subject_map[subject][condition]

	id, tab = DATATYPES[data_type]
	templates = next(filepath['templates'] for filepath in data_config['filepaths'] if filepath['id'] == id and filepath['tab'] == tab)
	return [ template.format(subject=subject, session=visit_id, dataset=tracer) for template in templates ] if templates else []

def get_data_for_filetype(filenames):
	file_matches = [ item for filename in filenames for item in glob(os.path.join(app.config['PROJECT_FOLDER'], filename)) ]
	# full_path = os.path.join(app.config['PROJECT_FOLDER'], filename)
	# file_matches = glob(full_path)
	if all(f.endswith('.csv') for f in file_matches):
		data = []
		for f in file_matches:
			table_html = pd.read_csv(f).to_html(index=False, justify='left').replace('<table border="1" class="dataframe">','<table class="table table-striped table-scrollable">')
			data.append({ 'table_html': table_html, 'source': f })
		return data
	else:
		return [ os.path.relpath(f, app.config['PROJECT_FOLDER']) for f in file_matches ]

def get_subject_data(data_type, tracer):
	rows = []
	for subject, condition_map in subject_map.items():
		row = {}
		row['subject'] = subject
		row['data'] = {}
		for condition in condition_map.keys():
			filenames = get_filename(data_type, subject, condition, tracer)
			file_data = get_data_for_filetype(filenames)
			row['data'][condition] = file_data
		row['data'] = dict(sorted(row['data'].items(), key=lambda x: x[0])) # sort conditions alphabetically ('basal', 'hygly', 'hyins')
		if any(row['data'].values()):
			rows.append(row)
	return rows

@app.route('/')
@app.route('/<tracer>')
@app.route('/<tracer>/<subject_id>')
def fig_report(tracer='fdg', subject_id=None):
	rows = get_subject_data(DataType.FIG.name, tracer)
	return render_template('fig001.html', tracer=tracer, subject_id=subject_id, plotData=get_plotdata(tracer, subject_id), data=rows,
		max_length=max([len(v) for row in rows for v in row['data'].values() ]))


@app.route('/<tracer>/figdata')
@app.route('/<tracer>/<subject_id>/figdata')
def plot_report(tracer, subject_id=None):
	data = {}
	for metric in [ 'TAC', 'AIF' ]:
		data[metric] = get_subject_data(DataType[metric].name, tracer)
	return render_template('plot-tables.html', tracer=tracer, subject_id=subject_id, data=data,
		max_length=max([len(v) for row in data['TAC'] for v in row['data'].values() ]))


@app.route('/<tracer>/figdata/tac')
@app.route('/<tracer>/<subject_id>/figdata/tac')
def tac_report(tracer, subject_id=None):
	slice = request.args.get('slice')
	frame = request.args.get('frame')

	data = {}
	for tab, img_type in [ ('time-integral', 'TAC3D'), ('time-series', 'TAC4D') ]:
		data[tab] = get_subject_data(DataType[img_type].name, tracer)
		if img_type == 'TAC4D':
			sample_4dimg = os.path.join(app.config['PROJECT_FOLDER'], data[tab][0]['data']['basal'][0])
			max_slice, max_frame = nb.load(sample_4dimg).get_data().shape[2:]
	return render_template('view-images.html', tracer=tracer, subject_id=subject_id, data=data, max_slice=max_slice, max_frame=max_frame,
		slice=slice, frame=frame, max_length=max([len(v) for row in data['time-integral'] for v in row['data'].values() ]))


@app.route('/<tracer>/figdata/aif')
@app.route('/<tracer>/<subject_id>/figdata/aif')
def aif_report(tracer, subject_id=None):
	return render_template('view-pdfs.html', tracer=tracer, subject_id=subject_id, subject_list=sorted(list(subject_map.keys())))


@app.route('/file/<path:filename>')
def access_file(filename):
	return send_from_directory(app.config['PROJECT_FOLDER'], filename, as_attachment=True)


@app.route('/imshow/<path:filename>')
def access_slice(filename):
	imgdata = nb.load(os.path.join(app.config['PROJECT_FOLDER'], filename)).get_data()
	slice = int(request.args.get('slice')) if request.args.get('slice') else imgdata.shape[2] // 2
	frame = int(request.args.get('frame')) if request.args.get('frame') and imgdata.shape[3] > 1 else 1

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
	tracer = request.args.get('tracer')

	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax2 = plt.axes([0.65, 0.45, .2, .2])

	conditions = list(subject_map[subject].keys()) if condition == 'all' else [ condition ]
	conditions.sort()
	legend = []
	for condition in conditions:
		filenames = glob(os.path.join(app.config['PROJECT_FOLDER'], get_filename(data_type, subject, condition, tracer)))
		for filename in filenames:
			x, y = np.genfromtxt(filename, delimiter=',', skip_header=1, unpack=True)
			ax1.plot(x,y)
			ax2.plot(x[:14], y[:14]) # create zoomed in plot of first 120 seconds

			run = re.search('_([a-z]+(\d?))v\dr\d.csv$', filename).groups()[-1]
			legend.append(condition + ('-' + run if run else ''))

	ax1.set_title('{} {} for {}'.format(tracer.upper(), data_type.upper(), subject))
	ax1.set_xlabel('Time (s)')
	ax1.set_ylabel('Specific activity')
	ax1.legend(legend)

	if tracer != 'fdg':
		fig.delaxes(ax2) # only need zoomed in plot for long timecourses

	plot = BytesIO()
	fig.savefig(plot)
	plot.seek(0)
	plt.close()
	return send_file(plot, mimetype='image/png')


if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0')
