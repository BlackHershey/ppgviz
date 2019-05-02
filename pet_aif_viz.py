import argparse
import itertools
import json
import matplotlib
matplotlib.use('Agg') # helps with webserver issues (https://stackoverflow.com/a/29172195)
import nibabel as nb
import numpy as np
import os
import os.path
import pandas as pd
import re

from datetime import datetime
from enum import Enum
from flask import Flask, abort, redirect, render_template, request, send_file, send_from_directory, url_for
from glob import glob
from io import BytesIO
from matplotlib import cm
from matplotlib import pyplot as plt

def create_app(data_config_file, subject_map_file, view_config_file):
	class DataType(Enum):
		FIG = 1
		TABLE = 2
		IMG = 3
		GRAPH = 4

	with open(data_config_file) as f:
		data_config = json.load(f)

	with open(view_config_file) as f:
		view_config = json.load(f)

	default_ds = data_config['datasets'][0] # TODO: do we want to allow no dataset specification?


	app = Flask(__name__)
	app.config['PROJECT_FOLDER'] = data_config['project_folder']

	df = pd.read_csv(subject_map_file).dropna().drop_duplicates()
	# df = df.loc[df.groupby([SUBJECT_ID, 'Condition'])[VISIT_ID].idxmin()]

	SUBJECT_ID = 'subject'
	VISIT_ID = 'session'
	CONDITION = 'label'

	subject_map = {}
	for index, row in df.iterrows():
		if row[SUBJECT_ID] not in subject_map:
			subject_map[row[SUBJECT_ID]] = {}
		subject_map[row[SUBJECT_ID]][row[CONDITION]] = int(row[VISIT_ID])


	def dim_color(rgb_str):
		return rgb_str.replace(')', ', .3)')


	def get_filename(view_id, tab, subject, condition, tracer):
		visit_id = subject_map[subject][condition] if subject and condition else None

		templates = next((filepath['templates'] for filepath in data_config['filepaths'] if filepath['id'].lower() == view_id.lower() \
		 	and (filepath['tab'] is None or filepath['tab'].lower() == tab.lower())), None)
		return [ template.replace('{subject}', str(subject)).replace('{session}', str(visit_id)).replace('{dataset}', str(tracer)) for template in templates ] if templates else []


	def get_data_for_filetype(filenames, data_type):
		file_matches = [ item for filename in filenames for item in glob(os.path.join(app.config['PROJECT_FOLDER'], filename)) ]
		if data_type.upper() == DataType.TABLE.name:
			data = []
			for f in file_matches:
				table_html = pd.read_csv(f).to_html(index=False, justify='left').replace('<table border="1" class="dataframe">','<table class="table table-striped table-scrollable">')
				data.append({ 'table_html': table_html, 'source': f })
			return data
		else:
			return [ os.path.relpath(f, app.config['PROJECT_FOLDER']) for f in file_matches ]


	def get_subject_data(view_id, tab, tracer):
		rows = []
		for subject, condition_map in subject_map.items():
			row = {}
			row['subject'] = subject
			row['data'] = {}
			for condition in condition_map.keys():
				filenames = get_filename(view_id, tab, subject, condition, tracer)
				file_data = get_data_for_filetype(filenames, view_config[view_id]['type'])
				row['data'][condition] = file_data
			row['data'] = dict(sorted(row['data'].items(), key=lambda x: x[0])) # sort conditions alphabetically ('basal', 'hygly', 'hyins')
			if any(row['data'].values()):
				rows.append(row)
		return rows


	def get_maxlength(data, tabname):
		return max([len(v) for row in data[tabname] for v in row['data'].values() ], default=None)


	@app.context_processor
	def inject_config():
	    return dict(view_config=view_config, datasets=data_config['datasets'])


	@app.route('/')
	def index(tracer=default_ds, subject=None):
		rows = None
		for id, conf in view_config.items():
			print(id)
			for tab in conf['tabs']:
				rows = get_subject_data(id, tab['name'], tracer) # find first datatype that has data (later on this will need to be view order dependent -- wont be able to rely on enum)
				if rows:
					type = conf['type'].lower()
					return redirect(url_for('{}_report'.format(type), subject=subject, tracer=tracer, view_id=id))

		 # if no return statement by this point, rows is empty -- abort
		abort(400)


	@app.route('/<tracer>/graph/<view_id>')
	@app.route('/<tracer>/graph/<view_id>/<subject_id>')
	def graph_report(tracer, view_id, subject_id=None):
		data = {}
		colors = [ tuple([x*255 for x in tup]) for tup in cm.tab10.colors ]

		for tab in view_config[view_id]['tabs']:
			data[tab['name']] = { 'datasets': [] }

			filename = [ item for filename in get_filename(view_id, tab, None, None, tracer) for item in glob(os.path.join(app.config['PROJECT_FOLDER'], filename)) ][0]
			graph_df = pd.read_csv(filename).dropna()
			conditions = np.unique(graph_df['condition'].values)

			for condition, color_tup in zip(conditions, colors):
				c_df = graph_df[graph_df.condition == condition]
				c_df = c_df.rename(columns={'aif': 'x', 'pet': 'y'})

				color_str = 'rgb{}'.format(color_tup)

				# hack: convert to json to get correct int serialization from df (to_dict will have numpy types that JS can't read)
				#   but then, we need to read the json back to dict so JS can properly deserialize it
				datapoints = json.loads(c_df.to_json(orient='records'))
				dataset = {
					'label': condition,
					'backgroundColor': color_str, # set overall backgroundColor to set up legend colors
					'pointBackgroundColor': [ color_str if not subject_id or point['subject'] == subject_id else dim_color(color_str) for point in datapoints ], #  dim points for non-selected subjects
					'fill': 'false',
					'data': datapoints
				}
				data[tab['name']]['datasets'].append(dataset)

		return render_template('graph_report.html', tracer=tracer, view_id=view_id, subject_id=subject_id, data=data)


	@app.route('/<tracer>/fig/<view_id>')
	@app.route('/<tracer>/fig/<view_id>/<subject_id>')
	def fig_report(tracer, view_id, subject_id=None):
		data = {}
		for tab in view_config[view_id]['tabs']:
			data[tab['name']] = get_subject_data(view_id, tab['name'], tracer)

		return render_template('fig_report.html', tracer=tracer, view_id=view_id, subject_id=subject_id,
			data=data, max_length=get_maxlength(data, view_config[view_id]['tabs'][0]['name']))


	@app.route('/<tracer>/tables/<view_id>')
	@app.route('/<tracer>/tables/<view_id>/<subject_id>')
	def table_report(tracer, view_id, subject_id=None):
		data = {}
		for tab in view_config[view_id]['tabs']:
			data[tab['name']] = get_subject_data(view_id, tab['name'], tracer)
		return render_template('table_report.html', tracer=tracer, view_id=view_id, subject_id=subject_id, data=data,
			max_length=get_maxlength(data, view_config[view_id]['tabs'][0]['name']))


	@app.route('/<tracer>/img/<view_id>')
	@app.route('/<tracer>/img/<view_id>/<subject_id>')
	def img_report(tracer, view_id, subject_id=None):
		slice = request.args.get('slice')
		frame = request.args.get('frame')

		data = {}
		dims = {}
		for tab in view_config[view_id]['tabs']:
			data[tab['name']] = get_subject_data(view_id, tab['name'], tracer)
			# figure out shape of images being shown in tab
			sample_img_filename = os.path.join(app.config['PROJECT_FOLDER'], list(itertools.chain.from_iterable(data[tab['name']][0]['data'].values()))[0])
			temp = nb.load(sample_img_filename).get_data()
			if temp.shape[-1] == 1:
				temp = np.reshape(temp, temp.shape[:-1])
			dims[tab['name']] = len(temp.shape)

			max_slice = temp.shape[2] if len(temp.shape) >= 3 else None
			max_frame = temp.shape[3] if len(temp.shape) == 4 else None

		return render_template('img_report.html', tracer=tracer, view_id=view_id, subject_id=subject_id, data=data, max_slice=max_slice,
			max_frame=max_frame, slice=slice, frame=frame, dims=dims, max_length=get_maxlength(data, view_config[view_id]['tabs'][0]['name']))


	@app.route('/file/<path:filename>')
	def access_file(filename):
		return send_from_directory(app.config['PROJECT_FOLDER'], filename, as_attachment=True)


	@app.route('/imshow/<path:filename>')
	def access_slice(filename):
		imgdata = nb.load(os.path.join(app.config['PROJECT_FOLDER'], filename)).get_data()

		# imgdata could be 2, 3, or 4-dimensional -- only apply slice/frame selection where applicable
		imgdata_mindim = np.reshape(imgdata, imgdata.shape[:-1]) if imgdata.shape[-1] == 1 else imgdata # reshape to min dimensions if extra dimension present (i.e. if 3D image has 40th dim of 1)
		if len(imgdata_mindim.shape) > 2:
			slice = int(request.args.get('slice')) if request.args.get('slice') else imgdata.shape[2] // 2
			imgdata = np.take(imgdata, min(slice, imgdata.shape[2])-1, axis=2)
		if len(imgdata_mindim.shape) > 3:
			frame = int(request.args.get('frame')) if request.args.get('frame') else 1
			imgdata = np.take(imgdata, min(frame, imgdata.shape[-1])-1, axis=len(imgdata.shape)-1)

		outimg = np.reshape(imgdata, imgdata.shape[:2])
		outimg = np.flip(outimg, 1).T

		fig, ax = plt.subplots()
		ax.imshow(outimg, cmap='gray')
		ax.set_axis_off()
		img = BytesIO()
		fig.savefig(img)
		img.seek(0)
		plt.close()
		return send_file(img, mimetype='image/png')


	@app.route('/figdata/plot', methods=['GET'])
	def gen_plot():
		view_id = request.args.get('view_id')
		data_type = request.args.get('data_type')
		subject = request.args.get('subject')
		condition = request.args.get('condition')
		tracer = request.args.get('tracer')
		zoom = request.args.get('zoom')

		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		ax2 = plt.axes([0.65, 0.45, .2, .2])

		conditions = list(subject_map[subject].keys()) if condition == 'all' else [ condition ]
		conditions.sort()
		legend = []
		zoom_range = None

		for condition in conditions:
			filenames = [ item for filename in get_filename(view_id, data_type, subject, condition, tracer) \
				for item in glob(os.path.join(app.config['PROJECT_FOLDER'], filename)) ]

			# filenames = glob(os.path.join(app.config['PROJECT_FOLDER'], ))
			for filename in filenames:
				x, y = np.genfromtxt(filename, delimiter=',', skip_header=1, unpack=True)
				ax1.plot(x,y)

				ax2.plot(x[:14], y[:14]) # create zoomed in plot of first 120 seconds

				run = re.search('_([a-z]+(\d?))v\dr\d.csv$', filename).groups()[-1]
				legend.append(condition + ('-' + run if run else ''))

		ax1.set_title('{} {} for {}'.format(tracer.upper(), data_type, subject))
		ax1.set_xlabel('Time (s)') # FIXME: source these axis labels from the data itself
		ax1.set_ylabel('Specific activity')
		ax1.legend(legend)

		if not zoom:
			fig.delaxes(ax2) # only need zoomed in plot for long timecourses

		plot = BytesIO()
		fig.savefig(plot)
		plot.seek(0)
		plt.close()
		return send_file(plot, mimetype='image/png')

	return app


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_config', help='json file containing mapping of project-specific files to views', required=True)
	parser.add_argument('--subject_map', help='3-col csv (subject, session, label) to source data from', required=True)
	parser.add_argument('--view_config', help='json file containing view specification (i.e. type + order)', default='static/pet_view_conf.json')
	parser.add_argument('--port', default=8000, help='port to run app on')
	args = parser.parse_args()

	app = create_app(args.data_config, args.subject_map, args.view_config)
	app.run(debug=True, host='0.0.0.0', port=args.port)
