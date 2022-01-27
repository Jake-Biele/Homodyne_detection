
from __future__ import print_function, division
import time, warnings, datetime, os

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.ndimage.filters import gaussian_filter
import numpy as np

# LR 12/07/17

import seaborn as sns

sns.set_context("paper", font_scale=4, rc={"axes.facecolor": 'white',
	                                       "font.family": "serif"})
	                                       #"font.sans-serif": ["Times"]})

sns.set_style("ticks", {"xtick.major.size": 7,
	                        "xtick.minor.size": 4,
	                        "ytick.major.size": 7,
	                        "ytick.minor.size": 4,
	                        "xtick.direction": "in",
	                        "ytick.direction": "in",
	                        "axes.linewidth": 2.75})

# sns.set_style({'font.family':'sans-serif', 'font.sans-serif':'Times'})

#print(sns.axes_style())


_fig_number = 0

class fig(object):

	def __init__(self, *data):
		super(fig, self).__init__()
		
		# Unique ID for keeping track of plot windows
		global _fig_number
		self.fig_number = _fig_number
		_fig_number += 1


		self.x_data = []
		self.y_data = []
		self.n_y_data = 1

		self.x_scale = 1
		self.y_scale = 1

		self.x_range = [None, None]
		self.y_range = [None, None]
		self.second_y_range = [None, None]

		self.update_data(data)

		self.x_label = ''
		self.y_label = ''
		self.labelpad = 10
		self.axfontweight = 300
		self.title = ''
		self.joined = True
		self.plotted = False
		
# 		self.figsize = (10, 6.25) # in inches
		self.figsize = (10,(10*3)/4) # in inches

		self.legend = None
		self.legend_loc = 'best'
		self.legend_box = True
		self.bbox = None
		self.legend_ncols = 1

		
		self.tick_lab = 'plain'
		self.tick_ax = 'both'
		self.tick_offset = False
		self.logx = False
		self.logy = False
		self.intx = False
		self.inty = False
		
		self.linewidth = 3.5
		self.colour = '#00035b'
		self.colours = []

		self.xerr = None
		self.yerr = None

		self.marker = ''
		self.markers = []
#         self.add_vline = []
		self.markerline = False

		self.markersize = 12
		self.capsize = 5
		self.elinewidth = 2.
		self.alpha = 0.6
		self.regression = False

		self.annotate = []
		self.annotate_xys = []  # each element should be of the form [(xi,yi), (xf,yf)] for start and end coords
		self.annotate_size = 14.0
		self.arrow_colour = 'k'
		self.arrow_width = 0.1
		self.headwidth = 0.2
		self.headlength = 0.4

		self.annotate_imgs = []
		self.annotate_img_xys = []
		self.annotate_zoom = 1.0
		self.annotate_rotation = 0

		# For second y-axis
		self.second_y_data = []
		self.second_y_label = ''
		self.second_y_colours = []
		self.second_y_markers = []
		self.second_legend = None
		self.second_legend_loc = 'best'
		
		self.fill = False
		self.fill_alpha = 1
		self.fill_colour = '#00035b'
		
		self.vline_x = None
		self.vline_y_range = None
		self.vline_colour = '#00035b'
		self.vline_marker = '--'
		self.vline_label = ''

		# For bar chart plotting set to True
		self.bar = False

	def update_data(self, data):
		if data is None:
			return
		
		# Find number of layers on the data sets
		onion, old_layer = data, []
		array_lengths, array_types, data_len = [], [], []
		try:
			i = 0
			while not isinstance(onion, (float, int, np.int32, np.float64)):
				array_lengths.append(len(onion))
				array_types.append(type(onion))
				old_layer = onion
# 				print (onion)
				onion = onion[0]
				i += 1
		except:
			pass

# 		print ('data (onion) has {:} layers'.format(i))
# 		print ('with structure {:}'.format(array_lengths))

		case1 = array_lengths[-1] == 2 and array_lengths[0] == 1   # [(x1,y1),(x2,y2)...]
		case2 = array_lengths[-2] >= 2 and array_lengths[-1] > 2   # [xdata, ydata1, ydata2, ...]
		case3 = array_lengths[-2] == 1 							   # ydata
		case4 = array_lengths[-2] == 2 and array_lengths[-1] > 2 and len(array_lengths) >= 3  # [(xdata1, ydata1),(xdata2, ydata2),...]
		
		
		if array_lengths[0] == 1:
			data = data[0]

		if case4:
			self.x_data = []
			self.y_data = []
			
			for d in data:
				if not isinstance(d[0], (np.ndarray)):
					self.x_data.append(np.array(d[0]))
				else:
					self.x_data.append(d[0])
				if not isinstance(d[1], (np.ndarray)):
					self.y_data.append(np.array(d[1]))
				else:
					self.y_data.append(d[1])

				
			self.n_y_data = array_lengths[-3]
# 			print ('Case 4')
			
		elif case3:
			# Case (3)
			self.y_data = np.array(data)
			self.x_data = np.arange(0,len(data))
			self.n_y_data = 1
# 			print ('Case 3')
			
		elif case2:
			# Case (2)
			if not isinstance(data[0], (np.ndarray)):
				self.x_data = np.array(data[0])
			else:
				self.x_data = data[0]
			
			
			self.n_y_data = array_lengths[-2] - 1
			self.y_data = []
			
			for i in range(self.n_y_data):
				d = data[i+1]
				
				if not isinstance(d, (np.ndarray)):
					d = np.array(d)
				self.y_data.append(d)
			
# 			print ('Case 2')
			
		elif case1:
			# Case (1)
			if not isinstance(data, (np.ndarray)):
				data = np.array(data)
			self.x_data = data[:, 0]
			self.y_data = data[:, 1]
			self.n_y_data = 1
# 			print ('Case 1')

		else:
			AttributeError("Unable to parse data :(")


	def plot(self, target='screen', filename=''):

		style = ''

		if not len(self.annotate_imgs) == len(self.annotate_img_xys):
			raise RuntimeError('Number of annotation images xy positions does not match the number of annotations')


		fig = plt.figure(self.fig_number, figsize=self.figsize)
		ax = fig.add_subplot(111)
		
		# Add some sexy major and minor ticks
		ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
		ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
		ax.tick_params(which='major', width=1.3, top = 'on', right = 'on')
		ax.tick_params(which='minor', width=1.15, top = 'on', right = 'on')
		
		markers = ['o', '^', 's', 'D', '*', 'v', '>','<'] if not self.markers else self.markers
		colours = ['#00035b', '#e50000', '#06470c', '#9a0eea', '#f97306', '#06c2ac', '#653700', '#ff66ff', '#ffff00', '#000000'] if not self.colours else self.colours
		# Figure out if we need to unpack the error bars for each dataset
		try:
			if isinstance(self.yerr[0], (tuple, list, np.ndarray)):
				unpack_yerr = True
			else:
				unpack_yerr = False
		except:
			unpack_yerr = False
		try:
			if isinstance(self.xerr[0], (tuple, list, np.ndarray)):
				unpack_xerr = True
		except:
			unpack_xerr = False
		else:
			unpack_xerr = False
		
		for i in range(self.n_y_data):
			# Do linear regression if specified and if we are not using log axes
			if self.regression and (not self.logx and not self.logy):
				m, c, r_value, p_value, std_err = linregress(self.x_data, self.y_data[i])
				self.gradient, self.intercept = m, c

				# Calculate the standard error between slope and the y-data
				std_err = (sum([((y - (m*x + c))**2) for x, y in zip(self.x_data, self.y_data[i])])/(len(self.y_data[i])))**0.5
				
				# take the minimum x value
				xmin = min([self.x_range[0], min(self.x_data)]) if self.x_range[0] is not None else min(self.x_data)
				xmax = max([self.x_range[1], max(self.x_data)]) if self.x_range[1] is not None else max(self.x_data)
				
				# Plot the regression
				xs = [xmin, xmax]
				ys = [m*x + c for x in xs]
				plt.plot(xs, ys, label='Regression')
				
				if self.yerr is None:
					self.yerr = std_err
				
			# Marker, legend and error bar handling
			if not self.joined:
				if self.marker:
					m = self.marker
				else:
					m = markers[i%len(markers)]
			elif self.markerline:
				if self.marker:
					m = '-' + self.marker
				else:
					m = '-' + markers[i % len(markers)]
			else:
				m = ''

			c = colours[i%len(colours)] if self.n_y_data > 1 else self.colour
			label = self.legend[i] if self.legend is not None else ''
			xerr = self.xerr[i] if unpack_xerr else self.xerr
			yerr = self.yerr[i] if unpack_yerr else self.yerr
			
			# Final round of parsing to plot different cases
			try:
				x_data = self.x_data[i] if not isinstance(self.x_data[0], (float, int, np.float64, np.int32)) else self.x_data
			except:
				x_data = self.x_data
			
			try:
				y_data = self.y_data[i] if self.n_y_data > 1 else self.y_data
			except:
				y_data = self.y_data
			
			try:
				if isinstance(y_data, list):
					y_data = y_data[0]
			except:
				pass

			# Plot the dataset
			if self.bar:
				ax.bar(x_data, y_data, align = 'center', color=c, label=label, alpha=self.alpha[i])
			else:
				line = plt.errorbar(x_data/self.x_scale, y_data/self.y_scale,
								label=label, xerr=xerr, yerr=yerr, fmt=m, color=c,
								markersize=self.markersize, capsize=self.capsize, elinewidth=self.elinewidth, linewidth = self.linewidth)
                # Handle errorbar style
				if self.xerr is not None or self.yerr is not None:
					caps = line[1]
					for cap in caps:
						 cap.set_markeredgewidth(self.linewidth)

		if self.y_range[0] is not None: plt.ylim(ymin=self.y_range[0])
		if self.y_range[1] is not None: plt.ylim(ymax=self.y_range[1])
		if self.x_range[0] is not None: plt.xlim(xmin=self.x_range[0])
		if self.x_range[1] is not None: plt.xlim(xmax=self.x_range[1])

		if self.fill is True:
			plt.fill_between(x_data/self.x_scale, self.y_data[0], self.y_data[1], facecolor=self.fill_colour[0], alpha=self.fill_alpha, where =  self.y_data[0]>= self.y_data[1])
			plt.fill_between(x_data/self.x_scale, self.y_data[0], self.y_data[1], facecolor=self.fill_colour[1], alpha=self.fill_alpha, where =  self.y_data[0]<= self.y_data[1])

		if self.vline_x is not None: plt.vlines(self.vline_x, self.vline_y_range[0], self.vline_y_range[1], colors=self.vline_colour, linestyles=self.vline_marker, label=self.vline_label, linewidths=self.linewidth)
			
		if self.annotate:
			font_properties = dict(size=self.annotate_size, rotation=self.annotate_rotation)
			arrow = dict(arrowstyle = "simple,head_length={0},head_width={0},tail_width={0}".format(self.headlength, self.headwidth, self.arrow_width),
						 ec = self.arrow_colour,
						 fc = self.arrow_colour)

			for ann, xy in zip(self.annotate, self.annotate_xys):
				offsetbox = mpl.offsetbox.TextArea(ann, textprops=font_properties, minimumdescent=False)

				start, end = xy
				ab = mpl.offsetbox.AnnotationBbox(offsetbox, xy,
										xybox=xy,
										xycoords='axes fraction',
										boxcoords='axes fraction',
										arrowprops=None,
										frameon=False)
				ax.add_artist(ab)



		# Handle any images used for annotating the figure
		if self.annotate_imgs:
			i = 0
			for img, xy in zip(self.annotate_imgs, self.annotate_img_xys):
				# add a annotation images
				arr_hand = plt.imread(img)#, format='png')

				# handle the zoom of annotation (number or array?)
				if isinstance(self.annotate_zoom, (float, int)):
					zoom = self.annotate_zoom
				elif isinstance(self.annotate_zoom, (list, np.ndarray)):
					zoom = self.annotate_zoom[i]
					i += 1

				# Add an image at the specified coordinates
				imagebox = mpl.offsetbox.OffsetImage(arr_hand, zoom=zoom)
				ab = mpl.offsetbox.AnnotationBbox(imagebox, xy,
												xybox=xy,
												xycoords='axes fraction',
												boxcoords='axes fraction',
												frameon=False)
				ax.add_artist(ab)


		
		# Log axes
		if self.logx:
			ax.set_xscale("log")
		if self.logy:
			ax.set_yscale("log")
		
		# Axis labels
		ax.set_xlabel(self.x_label, weight = self.axfontweight)
		ax.set_ylabel(self.y_label, weight = self.axfontweight)
		ax.set_title(self.title, weight = self.axfontweight)
		ax.xaxis.labelpad = self.labelpad
		ax.yaxis.labelpad = self.labelpad
		
		if self.second_y_data:
            
			ax2 = ax.twinx()
			label = self.second_legend[0] if self.second_legend is not None else ''
            
			if self.second_y_range[0] is not None: plt.ylim(ymin=self.second_y_range[0])
			if self.second_y_range[1] is not None: plt.ylim(ymax=self.second_y_range[1])
                
			for _d_ in range(np.shape(self.second_y_data)[0]):
                
				ax2.errorbar(x_data, self.second_y_data[_d_],
					label=label, xerr=xerr, yerr=yerr, fmt=self.second_y_markers[_d_], color=self.second_y_colours[_d_],
					markersize=self.markersize, capsize=self.capsize, elinewidth=self.elinewidth, linewidth = self.linewidth)
#			ax2.plot(self.x_data, self.second_y_data, color=self.second_y_colours, linewidth = self.linewidth)

			# Legend properties
			if self.second_legend != None and len(self.second_legend) < 15:
            
				lines, labels = ax.get_legend_handles_labels()
				lines2, labels2 = ax2.get_legend_handles_labels()
				ax2.legend(lines + lines2, labels + labels2, loc=self.second_legend_loc, ncol = self.legend_ncols)
				if self.bbox is None:
					leg = ax2.legend(loc=self.second_legend_loc, frameon=self.legend_box, ncol = self.legend_ncols)
				else:
					leg = ax2.legend(loc=self.second_legend_loc, frameon=self.legend_box, bbox_to_anchor=self.bbox, ncol = self.legend_ncols)
				if self.legend_box:
					leg.get_frame().set_edgecolor('k')
					leg.get_frame().set_linewidth(1.0)

            
			ax2.set_ylabel(self.second_y_label, weight = self.axfontweight)
			ax2.yaxis.labelpad = self.labelpad
            
			# Add some sexy major and minor ticks
			ax2.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
			ax2.tick_params(which='major', width=1.3, top = 'on', right = 'on')
			ax2.tick_params(which='minor', width=1.15, top = 'on', right = 'on')

		# Legend properties
		if self.legend != None and len(self.legend) < 15:
			if self.bbox is None:
				legg = ax.legend(loc=self.legend_loc, frameon=self.legend_box, ncol = self.legend_ncols)
			else:
				legg = ax.legend(loc=self.legend_loc, frameon=self.legend_box, bbox_to_anchor=self.bbox, ncol = self.legend_ncols)
			if self.legend_box:
				legg.get_frame().set_edgecolor('k')
				legg.get_frame().set_linewidth(1.0)
              
		# This no longer works for some reason, removes the offset from axes ticks
		axes = fig.gca()
		if self.intx: axes.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
		if self.inty: axes.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
		
		if target == 'screen':
			plt.show()

		if target == 'file':
			if os.path.exists(filename):
				if filename.endswith('.png'):
					filename = filename.split('.png')[0]
					filename += ' ' + timestamp() + '.png'
				elif filename.endswith('.pdf'):
					filename = filename.split('.pdf')[0]
					filename += ' ' + timestamp() + '.pdf'
			if filename == '':
				filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fig{}.png'.format(timestamp()))
			self.filename = filename
			fig.savefig(filename, bbox_inches='tight')
		
		self.plotted = True

		plt.close()

	def write_to_csv(self, filename = '', metadata_kvps = {}):
		if filename == '':
			try:
				filename = self.filename+'.csv'
			except:
				filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fig.csv')
		self.filename = filename

		multiple_x_data = True if not isinstance(self.x_data[0], (int, float)) else False

		data = []
		if not multiple_x_data:
			data.append(self.x_data)

		for i, d in enumerate(self.y_data):
			if multiple_x_data:
				data.append(self.x_data[i])
			data.append(d)
		
		# Transpose?
		try:
			data = [list(x) for x in zip(*data)]
		except:
			print ("This needs to be debugged:")
			print ("data was: {:}".format(data))
			print ("x_data was: {:}".format(self.x_data))
			print ("y_data was: {:}".format(self.y_data))
			data = []

		labels = []
		labels.append(self.x_label)

		if self.legend:
			for elem in self.legend:
				labels.append(elem)
				if multiple_x_data:
					labels.append(self.x_label)
		else:
			labels.append(self.y_label)


		metadata = {
			'Original filename':filename, 
			'Plotted':self.plotted,
			'Number of Y datasets':self.n_y_data}

		metadata.update(metadata_kvps)

		write_to_csv(data, filename, metadata, labels)


class contour(object):

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

        self.n_lines = 5
        self.linewidth = 0.6

        self.n_levels = 1000
        self.filter = True
        self.sigma = 0.01

        self.cmap = plt.cm.RdYlBu_r
        self.figsize = (14, 7)
        
        self.title = ''
        self.title_size = 20
        self.x_label = ''
        self.y_label = ''

        self.ticks = True
        self.xax_tick_dp = 0
        self.yax_tick_dp = 0
        
        self.cbar_ticks = []
        self.cbar_dp = None
        self.cbar_fmt = 'f'
        self.cbar_limits = []
        self.cbar_greaterThan = False
        self.cbar_lessThan = False
        self.cbar_label = ''
        
        self.major_dir = 'in'
        self.major_w = 1.75
        self.major_l = 10

        self.minor_dir = 'in'
        self.minor_w = 1.5
        self.minor_l = 5

        self.add_vline = []

        self.annotate = []
        self.annotate_xys = []
        self.annotate_size = 14.0
        self.arrow_colour = 'k'
        self.arrow_width = 0.1
        self.headwidth = 0.2
        self.headlength = 0.4
        
        self.annotate_imgs = []
        self.annotate_img_xys = []
        self.annotate_zoom = 1.0

        self.logx = False
        self.logy = False

        self.plotted = False

    def plot(self, target='screen', filename=''):
        _fig, ax = plt.subplots(figsize=self.figsize)
        
        if self.logy:
            ax.set_yscale("log")
        if self.logx:
            ax.set_xscale("log")
        
        if self.title:
        #_fig.suptitle(self.title, fontsize=self.title_size)
            ax.set_title(self.title)
        
        if self.filter:
            self.z = gaussian_filter(self.z, sigma=self.sigma)
        
        if self.n_lines > 0:
            ax.contour(self.x, self.y, self.z, self.n_lines, colors='k', linewidths=self.linewidth)
        elif isinstance(self.n_lines, (list, np.ndarray)):
            ax.contour(self.x, self.y, self.z, levels=self.n_lines, colors='k', linewidths=self.linewidth)
        cbar_limits = [self.z.min(), self.z.max()] if not self.cbar_limits else self.cbar_limits
        levels = np.linspace(cbar_limits[0], cbar_limits[-1], self.n_levels)
        cax = ax.contourf(self.x, self.y, self.z, levels=levels, cmap=self.cmap)
        
        #cbar_limits = [self.z.min(), self.z.max()] if not self.cbar_limits else self.cbar_limits
        #levels = np.linspace(cbar_limits[0], cbar_limits[-1], self.n_lines)
        
        #ax.contour(self.x, self.y, self.z, levels=levels, colors='k', linewidths=self.linewidth)
        #cax = ax.contourf(self.x, self.y, self.z, levels=levels, cmap=self.cmap, extend='both')


        # ColourBar tick options
        cbar_ticks = [self.z.min(), self.z.max()] if not self.cbar_ticks else self.cbar_ticks
        cbar = _fig.colorbar(cax, ax=ax, ticks=cbar_ticks, boundaries = self.cbar_limits)
        strfmt = '{:.'+str(self.cbar_dp)+self.cbar_fmt+'} dB '
        
        if self.cbar_ticks:
            
            if self.cbar_greaterThan is True:
                
                tic = [strfmt.format(tick) for tick in self.cbar_ticks]
                st = '>{:.'+str(self.cbar_dp)+self.cbar_fmt+'}'
                tic[-1] = st.format(self.cbar_ticks[-1])
                cbar.ax.set_yticklabels(tic)
                
            else:
                
                cbar.ax.set_yticklabels([strfmt.format(tick) for tick in self.cbar_ticks])
            
        elif self.cbar_dp == -1:
            cbar.ax.set_yticklabels([int(self.z.min()), int(self.z.max())])
        elif self.cbar_dp >= 0:
            #cbar.ax.set_yticklabels([round(self.z.min(), self.cbar_dp), round(self.z.max(), self.cbar_dp)])
            cbar.ax.set_yticklabels([strfmt.format(self.z.min()), strfmt.format(self.z.max())])
        else:
            cbar.ax.set_yticklabels([self.z.min(), self.z.max()])

        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        cbar.ax.set_ylabel(self.cbar_label)

        if self.ticks:
            # Add a new axes overlay
            newax = _fig.add_axes(ax.get_position())

            # Patch for inward facing ticks (contourf hides them!)
            x = ax.get_xticks()
            y = ax.get_yticks()


            # Hide old axes
            ax.yaxis.set_visible(False)
            ax.xaxis.set_visible(False)

            # Make plot transparent
            newax.patch.set_visible(False)

            # Set the axes names
            newax.set_xlabel(self.x_label)
            newax.set_ylabel(self.y_label)

            if self.logy:
                newax.set_yscale("log")
            if self.logx:
                newax.set_xscale("log")

            # Adjust the limits to the data set min/max
            newax.set_xlim([min(self.x), max(self.x)])
            newax.set_ylim([min(self.y), max(self.y)])

            # Automatically find best place for major minor ticks
            newax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
            newax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())

            # Set the new axes tick labels as the old values
            newax.set_xticklabels(x)
            newax.set_yticklabels(['', '1 \u03BCm'+r'$^{-1}$','10 \u03BCm'+r'$^{-1}$', '0.1 cm'+r'$^{-1}$', '1 cm'+r'$^{-1}$'])
            
            # Set the formatting strings for the ticks
            newax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.'+str(self.xax_tick_dp)+'f'))
            #newax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.'+str(self.yax_tick_dp)+r'f cm$^{-1}$'))

            # Finally set the size. length, width and direction for the ticks
            newax.tick_params(which='major', width=self.major_w, length=self.major_l, direction=self.major_dir)
            newax.tick_params(which='minor', width=self.minor_w, length=self.minor_l, direction=self.minor_dir)
    
        if self.add_vline:
            for v in self.add_vline:
                ax.plot([v, v], ax.get_ylim(), '--k')
                
        if self.annotate:
            font_properties = dict(size=self.annotate_size)
            arrow = dict(arrowstyle = "simple,head_length={},head_width={},tail_width={}".format(self.headlength, self.headwidth, self.arrow_width),
						 ec = self.arrow_colour,
						 fc = self.arrow_colour)

            for ann, xy in zip(self.annotate, self.annotate_xys):
                offsetbox = mpl.offsetbox.TextArea(ann, textprops=font_properties, minimumdescent=False)

                start, end = xy
                ab = mpl.offsetbox.AnnotationBbox(offsetbox, start,
										xybox=end,
										xycoords='axes fraction',
										boxcoords='axes fraction',
										arrowprops=arrow,
										frameon=False)
                ax.add_artist(ab)



		# Handle any images used for annotating the figure
        if self.annotate_imgs:
            i = 0
            for img, xy in zip(self.annotate_imgs, self.annotate_img_xys):
                # add a annotation images
                arr_hand = plt.imread(img)#, format='png')

                # handle the zoom of annotation (number or array?)
                if isinstance(self.annotate_zoom, (float, int)):
                    zoom = self.annotate_zoom
                elif isinstance(self.annotate_zoom, (list, np.ndarray)):
                    zoom = self.annotate_zoom[i]
                    i += 1

                # Add an image at the specified coordinates
                imagebox = mpl.offsetbox.OffsetImage(arr_hand, zoom=zoom)
                ab = mpl.offsetbox.AnnotationBbox(imagebox, xy,
                                                xybox=xy,
                                                xycoords='axes fraction',
                                                boxcoords='axes fraction',
                                                frameon=False)
                ax.add_artist(ab)
        
        if target == 'screen':

            plt.show()
        else:
            if not filename:
                raise AttributeError('Filename required')
            if not filename.endswith('.pdf'):
                filename += '.pdf'
            # Patch white lines between levels on pdf export
            for c in cax.collections:
                c.set_edgecolor("face")
            _fig.savefig(filename, bbox_inches='tight')

        self.plotted = True

        plt.close()


def timestamp():
	return datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')



def write_to_csv(data, filename = '', metadata_kvps = {}, top_labels = [], left_labels = []):
	
	if os.path.exists(filename):
		filename = filename.split('.csv')[0]
		filename += ' ' + timestamp() + '.csv'
	file = open(filename, 'w')

	# Add a timestamp to the metadata
	metadata_kvps.update({'Timestamp':timestamp()})



	for key in metadata_kvps:
		file.write(key)
		file.write(', ')
		file.write(str(metadata_kvps[key]))
		file.write('\n')
	
	file.write('\n')

	if top_labels != []:
		if left_labels != []:
			file.write(' , ')
		file.write(str(top_labels[0]))
		for label in top_labels[1:]:
			file.write(', ')
			file.write(str(label))
		file.write('\n')

	for i in range(max(len(left_labels), len(data) ) ):
		if left_labels != []:
			if i < len(left_labels):
				file.write(left_labels[i])

			else:
				file.write(' ')
			file.write(', ')

		if i < len(data):
			file.write(str(data[i][0]))
			for j in range(len(data[i] )-1 ):
				file.write(', ')
				file.write(str(data[i][j+1]))
		file.write('\n')

	file.close()


def unique_filename(base_fn, extension = '', suffix_style = 'decimal'):

	# Handle different suffix styles {0,1,2...}, {a,b,c...}, {0x0,0x1,0x2...}
	if suffix_style == 'decimal':
		def suf(index):
			return str(index)
	elif suffix_style == 'alpha':
		def suf(index):
			if index > 26:
				raise AttributeError('Index too high')

			return chr(index+ord('a'))
	elif suffix_style == 'hex':
		def suf(index):
			return hex(index)

	
	index = 0

	fn = base_fn.rstrip() + ' ' + suf(index) + extension

	while (os.path.isfile(fn) ):
		index += 1
		fn = base_fn.rstrip() + ' ' + suf(index) + extension

	return fn



def bar_str(f, f_max, n_tot = 20):
	"""ASCII bar charts"""
	rep = '='
	empty = ' '
	term = '|'

	string = term
	for i in range(n_tot):
		if float(i)/n_tot <= float(f)/f_max:
			string += rep
		else:
			string += empty
	string += term
	string += ' '
	string += str(int(round(f_max)))
	return string


if __name__ == "__main__":
    # # Test contour plot
    # x = np.array([-1, -2, -3])
    # y = np.array([4, 5, 8])
    # z = np.array([[-6, 2, 3], range(3, 6), [6,7,10]])
    #
    # f = contour(x,y,z)
    # f.n_levels = 1000
    # f.plot()

    #===========================================================================
	# case1:  [(x1,y1),(x2,y2)...]						OK - List, numpy
	# case2:  [xdata, ydata1, ydata2, ...] 				OK - List, numpy
	# case3:  ydata										OK - List, numpy
	# case4:  [(xdata1, ydata1),(xdata2, ydata2),...] 	OK - List, numpy
	#===========================================================================

#### PYTHON LIST IMPLEMENTEATION
#	f = fig([(1,2),(3,4),(5,6)]) 		 		# Case 1 OK
# 	f = fig([1,3,4],[5,6,7])   			 		# Case 2 OK
# 	f = fig([[1,3,4],[5,6,7]])					# Case 2.0 NOK --> apply  fig(*matrix)
# 	f = fig([1,3,4],[5,6,7],[7,8,9])     		# Case 2.1 OK
# 	f = fig([[1,3,4],[5,6,7],[7,8,9]])   		# Case 2.2 OK
#	f = fig([1,3,4])			 		 		# Case 3 OK
# 	f = fig([[[1,3,4],[5,6,7]],[[3,4],[7,9]]])  # Case 4 OK

#### NUMPY IMPLEMENTEATION
# 	f = fig(np.array([(1,2),(3,4),(5,6)]))								# Case 1 OK
# 	f = fig(np.array([1,3,4]),np.array([5,6,7]))   			   			# Case 2 OK
# 	f = fig(np.array([1,3,4]),np.array([5,6,7]),np.array([7,8,9]))     	# Case 2.1 OK
# 	f = fig([np.array([1,3,4]),np.array([5,6,7]),np.array([7,8,9])])    # Case 2.2 OK
# 	f = fig(np.array([1,3,4]))		 		 							# Case 3 OK
#	f = fig([[np.array([1,3,4]),np.array([5,6,7])],[np.array([3,4]),np.array([7,9])]])  					# Case 4 OK
#	f = fig(np.array([[np.array([1, 3, 4]), np.array([5, 6, 7])], [np.array([3, 4]), np.array([7, 9])]]))  	# Case 4.1 OK
#	f = fig(np.array([[[1, 3, 4], [5, 6, 7]], [[3, 4], [7, 9]]]))  											# Case 4.2 OK
#	f.plot(target='screen')


	f = fig([(1,2),(3,4),(5,6)])

# 	annotation = '/Users/lawrencerosenfeld/Google Drive/PhD Yr2/TeX/SWIR_Device_review/img/Old/grating_pic.png'
# 	xys = (0.15, 0.80)
# 	if os.path.exists(annotation):
# 		f.annotate_imgs = [annotation]
# 		f.annotate_img_xys = [xys]
# 		f.annotate_zoom = 0.10
# 
# 	f.annotate = ['Check this feature!']
# 	f.annotate_xys = [[(0.5, 0.5), (0.55, 0.7)]]
# 	f.headlength  = 0.2
# 	f.arrow_width = 0.1


#	f.x_range = (1, 5)
# 	f.y_range = (-1, 5)
# 	f.logx = True
# 	f.logy = True
# 	f.joined = False
# 	f.legend_box = True
# 	f.regression = True
# 	f.legend = 'data'
# 	f.yerr = [1,2,3,4]
# 	f.xerr = [1, 10, 100, 1000]
# 	f.marker = 's'
# 	print (f.x_data, f.y_data)
# 	print (f.fig_number)
# 	f.joined = False
# 	f.marker = 's'
# 	f.markerline = True
	f.plot()


# 	f = fig([3,2,1,0])
# 	f.x_label = 'X axis (A.U.)'
# 	f.y_label = 'Y axis (A.U.)'
# 	f.joined = False
# 	f.plot(target='file', filename='testing.pdf')

	# write_to_csv(data = [[0,1,2],[3,4,5]], filename = 'test.csv', metadata_kvps = {'Type':0}, top_labels = ["First", "Second", "Third"], left_labels = ['l1','l2'])