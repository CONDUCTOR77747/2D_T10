import matplotlib.pyplot as plt
import pandas as pd
import win32clipboard
from os import path
from statistics import mean
from datetime import datetime
import re
import sys
from matplotlib.widgets import SpanSelector, Cursor, Button, Slider, CheckButtons, TextBox

#Hello

""" MonkeyPatching matplotlib/widgets/TextBox/set_val "set_val without submitting"  """


def set_val(self, val):
    newval = str(val)
    if self.text == newval:
        return
    self.text_disp.set_text(newval)
    self._rendercursor()
    if self.eventson:
        self._observers.process('change', self.text)
        # self._observers.process('submit', self.text) <- without this row


TextBox.set_val = set_val


"""
required libraries: pandas, matplotlib, statistics, datetime, re, sys, win32clipboard, os.path
data file columns: Itot, Phi, ne, Radius, ECRH. Data file name: T10_%shot_number%_B%_I%.
First row must be: "Itot_x	Itot_y	Phi_x	Phi_y	ne_x	ne_y	Radius_x	Radius_y	ECRH_x	ECRH_y"
or if there is no ECRH: "Itot_x	Itot_y	Phi_x	Phi_y	ne_x	ne_y	Radius_x	Radius_y"
or if there is no ECRH and ne: "Itot_x	Itot_y	Phi_x	Phi_y	Radius_x	Radius_y"
"""


data_file = None  # path and name of data file
shot = None  # shot number

""" using sys.argv for input (command line input): 1st arg - path to data file; 2nd arg - energy """

if len(sys.argv) > 2:
    if sys.argv[1][0] == sys.argv[1][-1] == '\"':
        sys.argv[1] = sys.argv[1][1:-1]
    if path.exists(sys.argv[1]):
        data_file = sys.argv[1]  # data file name to load
        # parameters of signal
        shot = re.findall(r"T\d+_(.+)_B", data_file)[0]
        energy = sys.argv[2]
    else:
        print("Incorrect Path: ", sys.argv[1])
        sys.exit()
elif len(sys.argv) == 2 and sys.argv[1] == 'test':
    data_file = 'T10_70942_B17_I200_test.dat'
    energy = 120
else:
    print("Need 2 arguments: 1-data file path; 2-enegry.")
    sys.exit()

""" reading data from data file via pandas library. (Creating pandas dataframe) """
df = pd.read_csv(data_file, delimiter="\t")

Itot_y_max_value = 1  # for aligning Radius and ECRH signals
ne_flag = 0  # if ne signal exists ne_flag = 1 otherwise ne_flag = 0
ignore_itot_diapasons_flag = 0
accuracy = 1000  # accuracy for spline radius derivative recommend value: 500 - 1000
scans_limit = 100
maximize_window = 0

""" global variables counter, scan_counter. In functions: plt_scan_counter_update, span_onselect, btn_delete_last_scan
counts amount of scans and display it
"""
counter = 0  # initial value of scan counter
scan_counter = 'Scans: ' + str(counter)  # initial text of scan counter

list_axvspans = []  # list for saving (memorizing) spans. It helps to remove last scan by button and to create lists.
# scan counter position (Text "Scan: x")
s_c_pos_x = 0.01
s_c_pos_y = 0.65
# default values of plots
plot_Itot, plot_Phi, plot_ne, plot_Radius, plot_ECRH = None, None, None, None, None
list_Itot_x, list_Itot_y, list_Phi_x, list_Phi_y, list_ne_x = [], [], [], [], []
list_ne_y, list_Radius_x, list_Radius_y, list_ECRH_x, list_ECRH_y = [], [], [], [], []
# creating and setting up plot window
fig, ax = plt.subplots(figsize=(10, 6), dpi=80)
fig.canvas.manager.set_window_title(str('List Creator - ' + data_file))  # set plot window title
ax.grid()  # set grid
plt.title(data_file)  # set plot title
plt.xlabel("t (ms)")  # set x axis label


"""
Functions:

filters Itot and get time diapasons with signal
function goes through values of Itot signal. If its lower then threshold - pass, otherwise get time diapasons
"""


def get_time_diapasons_from_itot(data_list_x, data_list_y, threshold, min_time_diapason):
    time_diapasons = []  # list with itot time diapasons
    diapason_indexes = []  # list with indexes of itot_x for accessing itot_y
    flag = 0
    for i in range(len(data_list_y)):
        if data_list_y[i] < threshold:
            if flag == 0:
                pass
            else:
                flag = 0
                diapason_indexes.append(i)
            pass
        else:
            if flag == 1:
                pass
            else:
                flag = 1
                diapason_indexes.append(i)
            pass
    for i in range(0, len(diapason_indexes) - 1, 2):
        if data_list_x[diapason_indexes[i + 1]] - data_list_x[diapason_indexes[i]] >= min_time_diapason:
            time_diapasons.append(round(data_list_x[diapason_indexes[i]], 2))
            time_diapasons.append(round(data_list_x[diapason_indexes[i + 1]], 2))
    return time_diapasons


""" function applying changes (Sliders) """


def func_sliders_update(_):
    if len(list_axvspans_Itot_spans) > 0:
        for ind, val in enumerate(list_axvspans_Itot_spans):
            list_axvspans_Itot_spans[ind].remove()
        list_axvspans_Itot_spans.clear()
    # creating time diapasons by analyzing Itot signal
    time_diapasons = get_time_diapasons_from_itot(list_Itot_x, list_Itot_y, threshold_slider.val,
                                                  min_time_diapason_slider.val)
    # plotting time intervals for itot
    for i in range(0, len(time_diapasons) - 1, 2):
        list_axvspans_Itot_spans.append(
            ax.axvspan(time_diapasons[i], time_diapasons[i + 1], color="yellow", alpha=0.3))
    fig.canvas.draw_idle()  # redraw plot


""" scan counter updater """


def plt_scan_counter_update():
    text_scan_counter.set_position((s_c_pos_x, s_c_pos_y))  # set position of scan counter
    text_scan_counter.set_text(str(scan_counter))  # set updated text scan counter
    plt.draw()  # draw updated counter


"""
Span Selector:

span selector event
"""


def span_onselect(xmin, xmax):
    global counter, scan_counter
    if counter < scans_limit:
        counter += 1  # increment scan counter
        scan_counter = 'Scans: ' + str(counter)  # create updated string
        list_axvspans.append(ax.axvspan(xmin, xmax, color="red", alpha=0.3))  # add span to list
    else:
        scan_counter = 'Over Limit \n' + str(scans_limit) + ' scans'  # limit for scans - 100
    plt_scan_counter_update()  # display scans (axvspans)
    return xmin, xmax


""" button for deleting last scan """


def btn_delete_last_scan_on_clicked(_):
    global counter, scan_counter  # using global variables
    if counter > 0:  # if amount of scans more than 0
        counter -= 1  # decrement scan counter
        scan_counter = 'Scans: ' + str(counter)  # create updated string
        list_axvspans[-1].remove()  # remove span from plot
        list_axvspans.pop(-1)  # remove removed span object from list
        plt_scan_counter_update()  # display changes on plot
        # print('Delete last scan')  # print if button delete last scan was pressed


""" button for deleting all scans """


def btn_delete_all_scans_on_clicked(_):
    global counter, scan_counter  # global variables for counter
    if counter > 0:
        counter = 0  # set scans counter to 0
        scan_counter = 'Scans: ' + str(counter)  # create updated string
        for _ in range(len(list_axvspans)):  # for cleaning all spans one by one
            list_axvspans[-1].remove()  # remove last span from the plot
            list_axvspans.pop(-1)  # remove last element from the list
        plt_scan_counter_update()  # display changes on the plot
        # print('Delete all scans')  # print if the button delete all scans was pressed


""" opening list file and parsing data """


def func_load_list_from_file(data):
    with open(data, 'r') as file:
        content = file.read()
    pattern = r"from(.+)to(.+)}"
    list_file_load = re.findall(pattern, content)
    for elem in list_file_load:
        span_onselect(float(elem[0]), float(elem[1]))


""" list data file load clipboard and submitting """


def func_textbox_submit(__):
    win32clipboard.OpenClipboard()
    try:
        data = win32clipboard.GetClipboardData()
    except TypeError:
        data = ''  # non-text
    win32clipboard.CloseClipboard()
    if data:
        if data[0] == data[-1] == '\"':
            data = data[1:-1]
        if path.exists(data):
            if data[-5:] == ".list":
                text_box.set_val(data)  # MonkeyPatched set_val (matplotlib/widgets/text_box/set_val - patched)
                func_load_list_from_file(str(data))
            else:
                text_box.set_val("Incorrect Data Fromat (.list needed): " + data)
        else:
            text_box.set_val("Incorrect Path: " + data)


""" checking interception of itot spans "( )" and created spans "| |".
"|"-span_min, "|"-span_max, "("-itot_min, ")"-itot_max - explanation via brackets """


def intercept_intervals(span_min, span_max, itot_min, itot_max):
    if span_min <= itot_min and itot_max <= span_max:  # |()| (or spans are equal "|==( and )==|")
        return itot_min, itot_max  # ()
    elif itot_min < span_min < itot_max <= span_max:  # (|)| (or two right sides are equal ")==|")
        return round(span_min, 2), itot_max  # |)
    elif span_min <= itot_min < span_max < itot_max:  # |(|) (or two left sides are equal "|==(")
        return itot_min, round(span_max, 2)  # (|
    elif itot_min < span_min < span_max < itot_max:  # (||)
        return round(span_min, 2), round(span_max, 2)  # ||
    else:
        return None


""" returns list of signal mean values for each Itot diapason. properties: (signal_x, signal_y, list of selected
diapasons, flag if signal was loaded) """


def signal_mean_value(list_x, list_y, done_list, flag):
    if flag == 1:  # check flag if signal was loaded from data file
        list_mean_values = []  # creating list for signal mean values
        coord_sensitivity = 0.1  # deviation between signal coordinate and itot_x coordinate. necessary because of
        # different amount of points in itot and signal
        found_min_flag, found_max_flag = 0, 0  # if one value is found stop searching another
        index_x_min, index_x_max = 0, 0  # indexes of signal
        for scans in done_list:  # go through list with done diapasons of itot
            for i in range(len(list_x)):  # go through signal list
                # checking deviation between two coordinates - itot_x and signal to get indexes of signal
                if abs(round(list_x[i], 2) - scans[0]) < coord_sensitivity and found_min_flag == 0:
                    index_x_min = i  # getting index
                    found_min_flag = 1  # if index found set found flag - 1
                    # print('min', index_x_min, x_min, round(list_x[i], 2), abs(round(list_x[i], 2)-x_min))
                if abs(round(list_x[i], 2) - scans[1]) < coord_sensitivity and found_max_flag == 0:
                    index_x_max = i
                    found_max_flag = 1
                    # print('max', index_x_max, x_max, round(list_x[i], 2), abs(round(list_x[i], 2)-x_max))
                if found_min_flag == 1 and found_max_flag == 1:
                    # using indexes to get y values of signal to calculate mean value
                    list_mean_values.append(round(mean(list_y[index_x_min:index_x_max + 1]), 3))
                    break  # exit child loop and go to the next itot diapason
            found_min_flag, found_max_flag = 0, 0  # set flags to 0
        if len(done_list) == len(list_mean_values):  # if amount of itot diapasons is equal to amount of signal means
            return list_mean_values  # returns list of signal mean values for each Itot diapason
    else:  # if signal wasn't loaded return none
        return None


""" legend disable plots function """


def on_pick_legend(event):
    # On the pick event, find the original line corresponding to the legend
    # proxy line, and toggle its visibility.
    legend = event.artist
    origline1 = lined[legend]
    visible = not origline1.get_visible()
    origline1.set_visible(visible)
    # Change the alpha on the line in the legend so we can see what lines have been toggled.
    legend.set_alpha(1.0 if visible else 0.2)
    fig.canvas.draw()


""" creating list file function """


def create_list_file(list_scans, list_ne_means):
    # creating output filename
    time_format = '%d-%m-%Y %H:%M:%S'
    file_format = '.list'
    save_dir = 'Lists\\'
    file_name = re.findall(r"T\d+_\d+_B\d+_I\d+", data_file)[0]

    #  with timestamp
    # itot_file_path = '%s%s_%s_%s%s' % (save_dir, file_name, 'Itot', datetime.now().strftime(time_format), file_format)
    # phi_file_path = '%s%s_%s_%s%s' % (save_dir, file_name, 'Phi', datetime.now().strftime(time_format), file_format)

    #  without timestamp
    itot_file_path = '%s%s_%s%s' % (save_dir, file_name, 'Itot', file_format)
    phi_file_path = '%s%s_%s%s' % (save_dir, file_name, 'Phi', file_format)

    # creating content of output file

    f_itot = open(itot_file_path, 'a')
    f_itot.write("!"+datetime.now().strftime(time_format)+'\n')
    f_phi = open(phi_file_path, 'a')
    f_phi.write("!" + datetime.now().strftime(time_format) + '\n')

    if list_ne_means is not None:
        for i in range(len(list_scans)):
            f_itot.write("T10HIBP::Itot{relosc333, slit3, clean, noz, shot" + shot + ", from" + "{:.2f}".format(list_scans[i][0]) + "to" +
                         "{:.2f}".format(list_scans[i][1]) + "} !" + "{:.3f}".format(list_ne_means[i]) + " #" + shot + ' E = ' + energy + '\n')

            f_phi.write("T10HIBP::Phi{slit3, clean, noz, shot" + shot + ", from" + "{:.2f}".format(list_scans[i][0]) + "to" +
                    "{:.2f}".format(list_scans[i][1]) + "} !" + "{:.3f}".format(list_ne_means[i]) + " #" + shot + ' E = ' + energy + '\n')

    else:
        for i in range(len(list_scans)):
            f_itot.write("T10HIBP::Itot{relosc333, slit3, clean, noz, shot" + shot + ", from" + "{:.2f}".format(list_scans[i][0]) + "to" +
                         str("{:.2f}".format(str(list_scans[i][1]))) + "} !" + " #" + shot + ' E = ' + energy + '\n')

            f_phi.write("T10HIBP::Phi{slit3, clean, noz, shot" + shot + ", from" + "{:.2f}".format(list_scans[i][0]) + "to" +
                    "{:.2f}".format(list_scans[i][1]) + "} !" + " #" + shot + ' E = ' + energy + '\n')

    f_itot.close()
    f_phi.close()


""" button for creating lists """


def func_btn_create_lists_on_clicked(_):
    done_scans = []  # list for clean scans [(t1,t2),(*,*),...] diapasons of Itot
    if counter > 0:  # if counter more than 0
        if len(list_axvspans_Itot_spans) > 0 and ignore_itot_diapasons_flag == 0:
            # print('Create lists')
            for span in list_axvspans:  # go through created spans
                for itot in list_axvspans_Itot_spans:  # go through itot spans
                    # checking interception between Itot diapasons and created spans
                    diapason = intercept_intervals(span.get_xy()[0][0], span.get_xy()[-2][0],
                                                   itot.get_xy()[0][0],
                                                   itot.get_xy()[-2][0])
                    if diapason is not None:
                        done_scans.append(diapason)  # adding diapasons to list
        elif ignore_itot_diapasons_flag == 1:
            for span in list_axvspans:
                done_scans.append((round(span.get_xy()[0][0], 2), round(span.get_xy()[-2][0], 2)))
        done_scans.sort()  # sorting diapasons
        mean_ne = signal_mean_value(list_ne_x, list_ne_y, done_scans, ne_flag)  # getting list of ne mean values of itot
        # intervals
        # if mean_ne is not None:  # if mean values is correct
        create_list_file(done_scans, mean_ne)
        done_scans.clear()  # clearing diapasons list after saving


""" display and ignore yellow itot diapasons """


def ignore_itot_diapasons(_):
    global ignore_itot_diapasons_flag
    if ignore_itot_diapasons_flag == 0:
        ignore_itot_diapasons_flag = 1
        for _ in range(len(list_axvspans_Itot_spans)):  # for cleaning all itot spans one by one
            list_axvspans_Itot_spans[-1].remove()  # remove last itot span from the plot
            list_axvspans_Itot_spans.pop(-1)  # remove last element from the itot spans list
    else:
        ignore_itot_diapasons_flag = 0
        func_sliders_update(threshold_slider.val)


""" Itot - loading and analyzing Itot signal """


if {'Itot_x', 'Itot_y'}.issubset(df.columns):  # does Itot signal exist in data file
    list_Itot_y = df['Itot_y'].tolist()  # loading Itot signals to list
    list_Itot_x = df['Itot_x'].tolist()
    Itot_y_max_value = max(list_Itot_y)  # for aligning Radius and ECRH signals
    # Plotting Itot Signal
    plot_Itot, = ax.plot(list_Itot_x, list_Itot_y, label='Itot (kA)', color='royalblue')
    # Make a horizontal slider to control the min time diapason.
    mtd_slider_pos = plt.axes([0.65, 0.016, 0.25, 0.03], facecolor='lightgoldenrodyellow')  # min time diapason position
    # creating slider for min time diapason
    min_time_diapason_slider = Slider(ax=mtd_slider_pos, label='Minimal \nTime \nDiapason', valmin=0, valmax=3,
                                      valinit=2.56, valstep=0.01)

    # Make a vertical slider to control the threshold. (Itot signal)
    # min time diapason position
    threshold_slider_pos = plt.axes([0.94, 0.2, 0.02, 0.4], facecolor='lightgoldenrodyellow')
    # creating horizontal slider for min time diapason
    threshold_slider = Slider(ax=threshold_slider_pos, label='Threshold \n(Itot)', valmin=0, valmax=0.2,
                              valinit=0.2, valstep=0.01, orientation='vertical')
    list_axvspans_Itot_spans = []  # list for memorizing all Itot spans

    func_sliders_update(threshold_slider.val)  # plotting spans with initial slider values
    threshold_slider.on_changed(func_sliders_update)  # if changed create new time diapasons
    min_time_diapason_slider.on_changed(func_sliders_update)  # if changed create new spans with new threshold

    # scan counter parameters
    text_scan_counter = ax.text(s_c_pos_x, s_c_pos_y, scan_counter, fontsize=14, transform=plt.gcf().transFigure)

    # span selector parameters
    span_selector = SpanSelector(ax, span_onselect, 'horizontal', useblit=True,
                                 props=dict(alpha=0.3, facecolor='tomato'))

    # cursor parameters
    cursor = Cursor(ax, horizOn=True, vertOn=True, color='grey', linewidth=1, alpha=0.5)

    btn_del_l_s_pos = plt.axes([0.01, 0.79, 0.09, 0.08])  # button for deleting scans position
    btn_del_last_scan = Button(ax=btn_del_l_s_pos, label='Delete last \n scan')  # button for Del Last Scan parameters
    # if button clicked - execute function btn_delete_last_scan
    btn_del_last_scan.on_clicked(
        btn_delete_last_scan_on_clicked)  # if button was pressed do function btn_delete_last_scan_on_clicked

    btn_all_s_pos = plt.axes([0.01, 0.90, 0.09, 0.08])  # button for deleting scans position
    btn_all_scans = Button(ax=btn_all_s_pos, label='Delete all \n scans')  # button for Del All Scans parameters
    # if button clicked - execute function btn_delete_last_scan
    btn_all_scans.on_clicked(btn_delete_all_scans_on_clicked)  # if the button was pressed

    btn_create_lists_pos = plt.axes([0.01, 0.50, 0.09, 0.08])  # button for creating lists position
    btn_create_lists = Button(ax=btn_create_lists_pos, label='Create lists')  # button for creating lists parameters
    btn_create_lists.on_clicked(
        func_btn_create_lists_on_clicked)  # if button clicked - execute function btn_create_lists

    axCheckButton = plt.axes([0.01, 0.35, 0.08, 0.13])
    checkbox = CheckButtons(axCheckButton, ['Ignore\nItot\nDiapasons'], [False])
    checkbox.on_clicked(ignore_itot_diapasons)


""" Loading saved lists """


initial_text = "Press Here to Paste List Path from Clipboard and Load Data"
axbox = plt.axes([0.3, 0.94, 0.5, 0.05])  # position of textbox x,y,w,h
text_box = TextBox(axbox, 'Load List:', initial=initial_text)  # creating textbox
text_box.on_submit(func_textbox_submit)  # If Enter pressed (submitting by default)

# Phi - loading and plotting Phi signal
if {'Phi_x', 'Phi_y'}.issubset(df.columns):
    list_Phi_y = df['Phi_y'].tolist()  # loading Phi signals to list
    list_Phi_x = df['Phi_x'].tolist()
    # Plotting Phi Signal
    plot_Phi, = ax.plot(list_Phi_x, list_Phi_y, label='Phi (kV)', color='magenta')


""" ne - loading, cleaning and plotting ne signal. Cleaning because of different dimension of ne signal (about 11998
points) in comparison with Itot, Phi, Radius signals (about 1 million points)  """


if {'ne_x', 'ne_y'}.issubset(df.columns):
    ne_flag = 1
    list_ne_y_dirty = df['ne_y'].tolist()  # loading ne signals to list
    list_ne_x_dirty = df['ne_x'].tolist()
    list_ne_y = [float(y) for y in list_ne_y_dirty if y != "--"]  # making ne signal clean
    list_ne_x = [float(x) for x in list_ne_x_dirty if x != "--"]
    list_ne_y_dirty.clear()  # delete dirty signal lists with "--"
    list_ne_x_dirty.clear()
    # Plotting ne Signal
    plot_ne, = ax.plot(list_ne_x, list_ne_y, label='ne (10^-19 m^-3)', color='red')


""" Radius - loading, set align and scale of Radius signal """


if {'Radius_x', 'Radius_y'}.issubset(df.columns):
    list_Radius_y = df['Radius_y'].tolist()  # loading Radius signals to list
    list_Radius_x = df['Radius_x'].tolist()
    # aligning radius
    Radius_y_min_value = min(list_Radius_y)  # for aligning Radius
    Radius_y_max_value = max(list_Radius_y)
    list_Radius_y_aligned = [(y - Radius_y_min_value) * Itot_y_max_value / (Radius_y_max_value - Radius_y_min_value) for
                             y in list_Radius_y]
    # Plotting Aligned Radius Signal
    plot_Radius, = ax.plot(list_Radius_x, list_Radius_y_aligned, label='Radius Aligned', color='green')


""" loading ECRH signal and amplifying it """


if {'ECRH_x', 'ECRH_y'}.issubset(df.columns):
    list_ECRH_y_dirty = df['ECRH_y'].tolist()
    list_ECRH_x_dirty = df['ECRH_x'].tolist()
    # if there is no ECRH signal delete lists of nan (double check)
    if pd.isna(list_ECRH_x_dirty[0]) or pd.isna(list_ECRH_y_dirty[0]):
        list_ECRH_y_dirty.clear()  # clear unnecessary list of nan's
        list_ECRH_x_dirty.clear()
    else:
        list_ECRH_y = [float(y) for y in list_ECRH_y_dirty if
                       y != "--"]  # making ECRH signal clean without '--' values
        list_ECRH_x = [float(x) for x in list_ECRH_x_dirty if x != "--"]
        list_ECRH_y_dirty.clear()  # delete dirty signal lists with "--"
        list_ECRH_x_dirty.clear()
        ECRH_y_max_value = max(list_ECRH_y)  # for aligning ECRH signal
        list_ECRH_y_aligned = [(float(y) * Itot_y_max_value / ECRH_y_max_value) for y in list_ECRH_y]  # aligning
        # Plotting ECRH Signal
        plot_ECRH, = ax.plot(list_ECRH_x, list_ECRH_y_aligned, label='ECRH Aligned', color='orange')

del df  # removes all pandas dataframe from memory


""" creating interactive legend """


leg = ax.legend(loc='lower left', bbox_to_anchor=(-0.16, -0.15), framealpha=0.8)  # set legend on plot
pre_lines = [plot_Itot, plot_Phi, plot_ne, plot_Radius, plot_ECRH]
lines = list(filter(None, pre_lines))  # cleaning the list from Nones if some plots haven't been plotted
lined = {}  # Will map legend lines to original lines.
for legline, origline in zip(leg.get_lines(), lines):
    legline.set_picker(True)  # Enable picking on the legend line.
    legline.set_pickradius(10)
    lined[legline] = origline

fig.canvas.mpl_connect('pick_event', on_pick_legend)  # if legend item is pressed this item will hide

# adding HIBP logo image
im = plt.imread('Resources/HIBP_logo.png') # insert local path of the image.
newax = fig.add_axes([0.847,0.847,0.15,0.15], anchor='NE', zorder=1)
newax.imshow(im)
newax.axis('off')

# maximize and display plot window
if maximize_window:
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

plt.show()
