#nimue.py
#Nimue - a nice interface for a better Merlin user experience
#Copyright (c) 2019 Simon Kollbach


from tkinter import *
from tkinter import ttk
from tkinter.filedialog   import askopenfilenames
import glob
import os
import shutil
import subprocess
import math
from functools import partial
from subprocess import PIPE
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from matplotlib.backends.backend_agg import FigureCanvasAgg
import time
from threading import Thread
import multiprocessing
import simpleaudio as sa
from queue import Queue
import signal
import pickle
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.MatplotlibDeprecationWarning)
import matplotlib.backends.tkagg as tkagg

merlinPath = open("nimue/data/merlinpath.txt").readline()
merlinPath = merlinPath[:-1]
print("|"+ merlinPath + "|")

def loadDict(dictName):
	global selectedSubVoice
	dictionary = {}
	file = open("nimue/data/" + dictName + ".csv", "r")
	lines = file.readlines()
	for line in lines:
		try:
			line = line.replace("$selectedSubVoice", selectedSubVoice)
		except:
			True
		line = line.split("; ")
		key = line[0]
		value = line[1]
		if value[0] == "(":
			#handles the font-style tuples
			value = value[1:-2].split(", ")
			value[1] = int(value[1])
			dictionary[key] = (value[0][1:-1], value[1])
		else:
			dictionary[key] = value[:-1]
	return dictionary

root = Tk()
#try to load previously pickled settings
try:
	nimueSettings = pickle.load( open("nimue/nimue_settings.p", "rb"))
except:
	print("Could not load custom settings")
	#provide some standard settings
	nimueSettings = {"calcPowerFactor":None,
					"lang":"en",
					"cs":"avalon",
					"textSize":"Medium",
					"lastVoice":"slt_arctic",
					"lastSubVoice":"slt_arctic_demo"}
	
#set styles according to loaded settings
if nimueSettings["cs"] == "camelot":
	cs = loadDict("cs_ca")
	fs = loadDict("fs_ca")
#elif nimueSettings["cs"] == "mordred":
	#This is how you would add a new colorstyle
	#cs = loadDict("cs_mo")
	#fs = loadDict("fs_mo")
else:
	#if there is no valid setting, use the standard Avalon theme
	cs = loadDict("cs_av")
	fs = loadDict("fs_av")

layerTypeOptions = ["TANH", "SIGMOID", "RELU", "RESU", "TANH_LHUC", "RNN", "SGRU", "SLSTM", "LSTM", "BSLSTM", "GRU", "BLSTM", "LSTM_LHUC", "LSTM_NFG", "LSTM_NOG", "LSTM_NIG", "LSTM_NPH", "SOFTMAX"]
seqTypes = ["SLSTM", "SGRU", "GRU", "LSTM_NFG", "LSTM_NOG", "LSTM_NIG", "LSTM_NPH", "LSTM", "BSLSTM", "BLSTM", "RNN", "LSTM_LHUC"]
layerSizeOptions = ["32", "64", "128", "256", "512", "1024"]



if nimueSettings["lang"] == "de":
	tex = loadDict("tex_de")
	srs = loadDict("srs_de")
else:
	tex = loadDict("tex_en")
	srs = loadDict("srs_en")

root.option_add('*TCombobox*Listbox.background',cs["settingsBg"])
root.option_add('*TCombobox*Listbox.font', fs["settingText"])
style = ttk.Style()
style.configure("TCombobox", background=cs["settingsBg"], fieldbackground=cs["settingsBg"], selectbackground=cs["button"], selectforeground=cs["textColor"], arrowcolor=cs["button"])




###Functions that provide variables depending on settings or voice
def updateArguments():
	#Provides arguments for all scripts of the currently supported voices:
	#slt_arctic, build_your_own_voice
	#TODO: Provide more
	global selectedVoice
	global selectedsubVoice
	if selectedVoice == "slt_arctic":
		arguments = loadDict("args_slt")
	if selectedVoice == "build_your_own_voice":
		arguments = loadDict("args_byov")
	else:
		arguments = []
	return arguments	

def getLengthFactor(layerType):
	#every value marked with # is only estimated others are based on crude measurements
	lengthFactor = {"TANH":1.5, 
					"SIGMOID":1.09, 
					"RELU":1, 
					"RESU":2, 
					"TANH_LHUC":1.8, #
					"RNN":1.03, 
					"SGRU":2.9, 
					"SLSTM":2.24, 
					"LSTM":2.5, #
					"BSLSTM":6.38, 
					"GRU":3.1, #
					"BLSTM":8, #
					"LSTM_LHUC":2.7, #
					"LSTM_NFG":2.5, #
					"LSTM_NOG":2.5, #
					"LSTM_NIG":2.5, #
					"LSTM_NPH":2.5, #
					"SOFTMAX":1.5} #
	return lengthFactor[layerType]
	
def getBatchSizeFactor(i):
	batchFactor = {512:0.15,
					256:0.42,
					128:0.58,
					64:1,
					32:1.8,
					16:2.48,
					8:3.6}
	return batchFactor[i]

def isSupported(voice):
	sv = ["slt_arctic", "build_your_own_voice"]
	if voice in sv:
		return True
	else: 
		return False	

def getScriptReplaceString(script):
	global srs		
	name = srs[script]
	return name

def getSettingDescriptions():
	settingDescriptions_en = ["Choose an alignment", "Choose a vocoder",
			"Dropout Rate (0.0 to 1.0, recommended max 0.5 = 50%)", "Dropout Rate (0.0 to 1.0, recommended max 0.5 = 50%)",
			"Learning Rate (0.0 to 1.0, recommended max 0.02 = 2%)", "Learning Rate (0.0 to 1.0, recommended max 0.01 = 2%)",
			"Warm Up Epochs (recommended max: 10)", "Warm Up Epochs (recommended max: 10)",
			"Training Epochs (recommended at least 15)", "Training Epochs (recommended at least 15)", 
			"Batch Size", "Batch Size",
			"Number of Layers", "Number of Layers"]
	settingDescriptions_de = ["Alignment wählen", "Vocoder wählen",
			"Dropout Rate (0.0 bis 1.0, max. empfohlen 0.5 = 50%)", "Dropout Rate (0.0 bis 1.0, max. empfohlen 0.5 = 50%)",
			"Lernrate (0.0 to 1.0, max. empfohlen 0.02 = 2%)", "Lernrate (0.0 to 1.0, max. empfohlen: 0.02 = 2%)",
			"Aufwärmepochen (max. empfohlen: 10)", "Aufwärmepochen (max. empfohlen: 10)",
			"Traininsgepochen (min. empfohlen: 15)", "Traininsgepochen (min. empfohlen: 15)", 
			"Batch Size", "Batch Size",
			"Ebenenanzahl", "Ebenenanzahl"]
	if nimueSettings["lang"] == "de": 
		return settingDescriptions_de
	else:
		return settingDescriptions_en
		
###Function that draws diagrams
def draw_figure(canvas, figure, loc=(0, 0)):
    """ Draw a matplotlib figure onto a Tk canvas
    loc: location of top-left corner of figure on canvas in pixels.
    Inspired by matplotlib source: lib/matplotlib/backends/backend_tkagg.py
    "Copyright (c) 2002-2019 Matplotlib Development Team; All Rights Reserved"
    """
    figure_canvas_agg = FigureCanvasAgg(figure)
    figure_canvas_agg.draw()
    figure_x, figure_y, figure_w, figure_h = figure.bbox.bounds
    figure_w, figure_h = int(figure_w), int(figure_h)
    photo = PhotoImage(master=canvas, width=figure_w, height=figure_h)

    # Position: convert from top-left anchor to center anchor
    canvas.create_image(loc[0] + figure_w/2, loc[1] + figure_h/2, image=photo)

    # Unfortunately, there's no accessor for the pointer to the native renderer
    tkagg.blit(photo, figure_canvas_agg.get_renderer()._renderer, colormode=2)

    # Return a handle which contains a reference to the photo object
    # which must be kept live or else the picture disappears
    return photo

###Functions that help choosing a voice
def voiceFinder():
	#lists all the subfolders of /egs/ as voices
	voices = []
	for file in glob.glob(merlinPath + "/egs/*"):
		#exclude files, only use folders
		if "." in file:
			True
		else:	
    			voices.append(file[len(merlinPath)+5:])
	print(voices)
	return voices

def subVoiceFinder():
	global selectedVoice
	#lists all the subfolders of egs/selectedVoice/s1/experiments/ as subVoices
	subVoices = []
	#print("egs/"+selectedVoice+"/s1/experiments/*")
	subVoiceList = glob.glob(merlinPath + "/egs/"+selectedVoice+"/s1/experiments/*")
	#print(subVoiceList)
	for file in subVoiceList:
		#print(file)
		#exclude files, only use folders
		if "." in file:
			True
		else:
    			subVoices.append(file[21+len(selectedVoice)+len(merlinPath):])
	#print(subVoices)
	if subVoices == []:
		#if there are no subVoices, assume that there is a demo and a full version of every voice
		#TODO: exclude voices that don't fit in
		subVoices.append(selectedVoice+"_demo")
		subVoices.append(selectedVoice+"_full")
	return subVoices

def selectVoice(a, useLastVoice=False):
	#is called when a voice is selected
	global ownVoiceName
	global ownVoiceLabel
	global ownVoiceButton
	global subVoiceCombo
	global selectedVoice
	global selectSubVoiceLabel
	global buttonFrame
	global addWavButton
	global addTxtButton
	if useLastVoice:
		selectedVoice = nimueSettings["lastVoice"]
	else:
		selectedVoice = voice.get()
	#print("Selected voice: " + selectedVoice)
	subVoices = subVoiceFinder()
	try:
		subVoiceCombo.destroy()
		selectSubVoiceLabel.destroy()
	except:
		True
	selectSubVoiceLabel = Label(buttonFrame, text=tex["selectSubVoiceLabel"], font=fs["button"], bg=cs["specialBg"])
	selectSubVoiceLabel.pack(fill=X)
	subVoiceCombo = ttk.Combobox(buttonFrame, textvariable=voices, font=fs["settingText"])
	subVoiceCombo['values'] = subVoices
	func = partial(selectSubVoice, False)
	subVoiceCombo.bind('<<ComboboxSelected>>', func)
	subVoiceCombo.pack(fill=X, pady=2)
	#Give the user the opportunity to name his own voice
	try:
		ownVoiceLabel.destroy()
		ownVoiceName.destroy()
		ownVoiceButton.destroy()
		addWavButton.destroy()
		addTxtButton.destroy()
	except:
		True
	if selectedVoice == "build_your_own_voice":
		ownVoiceLabel = Label(buttonFrame, text=tex["ownVoiceLabel"], font=fs["button"], bg=cs["specialBg"])
		ownVoiceLabel.pack(fill=X, pady=2)
		ownVoiceName = Text(buttonFrame, font=fs["settingText"], fg=cs["textColor"], height=1, width=5)
		ownVoiceName.pack(fill=X, pady=2)
		func = partial(selectSubVoice, True, 1)
		ownVoiceButton = Button(buttonFrame, text=tex["ownVoiceButton"], font=fs["button"], bg=cs["button"], bd=cs["bdw"], relief=cs["bds"], highlightbackground=cs["borderColor"], activebackground=cs["buttonAct"], command=func)
		ownVoiceButton.pack(fill=X, pady=2)	
		addWavButton = Button(buttonFrame, text=tex["addWavButton"], font=fs["button"], bg=cs["button"], bd=cs["bdw"], relief=cs["bds"], highlightbackground=cs["borderColor"], activebackground=cs["buttonAct"], command=addWav)
		addWavButton.pack(fill=X, pady=2)	
		addTxtButton = Button(buttonFrame, text=tex["addTxtButton"], font=fs["button"], bg=cs["button"], bd=cs["bdw"], relief=cs["bds"], highlightbackground=cs["borderColor"], activebackground=cs["buttonAct"], command=addTxt)
		addTxtButton.pack(fill=X, pady=2)	
	#The next case should not occur as long as we assume every voice has a demo and full version
	if subVoices == []:
		updateButtons()
		updateInfo()

def selectSubVoice(new, a, useLastVoice=False):
	global selectedSubVoice
	global subVoiceCombo
	global ownVoiceName
	if new:
		selectedSubVoice = ownVoiceName.get(1.0, END)
		selectedSubVoice = selectedSubVoice[0:-1]
		print("|"+selectedSubVoice+"|")
		if selectedSubVoice == "":
			printOut(tex["voiceNeedsName"])
			return
	elif useLastVoice:
		try:
			selectedSubVoice = nimueSettings["lastSubVoice"]
		except:
			return
	else:
		selectedSubVoice = subVoiceCombo.get()
	updateButtons()
	updateInfo()


###Functions that display information or settings
def updateInfo():
	global cs
	global confFrame
	#Display information from the configuration file
	if isSupported(selectedVoice):
		updateConf()
		return
	global conf
	try:
		confFrame.destroy()
	except:
		True
	confFrame = Frame(bottomFrame, bg=cs["bg"])
	conf = Text(confFrame,  yscrollcommand=scrollbar.set, width=40, bg=cs["infoText"])
	conf.pack(side=LEFT)
	try:
		file = open(merlinPath + "/egs/"+selectedVoice+"/s1/conf/global_settings.cfg", "r")
	except FileNotFoundError:
		o = os.popen('.'+ merlinPath + '/egs/'+selectedVoice+'/s1/01_setup.sh').read()
		#print(o)
		try:
			file = open(merlinPath + "/egs/"+selectedVoice+"/s1/conf/global_settings.cfg", "r")
		except:
			True
			#print("File not there")
	try:
		lines = file.readlines()
	except:
		lines = tex["noConfFileInfo"]
	print(lines)
	global lc
	lc = len(lines)
	print(lc)
	global lineLabel
	lineLabel = []
	for i in range(0, lc):
		lineLabel.append(lines[i])
	
	for i in range(0, lc):
		conf.insert(END, lines[i])
	conf.config(state=DISABLED)
		
def updateButtons():
	#turns every .sh script into a button
	#TODO: add usage to "scripts"
	global cs
	global selectedVoice
	global commandField
	global indicator
	commandField.destroy()
	commandField = Frame(bottomFrame, padx=5, pady=5, bg=cs["specialBg"], bd=cs["bdw"], relief=cs["bds"])
	commandField.pack(side=RIGHT, fill=BOTH)
	indicatorField = Frame(commandField, bd=0, relief=FLAT, padx=5, pady=0, bg=cs["specialBg"])
	indicatorField.pack(side=RIGHT, fill=Y)
	scripts = []
	for file in glob.glob(merlinPath + "/egs/"+selectedVoice+"/s1/*"):
		if ".sh" in file:
			scripts.append(file[9+len(merlinPath)+len(selectedVoice):])
		else:
			
    			True
	global commandButton
	scripts.sort()
	commandButton = []
	commandSeparator = []
	indicator = []
	counter = 0
	xy = 1#28 #indicator height and width
	for i in range(0, len(scripts)):
		commandButton.append("")
		commandSeparator.append("")
		if scripts[i].find("0") >= 0:			
			indicator.append("")
	for i in range(0, len(scripts)):
		funcCall = partial(runScript, './' + merlinPath + '/egs/'+selectedVoice+'/s1/', scripts[i])
		commandButton[i] = Button(commandField, text=getScriptReplaceString(scripts[i]), font=fs["button"], anchor=W, bd=cs["bdw"], relief=cs["bds"], bg=cs["button"], highlightbackground=cs["borderColor"], activebackground=cs["buttonAct"], pady=5, command=funcCall).pack(fill=X, pady=3, padx=2)
		if scripts[i].find("0") >= 0:
			indicator[counter] = Button(indicatorField, font=fs["button"], pady=5, width=xy, bd=cs["bdw"], relief=cs["bds"], height=xy, highlightbackground=cs["borderColor"], bg=cs["inactive"])
			indicator[counter].pack(fill=X, pady=3, padx=2, side=TOP)
			counter = counter + 1
	saveButton = Button(commandField, text=tex["saveConfig"], font=fs["button"], anchor=W, bd=cs["bdw"], relief=cs["bds"], bg=cs["fixButton"], highlightbackground=cs["borderColor"], activebackground=cs["buttonAct"], pady=5, command=saveButtonFunc).pack(fill=X, pady=3, padx=2, side=BOTTOM)

def updateLayerSettings(parent, setIndex, init, init2, a):
	global settings
	global settingCategories
	global durLayers
	global acLayers
	global layerTypeOptions
	global layerSizeOptions
	layers = []
	#print(setIndex)
	#print(int(setIndex))
	#print(settings[int(setIndex)].get())
	try:
		num = int(settings[int(setIndex)].get())
	except:
		return
	init = getLayersFromString(init)
	init2 = getLayersFromString(init2)
	#print(init)
	#print(init2)
	#destroy every previous layerFrame
	try:
		x = durLayers[0]
	except:
		durLayers = []
	if settingCategories[setIndex] == "dur":
		try:
			for i in range(len(durLayers)):
				helpFrame = durLayers[i]
				helpHelpFrame = helpFrame[0]
				helpHelpFrame.destroy()
		except:
			True
	else:
		try:
			for i in range(len(acLayers)):
				helpFrame = acLayers[i]
				helpHelpFrame = helpFrame[0]
				helpHelpFrame.destroy()
		except:
			True
	layerFrame = []
	layerTitle = []
	layerType = []
	layerSize = []
	layerDescription = []
	for i in range(num):
		layerFrame.append(1)
		layerTitle.append(1)
		layerType.append(1)
		layerSize.append(1)
		layerDescription.append(1) 
		layers.append(1)
	for i in range(num):
		layerFrame[i] = Frame(parent, bg=cs["bg"])
		layerTitle[i] = Label(layerFrame[i], text=tex["layer"]+str(i+1),font=fs["settingLabel"], bg=cs["bg"], anchor=W, justify=LEFT)
		layerTitle[i].pack(fill=X, side=LEFT, padx=10)
		layerType[i] = ttk.Combobox(layerFrame[i], font=fs["settingText"], values=layerTypeOptions, width=15)
		layerSize[i] = ttk.Combobox(layerFrame[i], font=fs["settingText"], values=layerSizeOptions, width=10)
		try:
			initial = init2[i]
			#print(initial)
			initialIndex = layerTypeOptions.index(initial)
			#print(initialIndex)
			layerType[i].current(layerTypeOptions.index(init2[i]))
			layerSize[i].current(layerSizeOptions.index(init[i]))
		except:
			True
			print("Could not set initial value for layer " + str(i+1))
		layerType[i].pack(side=LEFT)
		layerSize[i].pack(side=RIGHT)
		layerDescription[i] = Label(layerFrame[i], text="", bg=cs["bg"], anchor=W, justify=LEFT)
		layerDescription[i].pack(fill=X, side=BOTTOM)
		layerFrame[i].pack(fill=X, pady=1)
		layers[i] = [layerFrame[i], layerTitle[i], layerType[i], layerSize[i], layerDescription[i]]
	#save local layers in global variable depending on category
	if settingCategories[setIndex] == "dur":
		durLayers = layers
	else:
		acLayers = layers	
	
def updateConf():
	global bottomFrame
	global cs
	global conf
	global confDur
	global confAc
	global confFrame
	global scrollbar
	global settings
	global selectedVoice
	global selectedSubVoice
	global infoFrame
	global settingCategories
	global settingKeywords
	global settingTypes
	scrollbar.destroy()
	try:	
		confFrame.destroy()
	except:
		#print("Could not destroy confFrame")
		True
	confFrame = Frame(bottomFrame, bg=cs["bg"])
	#read configuration files
	lines = tryReadFile(merlinPath + "/egs/"+selectedVoice+"/s1/conf/global_settings.cfg")
	linesDur = tryReadFile(merlinPath + "/egs/"+selectedVoice+"/s1/conf/duration_"+selectedSubVoice+".conf")
	linesAc = tryReadFile(merlinPath + "/egs/"+selectedVoice+"/s1/conf/acoustic_"+selectedSubVoice+".conf")
	if lines == "No File":
		fillIndicator(0)
		return
	elif linesDur == "No File" or linesAc == "No File":
		fillIndicator(-2)
	else:
		fillIndicator(-1)
	
	settings = [] 
	settingLabels = []
	#settingOptions = []
	#settingDescriptions = []
	#settingTypes = []
	#settingCategories = []
	#settingKeywords = []
	for i in range(14):
		settings.append(i)
		settingLabels.append(i)
		#settingOptions.append(i)
		#settingDescriptions.append(i)
		#settingTypes.append(i)
		#settingCategories.append(i)
		#settingKeywords.append(i)
	settingDescriptions = getSettingDescriptions()
	settingOptions = [["state_align", "phone_align"], ["STRAIGHT", "WORLD", "MAGPHASE"],
			 [],[],
			 [],[],
			 [],[],
			 [],[],
			["8","16","32","64","128","256"], ["8","16","32","64","128","256", "512"],
			["2","3","4","5","6"], ["2","3","4","5","6","7"] ]
	settingTypes = ["Combo", "Combo", 
			"Number", "Number",
			"Number", "Number",
			"Number", "Number",
			"Number", "Number",
			"Combo", "Combo",
			"LayerNumber", "LayerNumber"]
	settingCategories = ["gen", "gen",
				"dur", "ac",
				"dur", "ac",
				"dur", "ac",
				"dur", "ac",
				"dur", "ac",
				"dur", "ac"]
	settingKeywords = ["Labels=", "Vocoder=",
				"dropout_rate : ", "dropout_rate : ",
				"learning_rate : ", "learning_rate : ",
				"warmup_epoch    : ",  "warmup_epoch    : ", 
				"training_epochs : ", "training_epochs : ",
				"batch_size   : ", "batch_size   : ",
				 "hidden_layer_size", "hidden_layer_size"]
	x = 5
	y = 5
	conf = Frame(confFrame, bg=cs["bg"], pady=y, padx=x, bd=cs["bdw"], relief=cs["bds"])
	confDur = Frame(confFrame, bg=cs["bg"], pady=y, padx=x, bd=cs["bdw"], relief=cs["bds"])
	confAc = Frame(confFrame, bg=cs["bg"], pady=y, padx=x, bd=cs["bdw"], relief=cs["bds"])

	confTitle = Label(conf, text=tex["genConf"], font=fs["h1"], bg=cs["bg"], anchor=W, justify=LEFT).pack(fill=X, side=TOP)
	confDurTitle = Label(confDur, text=tex["durConf"], font=fs["h1"], bg=cs["bg"], anchor=W, justify=LEFT).pack(fill=X, side=TOP)
	confDurTitle = Label(confAc, text=tex["acConf"], font=fs["h1"], bg=cs["bg"], anchor=W, justify=LEFT).pack(fill=X, side=TOP)
	#Add commands and Buttons to open the configuration file
	genConfCommand = partial(openConfFile, merlinPath + "/egs/"+selectedVoice+"/s1/conf/global_settings.cfg")
	durConfCommand = partial(openConfFile, merlinPath + "/egs/"+selectedVoice+"/s1/conf/duration_"+selectedSubVoice+".conf")
	acConfCommand = partial(openConfFile, merlinPath + "/egs/"+selectedVoice+"/s1/conf/acoustic_"+selectedSubVoice+".conf")
	genConfFile = Button(conf, text=tex["openGenConfFile"], font=fs["button"], anchor=W, bd=cs["bdw"], relief=cs["bds"], bg=cs["fixButton"], highlightbackground=cs["borderColor"], activebackground=cs["buttonAct"], pady=5, command=genConfCommand).pack(fill=X, pady=3, padx=2, side=TOP)
	durConfFile = Button(confDur, text=tex["openDurConfFile"], font=fs["button"], anchor=W, bd=cs["bdw"], relief=cs["bds"], bg=cs["fixButton"], highlightbackground=cs["borderColor"], activebackground=cs["buttonAct"], pady=5, command=durConfCommand).pack(fill=X, pady=3, padx=2, side=TOP)
	acConfFile = Button(confAc, text=tex["openAcConfFile"], font=fs["button"], anchor=W, bd=cs["bdw"], relief=cs["bds"], bg=cs["fixButton"], highlightbackground=cs["borderColor"], activebackground=cs["buttonAct"], pady=5, command=acConfCommand).pack(fill=X, pady=3, padx=2, side=TOP)

	setFile = ""
	for i in range(len(settings)):
		#Specify from which file to read and in which frame to put the setting
		#if setting belongs to no category, skip it
		if settingCategories[i] == "gen":
			setFile = lines
			parent = conf
		elif settingCategories[i] == "dur":
			setFile = linesDur
			parent = confDur
		elif settingCategories[i] == "ac":
			setFile = linesAc
			parent = confAc
		else:
			#print("Setting does not belong to any category")
			continue
		#Handle missing config files by skipping the creation of this setting
		if setFile == "No File":
			#print("Could not create setting "+str(settingLabels[i])+": No valid configuration file found.")
			continue
		#Read the initial value
		for j in range(len(setFile)):
			index = setFile[j].find(settingKeywords[i])
			#print(index)
			if index >= 0:
				if setFile[j].find("hidden") >= 0:
					init = setFile[j][setFile[j].find("["):setFile[j].find("]")+1]
					index = j
					break
				index = j
				init = setFile[index][len(settingKeywords[i]):len(setFile[index])-1]
				break
		#print(settingKeywords[i])
		#print(setFile[j])
		
		#print("Initial Value is: "+init)
		#Create the label 
		settingLabels[i] = Label(parent, text=settingDescriptions[i], font=fs["settingLabel"], bg=cs["bg"], anchor=W, justify=LEFT).pack(fill=X)
		#create the setting
		if settingTypes[i] == "Combo":
			settings[i] = ttk.Combobox(parent, font=fs["settingText"], values=settingOptions[i])
			settings[i].pack(fill=X, pady=2)
			#Handle the case that the current value is not an option
			try:		
				initIndex = settingOptions[i].index(init)
				#print("Index of initial value is: " + str(initIndex))
				settings[i].current(initIndex)
			except:
				#print("I was not able to set the initial value")
				settings[i].current(0)
			
		elif settingTypes[i] == "Number":
			settings[i] = Text(parent, font=fs["settingText"], highlightbackground=cs["borderColor"], bg=cs["settingsBg"], fg=cs["textColor"], bd=cs["bdw"], relief=cs["bds"], height=1, width=5)
			settings[i].insert(END, init)
			settings[i].pack(fill=X, pady=2)
		elif settingTypes[i] == "DescriptionCombo":
			True
		elif settingTypes[i] == "LayerNumber":
			counter = 1
			for k in range(len(init)):
				if init[k] == ",":
					counter = counter + 1
			settings[i] = ttk.Combobox(parent, font=fs["settingText"], values=settingOptions[i])
			init2 = setFile[index+1][setFile[index+1].find("["):setFile[index+1].find("]")+1]
			funcCall = partial(updateLayerSettings, parent, i, init, init2)
			settings[i].bind('<<ComboboxSelected>>', funcCall)
			#Handle the case that the current value is not an option
			try:
				
				settings[i].current(settingOptions[i].index(str(counter)))
			except:
				#print("Could not set current value")
				#print("Current value is: "+str(counter))
				True
			settings[i].pack(fill=X, pady=2)
			layersLabel = Label(parent, text=tex["layers"], font=fs["h1"], bg=cs["bg"], anchor=W, justify=LEFT).pack(fill=X)
			updateLayerSettings(parent, i, init, init2, 1)
		else:
			#print("Setting has no type :(")
			continue


	conf.pack(side=LEFT, fill=Y, padx=5)
	confDur.pack(side=LEFT, fill=Y, padx=5)
	confAc.pack(side=RIGHT, fill=Y, padx=5)
	confFrame.pack(side=LEFT)

def updateDiagram(diagramType, vError, tError):
	diagramWindow = Toplevel()
	diagramWindow.title(tex["trainingResults"])
	#diagramWindow.geometry("700x700+200+200")
	infoFrame = Frame(diagramWindow)
	durationTrainingInfo = Frame(infoFrame, bd=2, relief=SUNKEN, bg=cs["bg"])
	durationTitle = Label(durationTrainingInfo, text=tex["durModelTraining"], bg=cs["bg"]).pack(side=TOP)
	durationTrainingInfo.pack(side=RIGHT, fill=Y)
	acousticTrainingInfo = Frame(infoFrame, bd=2, relief=SUNKEN, bg=cs["bg"])
	acousticTitle = Label(acousticTrainingInfo, text=tex["acModelTraining"], bg=cs["bg"]).pack(side=TOP)
	acousticTrainingInfo.pack(side=RIGHT, fill=Y)
	infoFrame.pack(side=RIGHT, fill=Y)
	global fig_photo
	global fig_photo2
	if diagramType == "Duration":
		try:
			durationDiagram.destroy()
		except:
			True
		durationDiagram = Canvas(durationTrainingInfo, width=660, height=660, bg=cs["bg"])
		durationDiagram.pack(fill=Y)
		acousticTrainingInfo.pack_forget()
	else:
		try:
			acousticDiagram.destroy()
		except:
			True
		acousticDiagram = Canvas(acousticTrainingInfo, width=660, height=660, bg=cs["bg"])
		acousticDiagram.pack(fill=Y)
		durationTrainingInfo.pack_forget()
	df = pd.DataFrame({'Epoch': list(range(len(vError))), 'Training Error': pd.Series(tError, dtype='float32'), 'Validation Error': pd.Series(vError, dtype='float32')})
	var = df['Training Error']
	var2 = df['Validation Error']
	factor = 2
	fig = pyplot.figure(figsize=[3.2*factor, 2.4*factor], dpi=100)
	ax1 = fig.add_subplot(1,1,1)
	ax1.set_xlabel(tex["diagramXLabel"])
	ax1.set_ylabel(tex["diagramYLabel"])
	var.plot(kind='line')
	var2.plot(kind='line')
	if diagramType == "Duration":
		fig_photo = draw_figure(durationDiagram, fig, loc=(5,5))
	else:
		fig_photo2 = draw_figure(acousticDiagram, fig, loc=(5,5))
	diagramWindow.mainloop()
	print("Diagram successfully updated!")

def fillIndicator(n):
	#-1 means fill to conf_files
	#any other negative means fill to setup

	global indicator
	#print(indicator)
	m = len(indicator)
	#print("There are " + str(m) + " indicator buttons")
	if n > m:
		return
	if n < 0:
		if n == -1:
			n = m-2+n
		else:
			return
	#print("Filling Indicator to " + str(n))
	for i in range(m):
		indicator[i].config(bg=cs["inactive"], activebackground=cs["inactive"])
	for i in range(n):
		#print("Filling indicator " + str(i))
		indicator[i].config(bg=cs["indicatorOn"], activebackground=cs["indicatorOn"])

def printOut(string):
	global printLabel
	printLabel.config(text=string)
	printLabel.pack(fill=X)

def openConfFile(confFile):
	saveButtonFunc()
	#print("attempting to open "+confFile)
	time.sleep(1.0)
	n = subprocess.Popen("xdg-open "+ confFile, stdout=PIPE, stderr = subprocess.STDOUT, shell=True)
	output = n.stdout.readline().strip().decode()
	print(output)
	time.sleep(0.5)
	n.poll()
	n.kill()
	#print("Successfully opened file "+confFile)
	printOut(tex["dontForgetSave"])

	
###Time estimation functions
###This is a highly unprecise feature although it is very useful
def calculate(i):
	if i <= 1:
		return 1
	else: 
		helpme = 0
		for j in range(i-1):
			helpme += calculate(j)
		return helpme
		
def timingTest2():
	global a
	a = Thread(target=timingTest)
	a.start()
	printOut(tex["startingCalibration"])
	
def timingTest():
	global nimueSettings
	#global printVariable
	#printVariable = "Starting calibration"
	start = time.time()
	a = Thread(target=calculate, args=(33,))
	b = Thread(target=calculate, args=(33,))
	c = Thread(target=calculate, args=(33,))
	d = Thread(target=calculate, args=(33,))
	a.start()
	b.start()
	c.start()
	d.start()
	a.join()
	b.join()
	c.join()
	d.join()
	end = time.time()
	result = end-start
	nimueSettings["calcPowerFactor"] = result
	if result < 10:
		printOut(tex["calibrationGood"])
	elif result < 30:
		printOut(tex["calibrationNeutral"])
	else:
		printOut(tex["calibrationBad"])
	print(end-start)

def estimateTime():
	global batchSize
	global epochs
	global nimueSettings
	if nimueSettings["calcPowerFactor"] is None:
		timingTest2()
	#time estimation for the current setting of acoustic training
	#(duration training takes not that long anyway)
	#well, that is the easiest way to get these...
	try:
		layers = getLayersFromString(getStringFromLayers("ac"))
		size = getLayersFromString(getStringFromLayers("ac", True))
	except:
		printOut(tex["timeEstimationError"])
		return
	time = 0
	num = len(layers)
	for i in range(num-1):
		if size[i] == '' or size[i+1] == '':
			printOut(tex["timeEstimationError"])
			return
		thisTime = (int(size[i])*int(size[i+1]))/9000.0
		totalLengthFactor = ((getLengthFactor(layers[i])+getLengthFactor(layers[i+1]))/2.0)*0.9
		#print("TTL : " + str(totalLengthFactor))
		thisTime = thisTime*totalLengthFactor
		time += thisTime
		#print("Layer " + str(i))
		#print(time)
	print(batchSize)
	print(getBatchSizeFactor(batchSize))
	time = time*getBatchSizeFactor(batchSize)
	print(time)
	time += 14.5
	time = time*epochs
	#correction factor (probably necessary to factor in input and output connections
	#correctionFactor = ((-0.0015)*num*num - 0.0115*num + 0.9544)
	time = time-(epochs*5)
	time = time * (nimueSettings["calcPowerFactor"]/19.5) #reference value
	print(time)
	if seqTrainNecessary("ac") == "True":
		printOut(tex["seqTrainingTime1"] + str(roundToFive(time*1)) + tex["seqTrainingTime2"] + str(roundToFive(time*2.5)) + tex["seqTrainingTime3"])
	else:
		printOut(tex["trainingTime1"] + str(roundToFive(time)) + tex["trainingTime2"])
	
def roundToFive(i):
	#Rounds up a value in seconds to a value in five-minute-steps
	i = math.ceil(i/300.0)
	return i*5
	
	
		
###Functions that help with saving or reading settings
def saveConfig(category):
	global settings
	global settingCategories
	global settingKeywords
	global batchSize
	global epochs 
	global selectedVoice
	epochs = 0
	if category == "gen":
		lines = tryReadFile(merlinPath + "/egs/"+selectedVoice+"/s1/conf/global_settings.cfg")
		lines2 = []
	elif category == "dur":
		lines = tryReadFile(merlinPath + "/egs/"+selectedVoice+"/s1/conf/duration_"+selectedSubVoice+".conf")
		lines2 = tryReadFile(merlinPath + "/egs/"+selectedVoice+"/s1/conf/test_dur_synth_"+selectedSubVoice+".conf")
	elif category == "ac":
		lines = tryReadFile(merlinPath + "/egs/"+selectedVoice+"/s1/conf/acoustic_"+selectedSubVoice+".conf")
		lines2 = tryReadFile(merlinPath + "/egs/"+selectedVoice+"/s1/conf/test_synth_"+selectedSubVoice+".conf")
	else:
		print("Category invalid")
		return
	newValue = ""
	#calculate the number of train, valid and test files
	if selectedVoice == "build_your_own_voice":
		fileDistribution = calculateFileDistribution()
		if fileDistribution[0] <= 0:
			printOut(tex["notEnoughFiles"])
			return
	for i in range(len(lines)):
		for j in range(len(settings)):
			if lines[i].find("hidden_layer_size") >= 0:
				lines[i] = "hidden_layer_size: "+ getStringFromLayers(category, size=True) + "\n"
				break
			if lines[i].find("hidden_layer_type") >= 0:
				lines[i] = "hidden_layer_type: "+ getStringFromLayers(category) + "\n"
				break
			if lines[i].find("sequential_training : ") >= 0:
				lines[i] = "sequential_training : " + seqTrainNecessary(category) + "\n"
				break
			if lines[i].find("train_file_number") >= 0 and selectedVoice == "build_your_own_voice":
				lines[i] = "train_file_number: " + str(fileDistribution[0]) + "\n"
				lines[i+1] = "valid_file_number: " + str(fileDistribution[1]) + "\n"
				lines[i+2] = "test_file_number: " + str(fileDistribution[2]) + "\n"
				break
			if lines[i].find("Train") >= 0 and selectedVoice == "build_your_own_voice" and category == "gen":
				lines[i] = "Train=" + str(fileDistribution[0]) + "\n"
				lines[i+1] = "Valid=" + str(fileDistribution[1]) + "\n"
				lines[i+2] = "Test=" + str(fileDistribution[2]) + "\n"
				break
					
			if (lines[i].find(settingKeywords[j]) >= 0 and settingCategories[j] == category):
				#useful for time estimation
				if settingKeywords[j] == "batch_size   : " and settingCategories[j] == "ac":
					batchSize = int(settings[j].get())
				if settingKeywords[j].find("epochs") >= 0 and settingCategories[j] == "ac":
					epochs += int(settings[j].get(1.0, END))
					print(epochs)
				try:				
					newValue = settings[j].get() + "\n"
				except:
					newValue = settings[j].get(1.0, END)
				lines[i] = settingKeywords[j] + newValue
				break
			
	for i in range(len(lines2)):
		for j in range(len(settings)):
			if lines2[i].find("hidden_layer_size") >= 0:
				lines2[i] = "hidden_layer_size: "+ getStringFromLayers(category, size=True) + "\n"
				break
			if lines2[i].find("hidden_layer_type") >= 0:
				lines2[i] = "hidden_layer_type: "+ getStringFromLayers(category) + "\n"
				break
			if lines2[i].find("sequential_training : ") >= 0:
				lines2[i] = "sequential_training : " + seqTrainNecessary(category) + "\n"
				break
			if lines[i].find("train_file_number") >= 0 and selectedVoice == "build_your_own_voice":
				lines2[i] = "train_file_number: " + str(fileDistribution[0]) + "\n"
				lines2[i+1] = "valid_file_number: " + str(fileDistribution[1]) + "\n"
				lines2[i+2] = "test_file_number: " + str(fileDistribution[2]) + "\n"
				break			
			if (lines2[i].find(settingKeywords[j]) >= 0 and settingCategories[j] == category):
				try:				
					newValue = settings[j].get() + "\n"
				except:
					newValue = settings[j].get(1.0, END)
				lines2[i] = settingKeywords[j]+newValue
				break

	#print(lines)
	#print(lines2)
	if category == "gen":
		tryWriteFile(merlinPath + "/egs/"+selectedVoice+"/s1/conf/global_settings.cfg", lines)
	elif category == "dur":
		tryWriteFile(merlinPath + "/egs/"+selectedVoice+"/s1/conf/duration_"+selectedSubVoice+".conf", lines)
		tryWriteFile(merlinPath + "/egs/"+selectedVoice+"/s1/conf/test_dur_synth_"+selectedSubVoice+".conf", lines2)
	elif category == "ac":
		tryWriteFile(merlinPath + "/egs/"+selectedVoice+"/s1/conf/acoustic_"+selectedSubVoice+".conf", lines)
		tryWriteFile(merlinPath + "/egs/"+selectedVoice+"/s1/conf/test_synth_"+selectedSubVoice+".conf", lines2)

def seqTrainNecessary(category):
	global seqTypes	
	global durLayers
	global acLayers
	layers = []
	if category == "ac":
		layers = acLayers
	elif category == "dur":
		layers = durLayers
	else:
		return "False"
	for i in range(len(layers)):
		help = layers[i]
		helphelp = help[2]
		#print(str(helphelp.get()))
		for j in seqTypes:
			if str(helphelp.get()) == j:
				return "True"
	return "False"
			
def tryReadFile(filename):
	try:
		file = open(filename, "r")
	except FileNotFoundError:
		o = os.popen('.' + merlinPath + '/egs/'+selectedVoice+'/s1/01_setup.sh').read()
		print(o)
		try:
			file = open(filename, "r")
		except:
			lines = "No File"
	try:
		lines = file.readlines()
	except:
		lines = "No File"
	return lines

def tryWriteFile(filename, text):
	try:
		#print("Attempting to open file " + filename)
		file = open(filename, "w")
		#print("File successfully opened!")
		file.writelines(text)
		#for i in range(len(text)):
		#	file.write(text[i])
		#print("Text written to file")
		file.close()
		#print("Closed file. It's always recommended to clean up after yourself!")
		return True
	except:
		print("Oh my god, something went horribly wrong while trying to write into configurationfile: "+filename)
		return False
	
def getStringFromLayers(category, size=False):
	global durLayers
	global acLayers
	index = 2
	if size:
		index = 3
	layers = []
	if category == "ac":
		layers = acLayers
	elif category == "dur":
		layers = durLayers
	else:
		return ""
	outStr = "["
	for i in range(len(layers)):
		help = layers[i]
		helphelp = help[index]
		#print(helphelp)
		if i != 0:
			outStr = outStr + " "
		if not size:
			outStr = outStr + "'"
		outStr = outStr + str(helphelp.get())
		if not size:
			outStr = outStr + "'"
		#print(str(helphelp.get()))
		if i != len(layers)-1:
			outStr = outStr + ","
	outStr = outStr + "]"
	#print("Saving String:")
	#print(outStr)
	return outStr

def getLayersFromString(string):
	global layerTypeOptions
	#remove "[" and "]" and split up in elements
	string = string[1:-1]
	string = string.split(",")
	for i in range(len(string)):
		#Case 1: Elements are LayerTypes, remove '' and spaces if necessary
		if string[i].find("'") >= 0:
			if string[i].find(" ") >= 0:
				#print("found a space in |" + string[i] + "|")
				string[i] = string[i][2:-1]
			else:
				string[i] = string[i][1:-1]
		#Case 2: Elements are LayerSizes, only remove spaces
		else:
			if string[i].find(" ") >= 0:
				string[i] = string[i][1:]
			else:
				string[i] = string[i]
	return string

def createDatabaseDirectory():
	global selectedVoice
	global selectedSubVoice
	targetDir = os.getcwd()+ "/" + merlinPath + "/egs/"+selectedVoice+"/s1/database/"+selectedSubVoice
	try:
		os.mkdir(targetDir)
	except:
		True
	subFolders = ["/wav/", "/txt/", "/labels/", "/feats/"]
	for folder in subFolders:
		try:
			os.mkdir(targetDir+folder)
		except:
			True

def addWav():
	try:
		targetDir = os.getcwd()+"/" + merlinPath + "/egs/"+selectedVoice+"/s1/database/"+selectedSubVoice+"/wav"
		files = askopenfilenames(filetypes=(("Wave files","*.wav"),))
		if files == []:
			return
		createDatabaseDirectory()
		for file in files:
			shutil.copy2(file, targetDir)
		printOut(tex["fileAddSuccess"])
	except:
		printOut(tex["fileAddError"])
		
def addTxt():
	try:
		targetDir = os.getcwd()+"/" + merlinPath + "/egs/"+selectedVoice+"/s1/database/"+selectedSubVoice+"/txt"
		files = askopenfilenames(filetypes=(("Wave files","*.txt"),))
		if files == []:
			return
		createDatabaseDirectory()
		for file in files:
			shutil.copy2(file, targetDir)
		printOut(tex["fileAddSuccess"])
	except:
		printOut(tex["fileAddError"])

def calculateFileDistribution():
	#targetDir = "egs/"+selectedVoice+"/s1/database/"+selectedSubVoice+"/wav/*.wav"
	#counter = 0
	#for file in glob.glob(targetDir):	
		#counter = counter + 1
	#print(str(counter) + " Files found")
	target = merlinPath + "/egs/"+selectedVoice+"/s1/experiments/"+selectedSubVoice+"/duration_model/data/file_id_list.scp"
	lines = tryReadFile(target)
	counter = len(lines)
	if counter < 10:
		return [counter-2, 1, 1]
	else:
		train = math.ceil(counter*0.8)
		valid = math.ceil(counter*0.1)
		test = math.floor(counter*0.1)
		together = train+valid+test
		print(together)
		if together > counter:
			train = train - (together-counter)
		if together < counter:
			train = train + (counter-together)
		print(train)
		print(valid)
		print(test)
		return [train, valid, test]

###Functions that run a script or help doing so

def doScriptAction(script):
	global printVariable
	#print("Doing the pre-run action for script:")
	#print(script)
	if script == "merlin_synthesis.sh" or script == "07_run_merlin":
		#set the Play Button to inactive
		playWav.config(bg=cs["inactive"], activebackground=cs["inactive"], command=None)
	if script.find("prepare_conf_files") >= 0:
		saveConfig("gen")
	if script.find("train_duration_model") >= 0:
		printOut(tex["durModelTrainingNotification"])
		saveButtonFunc()
	if script.find("train_acoustic_model") >= 0:
		printOut(tex["acModelTrainingNotification"])
		saveButtonFunc()
	#time.sleep(10)
	return

def runScript(directory, script):
	global queue
	global t3
	if queue.get() == False:
		#kill the running script
		queue.put(True)
		t3.join()
		try:
			#make sure not to kill the next script immediately
			queue.get()
		except:
			True
	#t3 = multiprocessing.Process(target=runScriptThread, args=(directory, script))
	t3 = Thread(target=runScriptThread, args=(directory, script, queue))
	t3.start()

def runScriptThread(directory, script, q):
	global cs
	global playWav
	global printLabel
	global printVariable
	global terminateScript

	#a number of scripts require additional actions before being run
	doScriptAction(script)
	arguments = updateArguments()
	#print(arguments)
	#runs a script with the given directory and collects data about training and validation error
	print("Running script "+directory+script+arguments[script])
	training_dur = []
	training_ac = []
	validation_dur = []
	validation_ac = []
	duration_training = False
	acoustic_training = False
	loop = True
	didNotRun = False
	terminateScript = False
	q.put(terminateScript)
	if script.find("train_duration_model") >= 0:
		duration_training = True
		acoustic_training = False
	if script.find("train_acoustic_model") >= 0:
		acoustic_training = True
		duration_training = False
	
	if script in arguments:
		n = subprocess.Popen("exec sudo ./"+ script + arguments[script], cwd=directory[2:], stdout=PIPE, stderr = subprocess.STDOUT, shell=True) 
	else:
		n = subprocess.Popen("exec sudo ./"+ script, cwd=directory[2:], stdout=PIPE, stderr = subprocess.STDOUT, shell=True)
	#n = subprocess.run("sudo ./"+script, cwd=directory[2:], shell = True, stdout=PIPE, stderr = PIPE)

	while loop == True:
		#Check wether the script needs to be terminated
		try:
			terminateScript = q.get()
		except:
			True
		if terminateScript == True:
			#print("Terminating Script")
			q.put(terminateScript)
			help =  n.stdout.readline()		
			rc = n.poll()
			os.kill(n.pid, signal.SIGINT)
			time.sleep(0.2)
			try:
				#That should do it
				os.kill(n.pid, signal.SIGINT)
				os.kill(n.pid, signal.SIGINT)
				os.kill(n.pid, signal.SIGINT)
			except:
				False
			return
		try:
			q.put(terminateScript)
		except:
			True
		#Now for the real work:
		#output = n.communicate()
		output = n.stdout.readline()
		error = '' #n.stderr.readline().strip().decode()
		if output.strip().decode() == '' and error == '' and n.poll() is not None:
			break
		if output:
			op = output.strip().decode()
			#look for some keywords
			ep = op.find("epoch")
			end = op.find("successfull")
			end2 = op.find("audio files are in")
			didntRun = op.find("Usage")
			if didntRun >= 0:
				didNotRun = True
			if ep >= 0:
				if duration_training:
					i = op.find("validation error")
					j = op.find("train error")
					if i >= 0:
						validation_dur.append(op[i+17:i+23])
					if j >= 0:
						training_dur.append(op[j+12:j+18])
				if acoustic_training:
					i = op.find("validation error")
					j = op.find("train error")
					if i >= 0:
						validation_ac.append(op[i+17:i+23])
					if j >= 0:
						training_ac.append(op[j+12:j+18])
				#print(op)
				#print(validation_dur)
			if end >= 0: 
				
				loop = False
			if end2 >= 0:
				playWav.config(bg=cs["spButton"], activebackground=cs["spButtonAct"], command=playWave)
				playWav.flash()
				loop = False
			if op is not "":
				True			
				print(op)
			
		if error is not "":		
			print(error)
	if validation_dur:
		try:
			updateDiagram("Duration", validation_dur, training_dur)
		except:
			False
	if validation_ac:
		try:
			updateDiagram("Acoustic", validation_ac, training_ac)
		except:
			False

	if didNotRun == False:
		if script.find("setup") >= 0:
			updateConf()
			try:
				confDur.pack_forget()
				confAc.pack_forget()
			except:
				True
		if script.find("conf_files") >= 0:
			updateConf()
		if script.find("0") >= 0 or script.find("1") >= 0:
			fillIndicator(int(script[1]))
	else:
		printOut(tex["somethingWentWrong"])
	print(validation_dur)
	print(training_dur)
	print(validation_ac)
	print(training_ac)
	q.get()
	q.put(True)
	
	try:		
		help =  n.stdout.readline()		
		rc = n.poll()
	except:
		True
	try:
		n.kill()
	except:
		return

###Functions that handle events (like pressing a button)		
def killSwitch():
	global styleVar
	global langVar
	global queue
	global t3
	global terminateScript
	global printLabelVariable
	global textSize
	nimueSettings["lastVoice"] = selectedVoice
	nimueSettings["lastSubVoice"] = selectedSubVoice
	nimueSettings["textSize"] = textSize.get()
	nimueSettings["cs"] = styleVar.get()
	nimueSettings["lang"] = langVar.get()
	try:
		pickle.dump(nimueSettings, open("nimue/nimue_settings.p", "wb"))
	except:
		print("Could not save Nimue's settings")
	try:
		terminateScript = True
		queue.put(terminateScript)
		t3.join()
		time.sleep(3)
		print("Terminating was successfull")
		#t3.terminate()
	except:
		print("If you executed a script, it might still be running.\nCheck your System Monitor for a process called 'Python'.")
	root.quit()
	root.destroy()
	sys.exit()	

def diagramTest(vTest, tTest):
	updateDiagram("Duration", vTest, tTest)	
	updateDiagram("Acoustic", vTest, tTest)

def changeTextSize(a):
	global speechText
	global textSize
	speechText.config(font=fs[textSize.get()])
	speechText.config(width=math.floor(900/int(fs[textSize.get()][1])))
	speechText.config(height=math.ceil(90/int(fs[textSize.get()][1])))

def genWave():
	global selectedVoice
	global selectedSubVoice
	f = open('./' + merlinPath + '/egs/'+selectedVoice+'/s1/experiments/'+selectedSubVoice+'/test_synthesis/txt/nimue.txt', "w")
	tex = speechText.get(1.0, END)
	if tex and tex != "\n":
		f.write(tex)
		f.close()
	else:
		printOut(tex["textNotEmpty"])
		return
	time.sleep(0.3)
	if selectedVoice == "slt_arctic":
		t1 = Thread(target=runScript, args=('./egs/'+selectedVoice+'/s1/', 'merlin_synthesis.sh',))
	elif selectedVoice == "build_your_own_voice":
		t1 = Thread(target=runScript, args=('./egs/'+selectedVoice+'/s1/', '07_run_merlin.sh',))
	else:
		printOut(tex["speakingNotSupported"])
		return
	t1.start()
	True

def playWave():
	
	#playWaveThread()
	global t2
	try:
		t2.terminate()
	except:
		print("Could not terminate t2")
		True
	time.sleep(0.3)
	t2 = multiprocessing.Process(target=playWaveThread, args=())
	t2.start()
	
def playWaveThread():
	global selectedVoice
	global selectedSubVoice
	wave_obj = sa.WaveObject.from_wave_file('./' + merlinPath + '/egs/'+selectedVoice+'/s1/experiments/'+selectedSubVoice+'/test_synthesis/wav/nimue.wav')
	play_obj = wave_obj.play()
	#print("Playing...")
	play_obj.wait_done()
	#print("I'm done playing.")
	del wave_obj
	del play_obj
		
def selectMode():
	global v
	global bottomFrame
	#print(v)
	if v.get() == "0":
		bottomFrame.pack_forget()
	if v.get() == "1":
		bottomFrame.pack(side=BOTTOM, fill=BOTH)

def saveButtonFunc():
	printOut(tex["trySave"])
	try:
		saveConfig("gen")
		saveConfig("dur")
		saveConfig("ac")
		printOut(tex["saveSuccess"])
		
	except:
		printOut(tex["saveError"])
	estimateTime()

def openOptions():
	global OptionsSwitch
	OptionsSwitch.config(command=closeOptions, text=tex["closeOptions"])
	optionsFrame.pack(side=LEFT, padx=10, pady=10, fill=Y)
	
def closeOptions():
	global OptionsSwitch
	OptionsSwitch.config(command=openOptions, text=tex["openOptions"])
	optionsFrame.pack_forget()
	



queue = Queue()
queue.put(True)
voices = voiceFinder()
#print(voices)
 
root.geometry("1400x980+0+0")
#root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
root.title(tex["nimueTitle"])
rootFrame = Frame(root, bg=cs["bg"])
topFrame = Frame(rootFrame, bg=cs["bg"])

buttonFrame = Frame(topFrame, bg=cs["specialBg"], padx=5, pady=5, bd=cs["bdw"], relief=cs["bds"])
killSwitch = Button(buttonFrame, text=tex["killSwitch"], bg=cs["killButton"], highlightbackground=cs["borderColor"], activebackground=cs["killButtonAct"], font=fs["h2"], bd=cs["bdw"], relief=cs["bds"], command=killSwitch )
killSwitch.pack(side=TOP, fill=X, pady=2)
funct = partial(openOptions)
OptionsSwitch = Button(buttonFrame, text=tex["openOptions"], font=fs["button"], highlightbackground=cs["borderColor"], bg=cs["button"], activebackground=cs["buttonAct"], bd=cs["bdw"], relief=cs["bds"], command=funct )
OptionsSwitch.pack(side=TOP, fill=X, pady=2)

selectVoiceLabel = Label(buttonFrame, text=tex["selectVoiceLabel"], bg=cs["specialBg"], font=fs["button"]).pack(fill=X)
voice = ttk.Combobox(buttonFrame, textvariable=voices, font=fs["settingText"])
voice['values'] = voices
voice.bind('<<ComboboxSelected>>', selectVoice)
voice.pack(fill=X, pady=2)
buttonFrame.pack(side=LEFT, padx=10, pady=10, fill=Y)

optionsFrame= Frame(topFrame, bg=cs["specialBg"], padx=5, pady=5, bd=cs["bdw"], relief=cs["bds"])
Button(optionsFrame, text=tex["calibrationButton"], bg=cs["testButton"], highlightbackground=cs["borderColor"], activebackground=cs["testButtonAct"], font=fs["button"], command=timingTest2, bd=cs["bdw"], relief=cs["bds"]).pack(fill=X, pady=0)
selectStyleLabel = Label(optionsFrame, text=tex["selectStyleLabel"], bg=cs["specialBg"], font=fs["button"]).pack(fill=X)
styleVar = StringVar()
styleVar.set(nimueSettings["cs"])
s1 = Radiobutton(optionsFrame, text="Avalon", font=fs["button"], variable=styleVar, value="avalon",  bg=cs["fixButton"], highlightbackground=cs["borderColor"], activebackground=cs["buttonAct"], bd=cs["bdw"], relief=cs["bds"], justify=LEFT, anchor=W)
s2 = Radiobutton(optionsFrame, text="Camelot", font=fs["button"], variable=styleVar, value="camelot",  bg=cs["fixButton"], highlightbackground=cs["borderColor"], activebackground=cs["buttonAct"], bd=cs["bdw"], relief=cs["bds"], justify=LEFT, anchor=W)
s1.pack(anchor=W, fill=X)
s2.pack(anchor=W, fill=X)
selectLangLabel = Label(optionsFrame, text=tex["selectLangLabel"], bg=cs["specialBg"], font=fs["button"]).pack(fill=X)
langVar = StringVar()
langVar.set(nimueSettings["lang"])
l1 = Radiobutton(optionsFrame, text="English", font=fs["button"], variable=langVar, value="en",  bg=cs["fixButton"], highlightbackground=cs["borderColor"], activebackground=cs["buttonAct"], bd=cs["bdw"], relief=cs["bds"], justify=LEFT, anchor=W)
l2 = Radiobutton(optionsFrame, text="Deutsch", font=fs["button"], variable=langVar, value="de",  bg=cs["fixButton"], highlightbackground=cs["borderColor"], activebackground=cs["buttonAct"], bd=cs["bdw"], relief=cs["bds"], justify=LEFT, anchor=W)
l1.pack(anchor=W, fill=X)
l2.pack(anchor=W, fill=X)
selectModeLabel = Label(optionsFrame, text=tex["selectModeLabel"], bg=cs["specialBg"], font=fs["button"]).pack(fill=X)
v = StringVar()
v.set("1")
b1 = Radiobutton(optionsFrame, text=tex["speechMode"], font=fs["button"], variable=v, value="0", command=selectMode, bg=cs["fixButton"], highlightbackground=cs["borderColor"], activebackground=cs["buttonAct"], bd=cs["bdw"], relief=cs["bds"], justify=LEFT, anchor=W)
b2 = Radiobutton(optionsFrame, text=tex["trainingMode"], font=fs["button"], variable=v, value="1", command=selectMode, bg=cs["fixButton"], highlightbackground=cs["borderColor"], activebackground=cs["buttonAct"], bd=cs["bdw"], relief=cs["bds"], justify=LEFT, anchor=W)
b1.pack(anchor=W, fill=X)
b2.pack(anchor=W, fill=X)




speechFrame = Frame(topFrame, bg=cs["bg"])
speechText = Text(speechFrame,  width=80, height=10, bg=cs["settingsBg"], highlightbackground=cs["borderColor"], font=("Courier", 10), bd=cs["bdw"], relief=cs["bds"])
speechText.pack(side=LEFT, padx=30, pady=10)
speechButtons = Frame(speechFrame, bg=cs["bg"])
textSize = ttk.Combobox(speechButtons, font=fs["settingText"])
textSize['values'] = ["Small", "Medium", "Large"]
textSize.bind('<<ComboboxSelected>>', changeTextSize)
textSize.pack(side=BOTTOM, pady=2)
generateWav = Button(speechButtons, text=tex["genWave"], font=fs["button"],  bg=cs["fixButton"], highlightbackground=cs["borderColor"], activebackground=cs["buttonAct"], bd=cs["bdw"], relief=cs["bds"], command=genWave)
playWav = Button(speechButtons, text=tex["playWave"], font=fs["button"], bd=cs["bdw"], relief=cs["bds"], bg=cs["inactive"], highlightbackground=cs["borderColor"], activebackground=cs["inactive"])
generateWav.pack(side=TOP, fill=X, pady=2)
playWav.pack(fill=X, pady=2)
textSizeLabel = Label(speechButtons, justify="left", anchor=W, text=tex["fontSize"], font=fs["button"], bg=cs["bg"])
textSizeLabel.pack(pady=1, fill=X)
speechButtons.pack(side=RIGHT, padx=10)
speechFrame.pack(side=RIGHT, fill=Y)


bottomFrame = Frame(rootFrame, bg=cs["bg"])
confFrame = Frame(bottomFrame, bd=0,  relief=SUNKEN, width=200)
scrollbar = Scrollbar(confFrame)
scrollbar.pack(side=RIGHT, fill=Y)
conf = Text(confFrame,  yscrollcommand=scrollbar.set, width=40, bg=cs["infoText"])
#conf.pack(side=LEFT)
scrollbar.config(command=conf.yview)
confFrame.pack(side=LEFT, padx=1)

commandField = Frame(bottomFrame, bd=2, relief=SUNKEN, bg="goldenrod1")
commandField.pack(side=LEFT)



middleFrame = Frame(rootFrame, bg=cs["bg"])
printFrame = Frame(middleFrame, bg=cs["specialBg"], padx=5, pady=5, bd=cs["bdw"], relief=cs["bds"])
printVariable = ""
printLabel = Label(printFrame, textvariable=printVariable, font=fs["printOut"], bg=cs["settingsBg"], anchor=W, justify=LEFT)
printLabel.pack(side=LEFT)
printFrame.pack(padx=10, pady=5, side=LEFT)

topFrame.pack(side=TOP, fill=BOTH)
middleFrame.pack(side=TOP, fill=BOTH)
bottomFrame.pack(side=BOTTOM, fill=BOTH)
rootFrame.pack(fill=BOTH, expand=1)

#restore last settings
selectVoice(1, useLastVoice=True)
voice.current(voices.index(nimueSettings["lastVoice"]))
selectSubVoice(False, 1, useLastVoice=True)
try:
	#In case the user has selected a voice but didn't select a fitting subVoice before closing.
	subVoiceCombo.current(subVoiceFinder().index(nimueSettings["lastSubVoice"]))
except:
	True
textSize.current(textSize["values"].index(nimueSettings["textSize"]))
changeTextSize(1)
printOut(tex["welcome"])

root.mainloop()
