# Databricks notebook source
import pandas as pd
import numpy as np
import os

# COMMAND ----------

# hypers

GRANULARITY = "fps"
dataDir = "/dbfs/mnt/S3_rtl-databricks-datascience/datasets/EEG-data/"

os.listdir(dataDir)

# COMMAND ----------

def toLongitudinal(d):

  # patient meta data (rows 1-23)

  c = ["patient", "age", "gender"]

  nrows = d.index[-1] # there are built-in row numbers in the data

  m = pd.DataFrame(np.zeros((nrows, len(c))), columns = c)

  m.patient = d.loc["#Respondent Name", "Unnamed: 1"]
  m.age = d.loc["#Respondent Age", "Unnamed: 1"]
  m.gender = d.loc["#Respondent Gender", "Unnamed: 1"]
  # m.group = d.loc["#Respondent Group", "Unnamed: 1"]

  # experiment data

  firstRow = np.where(d.index.values == "Row")[0][0]
  e = d.iloc[firstRow:,]
  e.columns = e.iloc[0]
  e = e.drop(e.index[0])
  e.reset_index(drop=True, inplace=True)

  D = pd.concat([m, e], axis = 1)
  
  return(D)

# COMMAND ----------

def formatAndClean(d):
  # format data
  d["event"] = d.EventSource.replace(np.nan, '', regex=True).apply(''.join, axis=1)

  d["Timestamp"] = d.Timestamp.astype("float")

  # useless columns
  d.drop(["EventSource", "SampleNumber",	"Epoch"], axis = 1, inplace = True)

  # eye tracker remove -1's for calculating the mean
  eyeColumns = d.columns[np.where(d.columns == "ET_GazeLeftx")[0][0] : np.where(d.columns == "ET_ValidityRight")[0][0] + 1]
  d[eyeColumns] = d[eyeColumns].replace("-1", np.nan)
  d[eyeColumns] = d[eyeColumns].replace(-1.0, np.nan)

  return(d)
# D[(D["SlideEvent"] == "StartSlide") | (D["SlideEvent"] == "EndSlide") | (D["SlideEvent"] == "StartMedia") | (D["SlideEvent"] == "EndMedia")]

# COMMAND ----------

def setGranularity(d):    
  i = 0
  granularity = []
  prevGran = 0.0
  fps = 25
  atRest = True

  if GRANULARITY == "eyetracker":
    for g, s in zip(d.ET_GazeLeftx, d.SlideEvent): # for e in D.event:
      if float(g) > 0 or s == "StartSlide": # != "Eyetracker Tobii X3 @120Hz":
        i += 1
        granularity += [i]
      else:
        granularity += [i]    

  if GRANULARITY == "fps":
    for t, s in zip(d.Timestamp, d.SlideEvent): # for e in D.event:
      if s == "EndMedia":
        atRest = True
      if s == "StartMedia":
        atRest = False
      if atRest == False and (float(t) > (prevGran + 1000 * 1/fps) or s == "StartMedia"):
        prevGran = float(t)
        i += 1
        granularity += [i]
      elif atRest == True:
        granularity += [-1]
      else:
        granularity += [i]

  d["granularity"] = granularity
  
  return(d)

# COMMAND ----------

def labelDuplicates(d):
  # label duplicate columns: contaminated VS decontaminated data on electrodes (F1, F2, ...)
  cols = []
  count = [1] * len(d.columns)
  for i, column in enumerate(d.columns):
      if column in cols:
          cols.append(column + "_" + str(count[i]))
          count[i]+=1
          continue
      cols.append(column)
  cols
  d.columns = cols
  return(d)

# COMMAND ----------

def setColumnSummary(d):
  # determine which columns are to be summarized and how 

  if GRANULARITY == "eyetracker":
    meanColumns = d.columns[np.where(d.columns == "High Engagement")[0][0] : np.where(d.columns == "P4_1")[0][0] + 1]
    firstColumns = d.columns[np.where(d.columns == "patient")[0][0] : np.where(d.columns == "SourceStimuliName")[0][0] + 1].to_list() + d.columns[np.where(d.columns == "ET_GazeLeftx")[0][0] : np.where(d.columns == "Frontal Asymmetry Alpha")[0][0] + 1].to_list()

  if GRANULARITY == "fps":
    meanColumns = d.columns[np.where(d.columns == "High Engagement")[0][0] : np.where(d.columns == "Frontal Asymmetry Alpha")[0][0] + 1]
    firstColumns = d.columns[np.where(d.columns == "patient")[0][0] : np.where(d.columns == "SourceStimuliName")[0][0] + 1].to_list()
  
  return(meanColumns, firstColumns)

# COMMAND ----------

D = pd.DataFrame()
failed = []

for f in os.listdir(dataDir + "Sensor Data/"):
  print(f)
  d = pd.read_csv(dataDir + "Sensor Data/"+ f, index_col=0)
  d = toLongitudinal(d)
  try:
    d = formatAndClean(d)
    d = setGranularity(d)
    d = labelDuplicates(d)
    meanColumns, firstColumns = setColumnSummary(d)

    # aggregate
    d[meanColumns] = d[meanColumns].astype("float")
    A = d.groupby("granularity").agg({**{i: 'first' for i in firstColumns}, **{i: 'mean' for i in meanColumns}})
    print(A.head())
    D = pd.concat([D, A])
  except:
    failed += [f]

# COMMAND ----------

D.to_csv(dataDir + "data@granularity/" + "perFrame.csv", index = False)

# COMMAND ----------

# MAGIC %md
# MAGIC # notes
# MAGIC 
# MAGIC ## emotions and decontaminated EEG
# MAGIC 
# MAGIC Blackbox models
# MAGIC 
# MAGIC ## Frontal Asymmetry Alpha
# MAGIC 
# MAGIC Frontal Asymmetry Alpha = ln(Alpha Left (F3)) - ln(Alpha Right (F4))
# MAGIC (related to desire)
# MAGIC 
# MAGIC ## rare events (1/2 s)
# MAGIC 
# MAGIC EventSource	ln(Alpha Left (F3))	ln(Alpha Right (F4))	Frontal Asymmetry Alpha
# MAGIC High Engagement	Distraction	Drowsy	Workload FBDS	
# MAGIC 
# MAGIC ## Timestamps
# MAGIC - EEG: 256 times (rows) per second
# MAGIC - eye tracker: 120 times (rows) per second
# MAGIC 
# MAGIC ## SLide Event
# MAGIC 
# MAGIC video start time
# MAGIC 
# MAGIC ## Eye tracking
# MAGIC 
# MAGIC ## EEG data (Brain state)
# MAGIC 999 -> NaN
# MAGIC 
# MAGIC ## Frontline Asymmetry
# MAGIC metric that correlate with emotions
# MAGIC 
# MAGIC ## EEG raw data
# MAGIC 
# MAGIC contaminated on the left
# MAGIC decontaminated on the right
# MAGIC 
# MAGIC - EventSource: eyetracking / EEG: determines what the row is about
# MAGIC - SLideEvent: "StartMedia" is when the video starts
# MAGIC - StimType: ignore it
# MAGIC - Duration: total duration
# MAGIC - StimuliDisplay: to ignore
# MAGIC - samplenumber:
# MAGIC - high Engagement: EEG, amount of attention that people devote to the stimulous. (preprocessed electrodes activations + regression)
# MAGIC 
# MAGIC - F3 & F4: asymetry between left and right frontal lobe (alpha frequency, 8-12 Hz)
# MAGIC 
# MAGIC people saw the trailers in different orders
# MAGIC 
# MAGIC ## email from Nikki
# MAGIC 
# MAGIC 
# MAGIC - Slide Events: startslide means the start command is sent to the media player, startMedia is the confirmation that the player has started. So that way you can verify when the video started. The others you can ignore
# MAGIC - B-Alert BrainState encompasses three pre-calculated metrics: Engagement which we talked about, Workload (prediction about the difficulty in processing) and distraction (prediction about distraction of participant).
# MAGIC - Then B-Alert Decontaminated EEG (microVolts) we talked about yesterday, where data is cleaned. I attached an image of what the EG looks like and where each electrode is placed on the head.
# MAGIC B-Alert EEG which is the raw EEG data  (microVolts)
# MAGIC - Eyetracker Tobii X3 @120Hz which was missing in the previous exports. Here you find the coordinates on the x and y axis for both eyes separately (pixels). It also includes both pupil sizes (mm) and distance to the screen (mm). There are also columns with camera which have put the coordinates of the eye on the screen relatively to the size of the screen (so pixels/max pixels) for each eye and axis. At last, validity implies whether the gaze was measured. As you can see, when validity=4, there's -1 on the other metrics. This indicates that the eyes were not measured. When validity=0, the gaze was measured correctly.
# MAGIC - The last eventsource is Processed and imported data Frontal Asymmetry (ABM Raw). This contains Frontal Asymmetry Alpha which calculated with the two columns before that: ln(Alpha Left (F4)) -  ln(Alpha Left (F3)). 
# MAGIC 
# MAGIC - screen resolution 1920 x 1080
# MAGIC 
# MAGIC - TOBii yetracker has up to 11ms latency

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # ressources
# MAGIC 
# MAGIC ## unravelResearch
# MAGIC 
# MAGIC - https://www.unravelresearch.com/docs/whitepapers/filmtrailer-rapport.pdf
# MAGIC 
# MAGIC ## eye-tracking
# MAGIC 
# MAGIC 
# MAGIC - measure sacccades in R: https://github.com/tmalsburg/saccades
# MAGIC - eye tracking path in R: https://github.com/tmalsburg/scanpath
# MAGIC - *heatmap in R* : https://stackoverflow.com/questions/16118566/creating-heatmap-with-r-with-eye-tracker-data
# MAGIC - fixation classification in Python: https://github.com/jonathanvanleeuwen/I2MC---Python / https://github.com/mathDR/BP-AR-HMM (more sophisticated but Python 2)
# MAGIC - run experiment in Python: https://www.psychopy.org/index.html
# MAGIC - *heatmap in python* : https://github.com/takyamamoto/Fixation-Densitymap
# MAGIC - proprietary software: https://www.sr-research.com/data-viewer/
# MAGIC - others: GraFix (C++)
# MAGIC - https://link.springer.com/article/10.3758/s13428-017-0974-7
# MAGIC 
# MAGIC ## EEG
# MAGIC 
# MAGIC - https://mne.tools/dev/index.html (python package)
# MAGIC - https://github.com/sari-saba-sadiya/EEGExtract (EEG feature extraction in Python)
# MAGIC 
# MAGIC ## EEG + Eye-tracking
# MAGIC 
# MAGIC - https://www.frontiersin.org/articles/10.3389/fnhum.2012.00278/full
# MAGIC - http://www2.hu-berlin.de/eyetracking-eeg/ (MATLAB package)
# MAGIC - EEG and Art: https://www.frontiersin.org/articles/10.3389/fpsyg.2018.01972/full
# MAGIC 
# MAGIC ## datasets
# MAGIC 
# MAGIC - https://figshare.com/articles/dataset/Loughborough_University_Multimodal_Emotion_Dataset_-_2/12644033
# MAGIC 
# MAGIC ## misc
# MAGIC 
# MAGIC - https://arxiv.org/pdf/2012.01074.pdf (attention based quality of EEG)
# MAGIC - https://www.researchgate.net/profile/Alessio_Zanga/publication/339907287_An_Attention-based_Architecture_for_EEG_Classification/links/5ec70810a6fdcc90d68c8ee4/An-Attention-based-Architecture-for-EEG-Classification.pdf (attention-based)
# MAGIC - https://www.mdpi.com/1660-4601/17/11/4152/pdf (dreams)

# COMMAND ----------

# MAGIC %md
# MAGIC # research questions
# MAGIC 
# MAGIC ## visualization track
# MAGIC 
# MAGIC - vizualize eyetracking (e.g. heatmap)
# MAGIC   - Goal: vizualize where the average gaze is on every frame with a heatmap
# MAGIC   - Use: help trailer producers see what elements captivated attention
# MAGIC 
# MAGIC - vizualize EEG
# MAGIC   - Goal: visualize EEG (probably with Python MNE)
# MAGIC   - Use: towards "reaction to stimuli"
# MAGIC 
# MAGIC - locate emotions on image (4 basic emotions or based on stimuli above)
# MAGIC   - Goal: Identify emotional parts of images
# MAGIC   - Use: A way to improve/evaluate emotional content for designers
# MAGIC   
# MAGIC - link with youtube trailer data (viewing numbers, ev. over time)
# MAGIC   - Goal: find the original Youtube trailers and visualize social network stats in comparison with emotions / EEG activations etc.
# MAGIC   - Use: link popularity with emotions / stimuli on the image
# MAGIC 
# MAGIC - play around with MNE
# MAGIC   - Goal: create different visualizations with the MNE Python package
# MAGIC   - Use: research-driven
# MAGIC 
# MAGIC ## Inference track
# MAGIC 
# MAGIC - reaction to stimuli (e.g. jump scare)
# MAGIC   - Goal: identify different punctual stimuli (jump scare, voluptuous moment, violent act) and identify a pattern in EEG reaction to it
# MAGIC   - Use: towards "locating emotions on an image" task
# MAGIC 
# MAGIC - EEG activations given image
# MAGIC   - Goal: identify EEG patterns given particular images
# MAGIC   - Use: research-driven
# MAGIC 
# MAGIC - relationship to image/video of the trailer with object recognition
# MAGIC   - Goal: Identify salient/reoccurring objects from image and trailer
# MAGIC   - Use: From a trailer/movie identify which objects should appear on an image to have most emotional impact (?)
# MAGIC 
# MAGIC - transfer learning for models trained on EEG with different amount of channels
# MAGIC   - Goal: Allow the same model to deliver inference on EEG data with different amounts of electrodes
# MAGIC   - Use: research-driven
# MAGIC   
# MAGIC - fixation length (eye-tracking) <-> emotions
# MAGIC   - Goal: establish a relationship between emotions and fixation
# MAGIC   - Use: understand what kind of content and emotions makes focusses the viewer's optical attention

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Iris Groen
# MAGIC 
# MAGIC fmri only has a seconds granularity but is better for spacial relationship
# MAGIC 
# MAGIC distribution of power accross low and high frequencies (take a little window over time)
# MAGIC 
# MAGIC ~50ms for info to flow eye to cortex!

# COMMAND ----------

# MAGIC %md
# MAGIC # todo
# MAGIC 
# MAGIC ask if there is an at-rest measurement
# MAGIC 
# MAGIC subtract ECG behind the ear to all signals: ask if it hasn't been done
# MAGIC 
# MAGIC ## sanity checks
# MAGIC - check if there is a stimuli in EEG (especially for pOz) at a new trailer por at a new scene
# MAGIC - one eyetracker plot
# MAGIC 
# MAGIC summarize data at 60Hz, i.e. at frame level (sometimes the video drops some frames)

# COMMAND ----------

# MAGIC %md
# MAGIC Event related Potential (rather for images than videos): event started (probably not clean enough)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC TODO
# MAGIC 
# MAGIC distance from screen (degree)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC EEG + eyetracking: https://repositorium.ub.uni-osnabrueck.de/handle/urn:nbn:de:gbv:700-20181116806  
# MAGIC eyetracking: https://repositorium.ub.uni-osnabrueck.de/handle/urn:nbn:de:gbv:700-202011053658

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # datasets
# MAGIC 
# MAGIC - [EEG + eye-tracking] eye-tracking granularity (~120Hz)
# MAGIC - [EEG + eye-tracking] frame rate granularity (25Hz)
# MAGIC - [EEG + eye-tracking + powerspectral density (black box decontamination)] preprocessed granularity (4Hz)
# MAGIC - [EEG + eye-tracking] raw data granularity (256Hz)

# COMMAND ----------

# MAGIC %md
# MAGIC #todo
# MAGIC - map datasets to questions
# MAGIC - check if joining 4Hz with others is easy
# MAGIC - check if each dataset load well in MNE