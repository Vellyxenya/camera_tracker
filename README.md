# Digit-board tracker

Scripts to track and analyze measurements from a 5x7 lcd display

## Requirements

Start by setting up a python 3.7 environment. 
To install the requirements, run:

```
pip install -r requirements.txt
```

## Run the tracker

To run the tracker, execute:

```commandline
python main.py
```

This fires up a window with sliders. Adjust the sliders so as to only capture the lcd display. Please make the bounds
as tight as possible for maximum precision. In my experiments, setting the display closer to the camera works better.

As the script is running, you should see decoded lines displayed on the terminal. You may want to adjust the 
`dilations` parameter in some cases. You may also need to adjust the `character_width` parameter. Setting this number
too large will cause artifacts such as adjacent bounding boxes to be merged together, and setting it too small may cause
the decoder to hallucinate unexisting characters.

Note that the bounding boxes are displayed at the bottom of the window to help adjust the parameters.

*IMPORTANT:* To properly terminate the tracker and save the collected data, you should close the tracking window
by pressing `q` on the application. Otherwise the data may not be saved for subsequent analysis.

## Analyze the data

Once the tracking data are saved, you can execute the analysis script by running:

```commandline
python analyze.py
```

This will impute missing values, remove outliers and display the cleaned collected data as graphs.
