# Auto Labeling of GARMIN sensory data

With this code, an auto labeled signal data set on Garmin .fit files can be created.
This has all been implemented for the use of .fit files generated with a Garmin Fenix 5 or newer utilizing the App [RawLogger](https://github.com/cedric-dufour/connectiq-app-rawlogger) by Cedric Dufour.


With regular .fit files, the code needs some rewriting, since there is no 25Hz acceleration data within the files.
Removing the values from the array of columns in "mtb_data_provider_garmin.py" might be sufficient though.

In *notebooks/MtbDataSet.ipynb*, an example is given.
The function ```mtb_data_set.create_data_set``` encapsulates the core functionality and needs either an array of file names in the "/data" folder or a "None", which leads to reading all .fit files in the data folder.
.fit filenames are split into three parts.

```mytracking_w_1```
This leads to the trailname "mytracking", the gender "w" and the rider ID "1", which will be added to the dataset.

For now, the rest has to be read from code, better documentation should be coming soon.

Requirements
* You need the FitSDK to be able to convert .fit files (This project uses FitSDKRelease_21.16.00 in the root folder)

For Gopro Mp4 Metadata To CSV:
* You need node and npm
* Install the packages in package.json
* Install ffmpeg and gpmd2csv on your system
**However, be aware, that the sync of GARMIN and Gopro data has not been tested well, since we found the Gopro to have inaccurate GPS**
