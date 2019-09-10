# Essil Visualisation

Repo that contains the web code for generating the ESSIL visualisation from an input .CSV datafile and the associated .MOV file. All files refer to the HTML/JS/CSS code for generating the website visualisation except for the code found in `python_scripts`. The code in `python_scripts` is the python source files that perform inference on a data log file to produce the period segmentation and the video transformation of the data.

## Website
- idex.html - entrypoint
- js - javascript code
- css - css code for styling

## Python Code
Entrypoint: `hdp_scripts.py`. This is the main file that calls all necessary dependencies. You run the script as follows:
```$xslt
python hdp_scripts.py -IL {path to csv file} -IM {path to mov file} -O {path to output folder}
```

Flags `-IL`, `-IM` and `O` refer to "Input Log File", "Input Movie File" and "Output File" respectively. The output of this script will produce a `model_output.csv` file that is exactly the file that is used for input in the `js/main.js` webpage (i.e., if you copy this `model_output.csv` to `data/input_file.csv` then the webpage is updated). The `output_video.mp4` is exactly the video file (in the expected format) that the website expects (i.e., you need to edit `index.html` to refer to the URI of this video file).

### Dependencies/Installation
Tested on `Ubuntu 18.04` and `MacOS Mojave`.

Most of the required dependencies (except Open CV) come with the [Anaconda](https://www.anaconda.com/) distribution of python. Once you are using anaconda, setup a new virtual environment `essil` and install python 3.6
```
conda create -n essil python=3.6 anaconda
source active essil
```

You will then need to `pip install` the Open CV libarary that is use to read the global view video and do the processing to transfrom this video:
```
pip install opencv-python
```

You will probably need to install `ffmpeg` on both Mac and Ubuntu. I used homebrew on Mac:
```
brew install ffmpeg
```

**The following was only to solve a problem on Ubuntu (not on Mac):**

You may have a problem with the `ffmpeg` video writer that creates the global view video (from the one provided by CW). I had to reinstall the one on my computer:
```
sudo apt-get remove ffmpeg
sudo apt-get install ffmpeg
```

I also had to down-grade my version of `matplotlib` for compatibility with this `ffmpeg` software:
```
pip uninstall matplotlib
pip install matplotlib==3.0.3
```
