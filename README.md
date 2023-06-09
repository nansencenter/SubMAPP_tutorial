# SubMAPP_tutorial

The goal of the SubMAPP package is to train a machine-learning based (ProfHMM) package to infer vertical profile
data with satellite surface data as inputs. In our application (and so this tutorial), vertical profile is for 
chlorophyll-a (CHL-A) and satellite data is composed of CHL-A and several variables linked to 
CHL-A like SST, MLD, I_0, SLA. 
The model depends on several parameters tuned during training part and should be trained for a specific ocean area.  

The goal of this workshop is to learn how to use ProfHMM model.

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Removing](#remove)

### Folder contents
data  # all the numpy.array data used for the tutorial
	info_prof_training.yml  # data descriptor for train_prof_2019_2020.npy
	info_prof_prediction.yml # data descriptor for truth_prof_2019_2020.npy
	info_surf_training.yml # data descriptor for train_surf_2019_2020.npy
	info_surf_prediction.yml # data descriptor for infer_surf_2019_2020.npy
    
figs  # a folder to save figures in with the tutorial
objects  # for saving SOM, HMM and MAP objects during tutorial
Profhmm_prediction.ipynb  # tutorial to infer vertical profile based on a trained HMM and new surface data 
Profhmm_training.ipynb  # tutorial to train both SOM and HMM
submapp  # contains all basic code for submapp package
tools # contains all code for using submapp package

# How to setup environment

## Technologies
Project is created with:
* Python 3.6
* With Miniconda for virtual environment (see https://docs.conda.io/en/latest/miniconda.html for the good link for you
system)
You will find help below to install Miniconda if you don't have.

## Setup for Linux
If you have already miniconda3 skip steps 2) and 3)
1) Open a terminal in the Linux_install folder
2) Run if you don't have already miniconda3, you should install it. You should use the provided script and adapt it for
your computer (depending on 64bits or not, aarch64 or ppc64le), so adapt the miniconda link in the script for your system 
(there is a default value that works on many system). Once it has been done, run bash -l install_miniconda.sh
3) Reset your shell for changes, by running in the shell : source ~/.bashrc
4) Run the script to install packages : bash -l init_tutorial.sh
	   
	   
It will open automatically the jupyter notebook to access tutorials.
If you close this jupyter session, just run in a NEW shell : jupyter notebook
to get back. If you close the shell, reactivate the conda virtual environment with : conda activate SOM_env

## Setup for Windows
1) Install miniconda3 (skip if you already have) with miniconda3.exe, you should create a miniconda3 folder in workshop 
folder  and then precise directory folder in the exe installer, it would be easier to uninstall.
2) Open : Anaconda Powershell Prompt (miniconda3) 
(search for it on the search menu of your task bar)
3) To go in the windows install directory, run in the anaconda powershell :
cd REPLACE_WITH_DIRECTORY\workshop\Windows_install 
(for example on my PC : cd C:\Users\Administrateur\Documents\workshop\Windows_install)
4) Run in the anaconda powershell :  .\WIN_init_anaconda_prompt.bat
5) Reactivate the conda virtual environment, run in the anaconda powershell : conda activate SOM_env
6) Then, start a jupyter session, run in the anaconda powershell : cd .. 
																   jupyter notebook
## Setup for macOS
If you have already miniconda3 skip steps 2) and 3)

1) Open a terminal in the Mac_os_install folder
2) If you don't have already miniconda3, you should install it. You should use the provided script and adapt it for
your computer (depending on 64bits or not), so adapt the miniconda link in the script for your system 
(there is a default value that works on many system). Once it has been done, run : bash -l install_miniconda.sh
3) Reset your shell for changes, by running in the shell : source ~/.bashrc
4) Run the script to install packages : bash -l init_tutorial.sh
	   
	   
It will open automatically the jupyter notebook to access tutorials.
If you close this jupyter session, just run in a NEW shell : jupyter notebook
to get back. If you close the shell, reactivate the conda virtual environment with : conda activate SOM_env

## Removing
To remove the virtual environment, run in your shell : conda deactivate SOM_env
													   conda remove -n SOM_env --all

To remove miniconda3:
	- On Linux, open a terminal in the folder where miniconda3 is : rm -rf miniconda3
	- On Windows, run the uninstall.exe in the miniconda3 folder
	- On macOS, remove the folder miniconda3/, by running in a terminal : rm -rf ~/opt/miniconda3

Rem : for both Linux and macOS, there is still a path added to your .bashrc to use the "conda" commande. You may need,
to delete these lines in .bashrc to be sure that in a future installation you won't have any conflicts.

# How to run tutorials
First run "Profhmm_training.ipynb" for training ProfHMM given dataset under the folder "data" and saving SOM and HMM objects under the folder "object". Then run "Profhmm_prediction.ipynb" to infer subsurface profile given surface data under the folder "data" and to compare the inferred profile against true profile.

When your jupyter notebook session is up:
1) Click on the "tutorial_submapp" folder
2) Then click on the "Profhmm_training.ipynb" tutorial
3) Then click on the "Profhmm_prediction.ipynb" tutorial
