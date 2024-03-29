## Project Description

### Inspiration
A human learns how to draw with simple shapes and sketching.  At first, we just try to copy it by following the image pixel by pixel and we don’t need a demonstration or a hard-coded drawing steps to achieve this. However, for robots it’s not the case and we would like to democratize art by enabling self-learning for robots.

### What it does
Sliver Maestro is a simulated artistic robot and its expertise is doodling! Sliver Maestro applies sketching by just one look at an image and experiencing how a human would draw it. 

## Model


### How we built it

We used [DeepMind’s Deep Recurrent Attentive Writer model](https://deepmind.com/research/publications/draw-recurrent-neural-network-image-generation) and [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) dataset to generate sequential images and extract stroke movements. 
Draw network is a recurrent auto encoder that uses attention mechanisms. Attention mechanism focus on a small part of the input data in each time step, and iteratively generates image that is closer to the original image. The network is trained with stochastic gradient descent and the
objective function is variational upper bound on the log likelihood of the data. The Quick, Draw! dataset is a collection of 50 million drawings across 345 categories, contributed by players of the game Quick, Draw!. Files are simplified 28x28 grayscale bitmaps in numpy format. 
Our experimental results of DRAW model on Quick Draw! dataset can be accessed from [here](https://github.com/melodiCyb/MSc/blob/master/deep-learning/research-project/draw.pdf).

As shown in the animation below, the advantage of DRAW to  other image generation approaches is that the model generates the entire scene by reconstructing images step by step where parts of a scene are created independently from others, and approximate sketches are successively refined.
 

![test](https://github.com/melodiCyb/neural-networks/blob/master/catdraw.gif)

An illustration of DRAW's refinements which is obtained by subtracting consecutive images 

![drawpostprocess](https://github.com/melodiCyb/sliver-maestro/blob/master/gifs/postprocessed_draw.gif)


We simulated drawings by using both PyGame and a robot simulation environment, CoppeliaSim simulator with Baxter robot
 model and Remote API for Python. In the post-processing, we first convert outputs into binary images and  then into svg files. We use an svg parser to convert into coordinates
 and Sliver Maestro draws the generated images with successive refinements provided by the model.

## Simulation 
 **Final simulation**

![bankstergif](https://github.com/melodiCyb/sliver-maestro/blob/master/gifs/generated.gif)


**Raw data sample simulation**

![drawgif](https://github.com/melodiCyb/baxter-drawing/blob/master/baxter_ws/baxter_drawing_cat.gif)



## Simulation Requirements

* Ubuntu 18.04 (Recommended)
* [CoppeliaSim 4_0_0](https://coppeliarobotics.com/previousVersions)

## How to run

### Initial Setup
1. Clone the repo and cd into it:
        
       git clone https://github.com/melodiCyb/sliver-maestro.git
       cd sliver-maestro
      
2. Setup the environment Install the requirements:

       export PYTHONNOUSERSITE=True
       conda env create -f environment.yml
       
3. Re-start the terminal and activate the environment:

       export PYTHONNOUSERSITE=True
       conda activate sliver-maestro
       cd src
      
### Download Quick, Draw! dataset

4. Download data
        
        gsutil -m cp gs://quickdraw_dataset/full/numpy_bitmap/cat.npy ./data/input/cat/cat.npy
        gsutil -m cp gs://quickdraw_dataset/full/raw/cat.ndjson ./data/raw/cat/cat.ndjson

### Deep Recurrent Attentive Writer

5. Train model (Optional) 

       python3 train.py --phase train --category cat
       
6. Generate image sequence 
    
       python3 generate_images.py --category cat --idx 1 --phase train

You can see the created output images in the directory ~/sliver-maestro/src/data/output/images    

### CoppeliaSim Simulation


7. Start another terminal and run:

       ./coppeliaSim.sh
       
8. After the simulation UI starts upload sliver maestro scene and run:
 
       cd sliver-maestro
       catkin_make
       source devel/setup.bash
       cd src
       
9. Generate drawing coordinates for the simulated robot
 
       python3 postprocess.py --category cat --idx 1
        
   
10. Run robot simulation for the generated drawing sequence

        python3 drawer.py 
       
11. Run robot simulation for a human drawing 
        
        python3 drawer.py --raw True

### PyGame Simulation
       
12. Run PyGame for the generated drawing sequence
        
        python3 pgame_runner.py
        
13. TODO: add PyGame raw version 

Directory layout:

        .
        ├── gifs        
        │   ├── cat_drawing.gif
        ├── notebooks
        |   └── draw_cat.ipynb
        |   └── draw_moon.ipynb
        │── src
        │   ├── data         
        |       └── input
        |       └── output
        |           └── images
        |           └── positions
        |       └── raw      
        |   ├── save
        |   ├── simulation
        |       └── sliver-maestro.ttt
        |    ├── utils
        |       └── im_utils.py
        |       └── model_utils.py
        │       └── vrep.py
        │       └── vrepConst.py
        │   └── config.cfg
        │   └── draw_model.py
        |   └── drawer.py
        |   └── generate_images.py
        |   └── pgame_runner.py
        |   └── postprocess.py
        |   └── train.py
        │── environment.yml
        └── Readme.md
        
## Notebooks
 You can see walkthrough examples in the notebooks folder for [cat](https://github.com/melodiCyb/sliver-maestro/blob/master/notebooks/draw_cat.ipynb) and [moon](https://github.com/melodiCyb/sliver-maestro/blob/master/notebooks/draw_moon.ipynb) datasets. 
## References
1. PyTorch implementation of Deep Recurrent Attentive Writer model was modified from [this repo](https://github.com/chenzhaomin123/draw_pytorch)
