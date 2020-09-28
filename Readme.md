## Project Description

[Hackathon project for Global PyTorch Summer Hackathon 2020](https://pytorch2020.devpost.com/)

### Inspiration
A human learns how to draw with simple shapes and sketching.  At first, we just try to copy it by following the image pixel by pixel and we don’t need a demonstration or a hard-coded drawing steps to achieve this. However, for robots it’s not the case and we would like to democratize art by enabling self-learning for robots.

### What it does
Sliver Maestro is a simulated artistic robot and its expertise is doodling! Sliver Maestro applies sketching by just one look at an image and experiencing how a human would draw it. 

## Model


### How we built it

We used [DeepMind’s Deep Recurrent Attentive Writer model](https://deepmind.com/research/publications/draw-recurrent-neural-network-image-generation) and [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) dataset to generate sequential images and extract stroke movements. The Quick, Draw! dataset is a collection of 50 million drawings across 345 categories, contributed by players of the game Quick, Draw!. Files are simplified 28x28 grayscale bitmaps in numpy format. As shown in the animation below, the advantage of DRAW to  other image generation approaches is that the model generates the entire scene by reconstructing images step by step where parts of a scene are created independently from others, and approximate sketches are successively refined. 

![test](https://github.com/melodiCyb/neural-networks/blob/master/catdraw.gif)

In the post-processing, we first convert outputs into binary images and  then into svg files. We use an svg parser to convert into coordinates and bankster draws the generated images with successive refinements provided by the model.


Recommended svg converter: [Link](https://image.online-convert.com/convert-to-svg)


* After diffs of Draw outputs

![drawpostprocess](https://github.com/melodiCyb/sliver-maestro/blob/master/gifs/postprocessed_draw.gif)

## Simulation 
* Final simulation

![bankstergif](https://github.com/melodiCyb/sliver-maestro/blob/master/gifs/bankster.gif)


* Raw data sample simulation

![drawgif](https://github.com/melodiCyb/baxter-drawing/blob/master/baxter_ws/baxter_drawing_cat.gif)



## Simulation Requirements

* Ubuntu 18.04
* CoppeliaSim 4_0_0 
* Remote API
* vrep_pkgs



## How to run

### Initial Setup
1. Clone the repo and cd into it:
        
       git clone https://github.com/melodiCyb/sliver-maestro.git
       cd sliver-maestro
      
2. Setup the environment Install the requirements:

       export PYTHONNOUSERSITE=True
       conda env create -f environment.yml

### Deep Recurrent Attentive Writer

TODO: add download data step

3. Train model (Optional) 
       
       python3 train.py --phase train --category cat
        
4. Generate image sequence 
    
       python3 generate_images.py --category cat

You can see the created output images in the directory ~/sliver-maestro/src/data/output/images    

### CoppeliaSim Simulation


5. Start another terminal and run:

       ./coppeliaSim.sh
       
6. After the simulation UI starts upload bankster scene and run:
 
       cd sliver-maestro
       catkin_make
       source devel/setup.bash
       cd src
       
7. Generate drawing coordinates for the simulated robot

       python3 postprocess.py
       
8. Run robot simulation

       python3 drawer.py

### PyGame Simulation
       
9. Run PyGame 
        
        python3 pgame_runner.py
        

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
        
        
## Resources
1. PyTorch implementation of Deep Recurrent Attentive Writer model was modified from [this repo](https://github.com/chenzhaomin123/draw_pytorch)
