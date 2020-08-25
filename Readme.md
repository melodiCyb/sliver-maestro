## Project Description

[Hackathon project for Global PyTorch Summer Hackathon 2020](https://pytorch2020.devpost.com/)

### Inspiration
A human learns how to draw with simple shapes and sketching.  At first, we just try to copy it by following the image pixel by pixel and we don’t need a demonstration or a hard-coded drawing steps to achieve this. However, for robots it’s not the case and we would like to democratize art by enabling self-learning for robots.

### What it does
Sliver Maestro is a simulated artistic robot and its expertise is doodling! Sliver Maestro generates the required stroke movements by just one look at an image.  

## Model

### Data

Dataset Description: The Quick Draw Dataset is a collection of 50 million drawings across 345 categories, contributed by players of the game Quick, Draw!. Files are simplified 28x28 grayscale bitmaps in numpy format.


### How we built it

We used [DeepMind’s Deep Recurrent Attentive Writer model](https://deepmind.com/research/publications/draw-recurrent-neural-network-image-generation) and [Quick! Draw](https://github.com/googlecreativelab/quickdraw-dataset) dataset to generate sequential images and extract stroke movements. As shown in the animation below, the advantage of DRAW to  other image generation approaches is that the model generates the entire scene by reconstructing images step by step where parts of a scene are created independently from others, and approximate sketches are successively refined. 

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



## Requirements
* Python >= 3.5

Simulation
* Ubuntu 18.04
* CoppeliaSim 4_0_0 
* Remote API
* vrep_pkgs



## How to run

1. Clone the repo and cd into it:
        
       git clone https://github.com/melodiCyb/sliver-maestro.git
       cd sliver-maestro
      
2. Setup the environment Install the requirements:

       conda create --name sliver-maestro 
       conda activate sliver-maestro
       pip install -r requirements.txt

3. Start another terminal and run:

       ./coppeliaSim.sh
       
4. After the simulation UI starts upload bankster scene and run:
 
       cd sliver-maestro
       catkin_make
       source devel/setup.bash
       cd src
       
5. Train model (Optional)
       
       python draw_model.py 
       
6. Use a pre-trained model (Optional): pre-trained models are located in notebooks/model folder with options:
        * cat_draw.pth
        * draw.pth (moon_draw.pth)
        
       
6. Generate drawings using samples
        
       python postprocess.py
       
7. Run robot simulation

       python drawer.py
       
8. Run PyGame 
        
        python pgame_runner.py
        
* You can see the created output images in the directory ~/sliver-maestro/src/data/output/images


Directory layout:

        .
        ├── gifs        
        │   ├── cat_drawing.gif
        ├── notebooks
        |   └── draw_model.ipynb
        |   └── postprocessing.ipynb
        │── src
        │   ├── data         
        |       └── input
        |           └──full_numpy_bitmap_cat.npy     
        |       └── output
        |           └── images
        |                └── datacat10206455_0.png
        |                └── .
        |                └── .
        |           └── positions
        |                └── final_motion.csv
        |                └── .
        |   ├── simulation
        |       └── sliver-maestro.ttt
        |    ├── utils
        |       └── dataset.py
        |       └── im_utils.py
        |       └── model_utils.py
        │       └── vrep.py
        │       └── vrepConst.py
        │   └── postprocess.py
        |   └── draw_model.py
        |   └── drawer.py
        └── Readme.md
        

