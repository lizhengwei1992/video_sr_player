# Video_SR_Player

Video Player including SR algorithm. You can view a video with Super Resolution(x3) or original Resolution  in real time.


# Installation

## Requirements
- python3.5 / 3.6
- pyTorch >= 0.2
- opencv >= 3.0
- PyQt5
- numpy

## Installation 

This Video Player GUI builds with PyQt5 in python 3.5 environment.(ubuntu 16.04, GPU:1070ti)

- ### numpy

    pip3 install numpy

- ### pytorch :

    pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl

    pip3 install --no-deps torchvision

- ### opencv (or make from original code) :

    pip3 install opencv-python

- ### PyQt5 :

    pip3 install PyQt5  

# Use

run these code to start the ideo_player:

    cd video_sr_player/
    python3 video_player.py


### Shortcut

|   Features     |   Shortcut   |
|   :--:     |   :--:       |
| open video |   Crl + o    | 
| load model | Crl + m      | 
| play       |   p          |
| pause      |   s          |
| replay     |   Crl + r    |
| SR / NO SR |   Space      |
| full_screen / quit full_screen | Esc|


