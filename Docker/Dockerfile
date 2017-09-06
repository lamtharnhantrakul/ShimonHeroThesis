###########
#
#      DOCKER CONTAINER FOR SHIMON HERO THESIS RESEARCH:
# MAINTAINED BY: Lamtharn (Hanoi) Hantrakul -- lhantrakul3@gatech.edu
#
# Please add new pip/apt installs in this block. Don't forget a "&& \" at the end
# of all non-final lines. Thanks!
#
###########

### Pull official Anaconda Python Distribution 4.4.0 (which has python3.6)
FROM continuumio/anaconda3:4.4.0

### Install Custom Libraries
# Install CPU version of Tensorflow
RUN pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.3.0-cp36-cp36m-linux_x86_64.whl

# Install Pygame
RUN pip install pygame
