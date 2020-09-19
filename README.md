# Use of TensorFlow HUB

## Steps to follow:
1. Install tensorflow_hub

  pip install --upgrade tensorflow_hub
2. clone git repository

  git clone --depth 1 https://github.com/tensorflow/models
3. Initializing the APIs

  %%bash

  sudo apt install -y protobuf-compiler

  cd models/research/

  protoc object_detection/protos/*.proto --python_out=.

  cp object_detection/packages/tf2/setup.py .

  python -m pip install .

  <u>for installing protobuf [on MAC]:</u>

    - Open Terminal and type the following

    - PROTOC_ZIP=protoc-3.7.1-osx-x86_64.zip

    - curl -OL https://github.com/google/protobuf/releases/download/v3.7.1/$PROTOC_ZIP

    - sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc

    - rm -f $PROTOC_ZIP

4. File Name hub_project_pretrained is pre-trained model
5. File Name hub_project_ImageApplication is utilizing trained model to detect objects in our images.
6. File Name hub_project_VideoApplication is utilizing trained model to detect objects in video using webcam.

# Join me:

Whatsapp: https://qrgo.page.link/TMiTp

YouTUBE: https://qrgo.page.link/wh5tq
