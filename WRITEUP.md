# Project Write-Up
Detect people in a designated area and determine the number of people in the frame, the average time they are in the frame, and the total count. Gain important business insight using the information generated.

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Tensorflow SSD Mobilenet V2]
  - [http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz]
  
  - I converted the model to an Intermediate Representation with the following arguments
    `python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
`
  - The model was insufficient for the app because...
  The information provided is inaccurate for the number of people and the average time they were present. 
  
  - I tried to improve the model for the app by...
  Do additional processing of the output to handle incorrect detections, such as adjusting confidence threshold or accounting for 1-2 frames where the model fails to see a person already counted and would otherwise double count.
  
- Model 2: [Tensorflow SSD_ResNet_50_fpn_coco]
  - [http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz]
  - I converted the model to an Intermediate Representation with the following arguments...
  `python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
`
  - The model was insufficient for the app because...
  This model was very very slow.
  
  - I tried to improve the model for the app by...
  Do additional processing of the output to handle incorrect detections, such as adjusting confidence threshold or accounting for 1-2 frames where the model fails to see a person already counted and would otherwise double count.

- Model 3: [Tensorflow SSD_Inception_v2_coco]
  - [http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz]
  
  - I converted the model to an Intermediate Representation with the following arguments...
  `python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
`

  - The model was insufficient for the app because...
    This model was faster than the second model but had a very poor accuracy and calculated a lot more people than reality. 
    
  - I tried to improve the model for the app by...
  Do additional processing of the output to handle incorrect detections, such as adjusting confidence threshold or accounting for 1-2 frames where the model fails to see a person already counted and would otherwise double count.

## What model I've used

- Final model: [person-detection-retail-0013]
[https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html]

I used the _person-detection-retail-0013 Intel® model_, that can be accessed using the model downloader. The model downloader downloads the .xml and .bin files that will be used in this project.

### Download the .xml and .bin files

Go to the **model downloader** directory present inside Intel® Distribution of OpenVINO™ toolkit:

`cd /opt/intel/openvino/deployment_tools/tools/model_downloader
`

Specify which model to download with `--name`. To download the _person-detection-retail-0013 model_, run the following command:

`sudo ./downloader.py --name person-detection-retail-0013 -o /home/workspace/
`

### Running on the CPU
Though by default application runs on CPU, this can also be explicitly specified by `-d CPU` command-line argument:

`
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -pt 0.7 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
`

## Use cases of a people counter app

1.  This app can be used in big stores to evaluate customer check-out points queues to determine if need to add more queues or not based on number of people in each queue.
2. The program can be used for social activities such as marches and street campaigns and estimates the number of participants.
3. The program can be used at public transport system stations and increases or decreases the number of public transport services depending on the crowd.

In general, the program is applicable to scenarios where the number of people present at the site is highly variable over time.


## Assess Effects on End User Needs
This is a pedestrian detector for the Retail scenario. It is based on MobileNetV2-like backbone that includes depth-wise convolutions to reduce the amount of computation for the 3x3 convolution block. The single SSD head from 1/16 scale feature map has 12 clustered prior boxes.

### Example
![Example](images/person-detection-retail-0013.png)

**Pose coverage:** Standing upright, parallel to image plane
**Support of occluded pedestrians:** Yes
**Occlusion coverage:**	<50%
**Min pedestrian height:**	100 pixels (on 1080p)
**Input Image Size:** image height 320 x image width 544
**Outputs:** label, bounding box



Sepid M.