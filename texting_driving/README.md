
# Texting and Driving Project 

### Raj Agrawal, Summer 2016

## Overview 
- The goal of this project was to see if we could detect texting from 
a video stream using convolutional neural nets
- As of now, I am unfortunately not allowed to provide access to the texting and driving data 
- I hope that the pipeline/code is flexible enough so that other people can easily modify the code to adapt to their needs. If there are any questions or problems, feel free to submit an issue

## Training the Model on AWS 

- Launch instance from N. California 
- Use 'ami-125b2c72' on a g2.2xlarge instance 
- Log-in into the instance using ssh -i path_to_pem/yourkey.pem ubuntu@your_dns_address
- If the above does not work you may need to run $chmod 400 yourkey.pem in the directory where yourkey.pem is located
- Enter $vi ~/.theanorc
- Copy and paste the text from ./data/make_theano_work.txt 
- $pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt
- $pip install https://github.com/Lasagne/Lasagne/archive/master.zip
- To make sure everything installed correctly, run $python -c "import theano; print theano.sandbox.cuda.dnn.dnn_available()"
- Make sure you see an output of True
- Clone this repository and cd into texting_driving directory 
- In the code directory place the texting_driving.MOV file
- Run $make data 
- Run $make run_model  
- Note: For more detailed instructions see http://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/

## Hints to Modify Code for Different Tasks  
- images_to_matrix.py
    - Redefine the 'toMatrix' function without the 'num_frames' and just read in all the images as a seperate sample. Don't use the 'to3DMatrix' helper function. 
    - Add functions to generate labelings for the images by perhaps parsing a text file
- random_image_generator.py 
    - Use the 'random_2D_image_generator' function instead if doing work on images instead of videos 
- 3d_cnn_lasagne.py 
    - Change the layers in the 'build_cnn' function. If the inputs are images be sure that the 'input_var' parameter in the function is a 4d Theano Tensor. More details can be found on the MNIST Lasagne Tutorial.  
    - Change the 'iterate_minibatches' to whatever you would like your mini-batches to look like. If dealing w/ images, be sure to change the slicings in this function. 
- Note: Eventually I will have functions to do all of this. 

## Instance Notifications on AWS to Save Money  
- Right click on the gpu instance and select add/edit alarms
- From there you can specify the email address to recieve the notifications 
- I chose to be notified when the max CPU usage on the instance drops below 4% for 5 minutes straight to signal that my net has finished training   
- For more details see https://docs.aws.amazon.com/AmazonCloudWatch/latest/DeveloperGuide/UsingAlarmActions.html

### Notes: 
- As of now, the script 3d_cnn.py has not been tested. Refer to 3d_cnn_lasagne.py for a functioning and tested script. 
- You need to increase the number of allowed instances for GPU instances in AWS if you have never used a GPU in AWS before. See https://docs.aws.amazon.com/general/latest/gr/aws_service_limits.html for details. 

