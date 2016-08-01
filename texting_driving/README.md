
# Texting and Driving Project 

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
- $pip install -r https://raw.githubusercontent.com/dnouri/nolearn/master/requirements.txt
- $pip install git+https://github.com/dnouri/nolearn.git@master#egg=nolearn==0.7.git
- To make sure everything installed correctly, run $python -c "import theano; print theano.sandbox.cuda.dnn.dnn_available()"
- Make sure you see an output of True
- Clone this repository and cd into texting_driving directory 
- In the code directory place the texting_driving.MOV file
- Run $make data 
- Run $make run_model  
- Note: For more detailed instructions see http://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/

## Hints to Modify Code for Different Tasks 
- images_to_matrix.py
    - 
    - 
    - 
- random_image_generator.py 
    - 
    - 
    - 
- 3d_cnn_lasagne.py 

## Instance Notifications in AWS to Save Money  
- Right click on the gpu instance and select add/edit alarms
- From there you can specify the email to recieve the notification 
- I chose to be notified when the max CPU usage on the instance drops below 4% for 5 minutes straight to signal that my net has finished training   
- For more details see https://docs.aws.amazon.com/AmazonCloudWatch/latest/DeveloperGuide/UsingAlarmActions.html

### Notes: 
- As of now, the script 3d_cnn.py has not been tested. Refer to 3d_cnn_lasagne.py for a functioning and tested script. 
- You need to increase the number of allowed instances for GPU instances in AWS if you have never used a GPU in AWS before. See https://docs.aws.amazon.com/general/latest/gr/aws_service_limits.html for details.  