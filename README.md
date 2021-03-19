# dog-breed-classifier

### Table of Contents

1. [Installation](#install)

2. [Project Motivation](#motive)

3. [File Description](#desc)

4. [Results](#rslt_table)

5. [Conclusion](#conc)

6. [Screenshots](#rslts)

7. [Licensing, Authors, Acknowledgements](#ack)

<a name="install"></a>
## Installation
The project was written primarily in Python 3  and the libraries used are provided in the Anaconda Distribution in addition to the following:
**cv2,
Flask, 
torch,
PIL**

Run app from command line using **python web_app.py**


<a name="motive"></a>
## Project Motivation
This project aims to classify different dog breeds. It can be used as a learning aid for children or just a fun way to know about the different dog breeds.
The app correctly classifies 133 different dog breeds with an 82% precision - that is much more than the average human :)
It can also be used to batch detect dogs or humans in several images
It does this in the below steps:

  1. Detects a dog in a given image
  
  2. Classifies the detected dog into the appropriate breed
  
  3. If a human is detected it finds the closest dog breed resemblance 

<a name="desc"></a>
## File Description
**dog_app.py:** Python module to perform dog classification

**dog_app_pytorch.ipynb:** Notebook to explore raw data and visualize data and also process data and train model Notebook

**web_app.py:** Module to launch Flask web app

**go.html/master.html** html files for web app

**model_transfer.pt** Output classifier pytorch model


<a name="rslt_table"></a>
## Results

| Model           | Accuracy      | Loss   |
| -------------   |:-------------:| -----: |
| model_scratch   |     11%       |  3.87  |
| model_transfer  |     82%       |  0.76  |

The scratch model used initially was basic and required much more time to reach appreciable accuracy.
Even with batch norm and bigger learning rates training was slow with low accuracy.
Using a pre-trained **densenet-121** model and finetuning the classifer layer achieved much better results.


<a name="conc"></a>
## Conclusion

The implementation of this project was challenging as it was interesting.
From exploring the data to processing the images for use in a transfer learning model.
One aspect that i faced some difficulties was selecting the right neural architecture to use as the go-to models like **VGG-16** had many parameters and therefore had very large pytorch model sizes which was not ideal for web deployment.
Finally settled on **densenet121** since the model size is small but delivers great accuracy.
I also got to improve a little bit on my web development skills. I still have much to do in terms of writing javacript code to make the webapp more interactive and also deploy on a hosting site such as **Heroku**.

<a name="rslts"></a>
## Screenshots
![result](screenshots/dog_app.PNG)

<a name="ack"></a>
## Licensing, Authors, Acknowledgements
Please see  LICENSE and CODEOWNERS files

