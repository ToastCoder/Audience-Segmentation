## Audience Segmentation

### About:

A basic replica of youtube algorithm! Classifies the viewers into several groups based on their interests. Implemented with TensorFlow

The code collects the interest of various people given either 1 or 0 (Yes or No) in the dataset and gets the corresponding labels and classifies the type of audience for user defined person in train.py. Useful for classifying audience based on their interests and personalizing ads for each type of audience type.

### Developed by:
[Vigneshwar Ravichandar](https://github.com/ToastCoder)

[Moulishankar M R](https://github.com/Moulishankar10)

### List of Classes:

1. Young adult ( if the value of 'res' in test.py is '[0]' )

2. Engineering Student ( if the value of 'res' in test.py is '[1]' )

3. Medical Student ( if the value of 'res' in test.py is '[2]' )

4. Teachers ( if the value of 'res' in test.py is '[3]' )

5. Adults ( if the value of 'res' in test.py is '[4]' )

6. Travelling kinda person ( if the value of 'res' in test.py is '[5]' )

7. Media Freak ( if the value of 'res' in test.py is '[6]' )

### List of Features: 

1. Technology

2. Politics

3. Food

4. Education

5. Media

6. Travel

7. Medicine

### Execution Instructions:

### Windows:

1. Install the required packages.

2. Execute train.py to retrain the model again.

3. Execute test.py to test the model for executing the algorithm with user defined data.

### MacOS / Linux:

There is a script autorun.sh for automating all the process.

1. Execute the autorun.sh script in terminal.

2. Press Y for executing the script.

![shell1](https://github.com/ToastCoder/Audience-Segmentation/blob/master/images/image1.png)

3. Enter password if asked.

4. This will install all the dependencies and runs the code in order.

5. Press Y for retraining the model. Else press N.

![shell2](https://github.com/ToastCoder/Audience-Segmentation/blob/master/images/image2.png)

7. For classifying a particular data. Follow the wizard by either by entering the user's number of watched videos per category.

![shell3](https://github.com/ToastCoder/Audience-Segmentation/blob/master/images/image3.png)

8. The algorithm will classify the type of audience for the mentioned person. Enjoy!

![shell4](https://github.com/ToastCoder/Audience-Segmentation/blob/master/images/image4.png)

