## Audience Segmentation

### About:

A basic replica of youtube algorithm! Classifies the viewers into several groups based on their interests. Implemented with TensorFlow.  

The code collects the amount of videos watched in particular category by any person from the dataset and gets the corresponding labels and classifies the type of audience for user defined person in train.py. Useful for classifying audience based on their interests and personalizing ads for each type of audience type.  

Model accuracy of 95-100% is achieved while testing.

### Supported Operating Systems:  
Runs on Windows, Linux and MacOS

### Tested with:  
1. Python 3.8.6 64-bit  
2. TensorFlow 2.4.1  
3. Pop OS 20.10 LTS  

### Developed by:  
[Vigneshwar Ravichandar](https://github.com/ToastCoder)  
[Moulishankar M R](https://github.com/Moulishankar10)  

### List of Classes:  
1. Young adult ( if the value of 'res' in test.py is '[0]' )  
2. Engineering Student ( if the value of 'res' in test.py is '[1]' )  
3. Medical Student ( if the value of 'res' in test.py is '[2]' )  
4. Teachers ( if the value of 'res' in test.py is '[3]' )  
5. Adults ( if the value of 'res' in test.py is '[4]' )  
6. Travellophilic ( if the value of 'res' in test.py is '[5]' )  
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
Execute the following command in the terminal to run with default procedure.  
```python
python3 main.py --test=True
```

### Command Line Arguments:
* `-tr` (or) `--train` - Used to train the Neural Network.  
  * **Argument type:** bool  
  * **Parameter type:** Optional  
  * **Default value:** False

* `-t` (or) `--test` - Used to test the Neural Network with custom inputs.
  * **Argument type:** bool  
  * **Parameter type:** Mandatory 
  
* `-v` (or) `--visualize` - Used to vizualize the metrics.
  * **Argument type:** bool  
  * **Parameter type:** Optional
  * **Default value:** False
  
* `-req` (or) `--install_requirements` - Used to install the required dependancies.
  * **Argument type:** bool  
  * **Parameter type:** Optional  
  * **Default value:** False

* `-e` (or) `--epochs` - Used for mentioning the number of epochs for the model.
  * **Argument type:** int
  * **Parameter type:** Optional
  * **Default value:** 50

* `-bs` (or) `--batch_size` - Used for mentioning the batch size for the model.
  * **Argument type:** int
  * **Parameter type:** Optional
  * **Default value:** 5

* `-l` (or) `--loss` - Used for mentioning the loss function for the model.
  * **Argument type:** str
  * **Parameter type:** Optional
  * **Default value:** "sparse_categorical_crossentropy"

* `-op` (or) `--optimizer` - Used for mentioning the optimizer for the model.
  * **Argument type:** str
  * **Parameter type:** Optional
  * **Default value:** "adam"


### Images:  
![img1](https://github.com/ToastCoder/Audience-Segmentation/blob/master/images/img1.png)  

                              *Screenshot mentioning the training command*

![img2](https://github.com/ToastCoder/Audience-Segmentation/blob/master/images/img2.png)  

                              *Screenshot mentioning the testing command*

