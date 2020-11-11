
#!/bin/sh
# Author: Vigneshwar Ravichandar
# Autorun shell script to ensure requirements,train the model and test the model.

echo "Autorun.sh verifies the requirements and runs the code properly"
echo "To proceed enter Y, To cancel press N"
read response1
if (($response1 == "y" || $response1 =="Y"))
then
    sudo apt install python3 -y
    sudo apt install python3-pip -y 
    pip3 install numpy
    pip3 install pandas
    pip3 install sklearn
    pip3 install pickle
    pip3 install matplotlib
    echo "Do you want to retrain the model (Y/N): "
    read response2
    if (($response2 == "y" || $response2 == "Y"))
    then
        python3 train.py
    fi
    echo "Do you want to see the data visualization, confusion matrix and other metrics? (Y/N): "
    read response3
    if (($response3 == "y" || $response3 == "n" ))
    then
        python3 visualize.py
    fi

    python3 test.py
fi
echo "Thank you for watching!"
