#-------------------------------------------------------------------------------------------------------------------------------

# AUDIENCE SEGMENTATION

# FILE NAME: main.py

# DEVELOPED BY: Vigneshwar Ravichandar, Moulishankar M R

# TOPICS: Regression, Machine Learning, TensorFlow

#-------------------------------------------------------------------------------------------------------------------------------

# SET OF DESCRIPTIONS
description = [ 'Command Line Interface for Audience Segmentation',
                'Argument taken for training model.',
                'Argument taken for installing requirements',
                'Argument taken for visualizing metrics',
                'Argument for testing with custom input',
                'Argument for mentioning the number of Epochs',
                'Argument for mentioning the amount of Batch Size',
                'Argument for mentioning the Loss Function',
                'Argument for mentioning the Optimizer']

# FUNCTION TO CONVERT STR INPUT TO BOOL
def strBool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Expected a Boolean Value.')

# FUNCTION FOR PARSING ARGUMENTS
def parse():
    parser = argparse.ArgumentParser(description = description[0])
    parser.add_argument('-tr','--train',
                        type = strBool, 
                        help = description[1], 
                        default = False)

    parser.add_argument('-req','--install_requirements', 
                        type = strBool, 
                        help = description[2], 
                        default = False)

    parser.add_argument('-v','--visualize', 
                        type = strBool, 
                        help = description[3],
                        required = False)

    parser.add_argument('-t','--test', 
                        type = strBool, 
                        help = description[4],
                        required = True)

    parser.add_argument('-e','--epochs', 
                        type = int, 
                        help = description[5],
                        default = 50)

    parser.add_argument('-bs','--batch_size', 
                        type = int, 
                        help = description[6],
                        default = 5)
    
    parser.add_argument('-l','--loss', 
                        type = str, 
                        help = description[7],
                        default = "sparse_categorical_crossentropy")
    
    parser.add_argument('-op','--optimizer', 
                        type = str, 
                        help = description[8],
                        default = "adam")
                        
    args = parser.parse_args()
    return args

# MAIN FUNCTION
if __name__ == "__main__":

    # IMPORTING REQUIRED MODULES
    import os
    import argparse

    # DISABLING TENSORFLOW DEBUGGING INFORMATION
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    print("TensorFlow Debugging Information is hidden.")
    
    args = parse()

    if (args.install_requirements):
        os.system('sudo apt install python3-pip')
        os.system('pip3 install -r requirements.txt')
    
    if (args.train):
        os.system(f'python3 src/train.py --epochs={args.epochs} --batch_size={args.batch_size} --loss={args.loss} --optimizer={args.optimizer}')

    if (args.test):
        os.system(f'python3 src/test.py --epochs={args.epochs} --batch_size={args.batch_size} --loss={args.loss} --optimizer={args.optimizer}')

    if(args.visualize):
        os.system('python3 src/visualize.py')
