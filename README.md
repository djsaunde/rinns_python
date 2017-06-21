# Representations in Neural Networks

- See the [VDL page](https://restricted.vdl.afrl.af.mil/programs/atrpedia/dist_c/wiki/Representation_and_Phase_Transitions_in_Multi-Layer_Networks) for more information. 
- See also the [GitHub repository](https://github.com/djsaunde/rinns_python).

Contributors: Dan Saunders (djsaunde@cs.umass.edu), Ryan McCormick (rmccorm4@binghamton.edu)

## Setting things up

Clone the GitHub repository in a directory of your choosing using `git clone https://github.com/djsaunde/rinns_python.git`. This will create a directory named `rinns_python` inside of your current directory. This statement should be run in a terminal in \*nix or Mac systems, or in a Git Bash shell on Windows.

Consult `requirements.txt` for the packages which the project code depends on. If you have `pip` installed (which you should), use `pip install -r requirements.txt` to recursively install the packages listed.

## Training

### CIFAR-10

Navigate to the 'cifar10' folder

Run 'train_cifar10_lenet.py'
* `python train_cifar10_lenet.py`

Optional flags are included in the code for your preferences and hardware capability	
* --hardware=(string)
	* 'cpu' (Default)
	* 'gpu'
	* '2gpu'
* --batch_size=(int)
	* 100 (Default)
* --num_epochs=(int)
	* 25 (Default)
* --best_criterion=(string)
	* 'val_loss' (Default)
	* 'val_acc'
	* 'train_loss'
	* 'train_acc'

### Tiny-Imagenet

Navigate to the 'tiny-imagenet' folder

Run 'train_tiny_imagenet.py'
* `python train_tiny_imagenet.py`

Optional flags
* --hardware
* --batch_size
* --num_epochs
* --best_criterion
* --num_classes=(int)
	* 200 (Default)
	* 1-200
* --data_augmentation=(bool)
	* False (Default)
	* True
