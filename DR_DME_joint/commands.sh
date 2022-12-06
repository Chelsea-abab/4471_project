
#use the following sentence to run the program
#you need have a gpu to run it
python ./4471_project/DR_DME_joint/baseline.py --num_epoch 100 --gpu 0 --multitask --crossCBAM --log_name test_1

tensorboard --logdir=./4471_project/DR_DME_joint/logs/test_7 --port 8123





