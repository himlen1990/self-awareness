# self-awareness

How to use:

#1.build caffe:
cd self-awarenss/caffe
make all
make distribute

if you want the system run faster, compile caffe with CUDA. 

#2. build ros package
cd self-awareness
catkin bt

#3.run package
roscd self-awareness
rosrun self-awarenss self_rec

#open another terminal:
roscd self-awareness/data
rosrun play close_drawer2.bag

