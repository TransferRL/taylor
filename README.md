# 3D environment:

## 3D environment:

	For more details, the code can be found in ./lib/env/threedmountain_car.py
	Currently the rendering only shows the x and z axises(i.e. The 3D graphics is projected onto the (0,1,0) plane)
	
	action space = Discrete(5)
	0 = neutral
	1 = push left along x = west
	2 = push right along x = east
	3 = push left along y = south
	4 = push right along y = north
	
	observation space = Box(4,)
	obs[0] = x
	obs[1] = y
	obs[2] = x_dot
	obs[3] = y_dot
	
	done is defined as:
		x and y are both greater or equal than the goal position (goal position = 0.5)
	
## 3D env test:

	The test is performed using Q learning. 
	The relevant files are:
		3D mountaincar Q learning.ipynb
		3dmountaincar_qlearning.py
		
	They are the same code, just in different formats
	
	
# Taylor's MASTER algorithm method 1 (obtaining the mapping)

## Algorithm
	1. Do Qlearning on source task (2d mountain car env) and obtain a replay memory (currently 100000 long)
	2. Do RandomAction on target task (3d mountain car env) and obtain a replay memory (currently 100000 long)
	3. Train neural nets to get target one-step transition model 
	4. Find MSE for source task replay memory using one-step transition model
	5. For all combinations of states/actions mappings for (4) find out which one has the least MSE and use that as the state/action mappings
	
	
	
	






	