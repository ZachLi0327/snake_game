The main function of this project is a snake game controlled by an agent.Before the implementation, please make sure you have already done the following three steps.

1.Firstly, make sure you have installed pygame and numpy.You can try pip install numpy pygame in terminal.

2.Secondly,run the train.py, there will be a 10 million times of iteration which trains the agent and form aq-table. Then a file called q_table.npy will be saved in the root folder.

3.Thirdly, run the main.py , the program will load the pre-trained q-table in the folder and the snake game will run atuomatically. If the snake hit the wall or itself, the game over and returns the total reward of the epoch.

For faster training, you can change the parameter called "num_episodes" from 10000000 to 1000000.
