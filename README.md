# GeneralAITest
a General AI for playing games. Example of general intellegence discussed in [this](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) paper.

## Running
Model is located in SimpleModel.py. The model can be run in whole using the run.py file.

## Dependancies
* Pytroch
* Open AI gym/ Universe

## Challanges

### Open AI Gym/ Universe
The Open AI libraries make use of Netwrking data through a docer enviroment. This is convienient for running the enviroments on many different machines. But his brings about a lot of errors that are not neccessarily caused by faulty code in execution and issues are usually only solved with a reboot. This is why A transition to [SerpantAI](https://github.com/SerpentAI) will be made in order to more reliably train models for longer periods of time.

### Models
A model that can acurtly encode data from the screen into a next move was difficult using only a reward value to predict a Q function form an image. Usign hard data like velocity and location makes fore a much faster convergance. However, this data is not always avalible especially in the case of real life examples. To avoid writing software to extract this data whenever it is needed it is important to build models that are able to deduce required infor from the images alone.

## Acomplishments
Learned how a to use [DQN](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)'s and buid an effective Convolutional nueral network model fore applying priciples of general intelegence in a riencfcement learning enviroment. I made use of the google cloud computing platofrm in order to build a larger model with more layers that would not fit onto my personal GPU and instead of using CPU to train which is slow I opted to use the cloud computing platform.

## Notes
A CNN autoencoder was used in an attempt to maximize the ammount of data pulled from each training cycle. This seemed like an effective method to improve CNN training times. I am in the process of collecting concrete data with many different play scenarios but in the current sceario of the driving game the auto encoder slightly improved trainign times on a epoch to epoch basis. This however lead to less space on the GPU so a smaller batch size had to be used. This most likely lead to a long term slowdown of training.

## What's Next
Implement [SerpantAI](https://github.com/SerpentAI) to have a more general testing set for different general intelegences. Implemnt an LSTM with auto encoded frames to hopfully make a more broad testing suite.
