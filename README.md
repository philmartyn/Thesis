# NII Predictor App - M.Sc Thesis, ML Python bits

This is the backend ML portions of my M.Sc Thesis. It involved the training and prediction elements of a Bi-Polar prediction app.

The ML aspects of the project involved training a CNN model, using transfer learning, in order to get the model to recognise
Bi-Polar in MRI images. 

The app accepts MRI images from a frontend client in order to analyse and return a percentage Bi-Polar score back to the user. 

The app uses a RabbitMQ RPC messaging system to connect the front and backend sections.

Again, some sections of the code are very rough as they may not have been used. They're basically scratch work, which I never removed. 