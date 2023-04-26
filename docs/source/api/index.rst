API
===
* Model is a generic class for training and inference, which builds model from base_components and TrainingPlan.
* Base components contain basic components for building neural network, such as inference networks (encoders), generative netwoeks(decoders), and loss function.
* TrainingPlan is a class for efficient training, which specifies dataloader, loss aggregation, optimizer, training and validation process.

.. toctree::
     :maxdepth: 2

     model
     base_components
     TrainingPlan