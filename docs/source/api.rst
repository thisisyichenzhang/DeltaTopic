API
===

Models
------
A ready-to-run class for training and inference, defines module and TrainingPlan.

.. autosummary::
   :toctree: .
   :nosignatures:
   
   DeltaTopic.nn.modelhub.BALSAM
   DeltaTopic.nn.modelhub.DeltaTopic

Module 
------
A generic pytorch module, defines the inference and generation step, and loss function.

.. autosummary::
   :toctree: .
   :nosignatures:

   DeltaTopic.nn.module.BALSAM_module
   DeltaTopic.nn.module.DeltaTopic_module
   
Base components 
---------------
Building blocks for the modules, defines encoder and decoder.

.. autosummary::
   :toctree: .
   :nosignatures:

   DeltaTopic.nn.base_components.BALSAMEncoder
   DeltaTopic.nn.base_components.BALSAMDecoder
   DeltaTopic.nn.base_components.DeltaTopicEncoder
   DeltaTopic.nn.base_components.DeltaTopicDecoder
   
Training plan   
-------------
A Pytorch lightning wrappper, defines the training/validation step, optimizers, and data loaders.

.. autosummary::
   :toctree: .
   :nosignatures:

   DeltaTopic.nn.TrainingPlan.TrainingPlan