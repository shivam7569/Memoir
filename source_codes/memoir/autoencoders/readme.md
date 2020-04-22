## How to use Autoencoder Class

* Make sure you have dataset ready for the training before you implement this.
* Design the required architecture in `dense_network.json` file. Here's an example for one layer:
```
"Layer_1": {
        "units": 400, # Number of neurons in this layer.
        "kernel_init": "glorot_uniform", # Initializer for kernels of this layer.
        "bias_init": "glorot_normal", # Initializer for bias of this layer.
        "activation_fn": "tanh", # Activation function to be used.
        "use_dropout": "True", # Ignore this. It's for exception handling.
        "kernel_regularizer": "None", # Regularizer to be used for kernel of this layer. 
        "bias_regularizer": "None", # Regularizer to be used for bias of this layer.
        "use_batch_norm": "True", # Ignore this. It's for exception handling.
        "order": ["dense", "batch_norm", "activation", "dropout"] # Order of secondary layers.
    },
```
* You can get the available regularizers, optimizers, activations, initializers, and metrics (loss functions) from `dense_network_directory.py`, refer to documentation for further details.
* Now you have to create an object of the class and set its attributes.
```
import memoir.autoencoders.dense_ae as dae

ae = dae.Autoencoder(
                    name= ...,
                    learning_rate= ...,
                    loss_function= ...,
                    batch_size= ...,
                    epochs= ...,
                    optimizer= ...,
                    image_size= ...,
                    dropout_keep_prob= ..., (# ignore if not using dropout, will be removed in next update)
                    channel_of_input= ...
                    )
ae.build_graph(
            _return_= ...,
            _print_= ...
            )
ae.fit(
    publish_summaries= ...,
    save_checkpoints= ...,
    interval_of_checkpoints= ...,
    save_outputs= ...,
    output_interval= ...,
    max_keep_checkpoints= ...
    )
```
* That's it. This should get the model running.

Please open an issue if you find any trouble running it.