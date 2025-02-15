import sys, os
import time,math
import json

# Batch size (controls images per step)
BATCH_SIZE = 256    
# Output file to write data
outfile = sys.argv[1]
# This controls how many batches to prefetch
prefetch_buffer_size = int(sys.argv[2])
# This control parallelism in Tensorflow
parallel_threads = int(sys.argv[3])

print(f"Read arguments 'outfile = {outfile}, prefetch = {prefetch_buffer_size}, threads = {parallel_threads}'")

# This limits the amount of memory used:
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
os.environ['OMP_NUM_THREADS'] = str(parallel_threads)

# how many training steps to take during profiling
num_steps = 30 + prefetch_buffer_size
skip_steps = 10 + prefetch_buffer_size
import tensorflow as tf

#########################################################################
# Here's the Residual layer from the first half again:
#########################################################################
class ResidualLayer(tf.keras.Model):

    def __init__(self, n_filters):
        # tf.keras.Model.__init__(self)
        super(ResidualLayer, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(
            filters     = n_filters,
            kernel_size = (3,3),
            padding     = "same"
        )

        self.norm1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(
            filters     = n_filters,
            kernel_size = (3,3),
            padding     = "same"
        )

        self.norm2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs):

        x = inputs

        output1 = self.norm1(self.conv1(inputs))

        output1 = tf.keras.activations.relu(output1)

        output2 = self.norm2(self.conv2(output1))

        return tf.keras.activations.relu(output2 + x)

#########################################################################
# Here's layer that does a spatial downsampling:
#########################################################################
class ResidualDownsample(tf.keras.Model):

    def __init__(self, n_filters):
        # tf.keras.Model.__init__(self)
        super(ResidualDownsample, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(
            filters     = n_filters,
            kernel_size = (3,3),
            padding     = "same",
            strides     = (2,2)
        )

        self.identity = tf.keras.layers.Conv2D(
            filters     = n_filters,
            kernel_size = (1,1),
            strides     = (2,2),
            padding     = "same"
        )

        self.norm1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(
            filters     = n_filters,
            kernel_size = (3,3),
            padding     = "same"
        )

        self.norm2 = tf.keras.layers.BatchNormalization()

    @tf.function
    def call(self, inputs):

        x = self.identity(inputs)
        output1 = self.norm1(self.conv1(inputs))
        output1 = tf.keras.activations.relu(output1)

        output2 = self.norm2(self.conv2(output1))

        return tf.keras.activations.relu(output2 + x)


#########################################################################
# Armed with that, let's build ResNet (this particular one is called ResNet34)
#########################################################################

class ResNet34(tf.keras.Model):

    def __init__(self):
        super(ResNet34, self).__init__()

        self.conv_init = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters     = 64,
                kernel_size = (7,7),
                strides     = (2,2),
                padding     = "same",
                use_bias    = False
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same")

        ])

        self.residual_series_1 = tf.keras.Sequential([
            ResidualLayer(64),
            ResidualLayer(64),
            ResidualLayer(64),
        ])

        # Increase the number of filters:
        self.downsample_1 = ResidualDownsample(128)

        self.residual_series_2 = tf.keras.Sequential([
            ResidualLayer(128),
            ResidualLayer(128),
            ResidualLayer(128),
        ])

        # Increase the number of filters:
        self.downsample_2 = ResidualDownsample(256)

        self.residual_series_3 = tf.keras.Sequential([
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
        ])

        # Increase the number of filters:
        self.downsample_3 = ResidualDownsample(512)


        self.residual_series_4 = tf.keras.Sequential([
            ResidualLayer(512),
            ResidualLayer(512),
        ])

        self.final_pool = tf.keras.layers.AveragePooling2D(
            pool_size=(8,8)
        )

        self.flatten = tf.keras.layers.Flatten()
        self.classifier = tf.keras.layers.Dense(1000)

    @tf.function
    def call(self, inputs):

        x = self.conv_init(inputs)
        x = self.residual_series_1(x)
        x = self.downsample_1(x)
        x = self.residual_series_2(x)
        x = self.downsample_2(x)
        x = self.residual_series_3(x)
        x = self.downsample_3(x)
        x = self.residual_series_4(x)
        x = self.final_pool(x)
        x = self.flatten(x)

        logits = self.classifier(x)

        return logits

#########################################################################
# Training a single step
#########################################################################

@tf.function()
def calculate_accuracy(logits, labels):
    # We calculate top1 accuracy only here:
    selected_class = tf.argmax(logits, axis=1)

    correct = tf.cast(selected_class, tf.float32) == tf.cast(labels, tf.float32)

    return tf.reduce_mean(tf.cast(correct, tf.float32))


@tf.function()
def calculate_loss(logits, labels):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
    return tf.reduce_mean(loss)

@tf.function()
def training_step(network, optimizer, images, labels):
    with tf.GradientTape() as tape:
        logits = network(images)
        loss = calculate_loss(logits, labels)

    gradients = tape.gradient(loss, network.trainable_variables)

    optimizer.apply_gradients(zip(gradients, network.trainable_variables))

    accuracy = calculate_accuracy(logits, labels)

    return loss, accuracy

#########################################################################
# Prepare a dataset
#########################################################################
def prepare_data_loader():

    tf.config.threading.set_inter_op_parallelism_threads(parallel_threads)
    tf.config.threading.set_intra_op_parallelism_threads(parallel_threads)
    print('threading set: ',tf.config.threading.get_inter_op_parallelism_threads(),tf.config.threading.get_intra_op_parallelism_threads())

    print("Parameters set, preparing dataloading")
    #########################################################################
    # Here's the part where we load datasets:
    from ilsvrc_dataset import get_datasets

    class FakeHvd:
        def size(self): return 1
        def rank(self): return 0

    with open("ilsvrc.json", 'r') as f:
        config = json.load(f)

    config['data']['batch_size'] = BATCH_SIZE
    config['data']['num_parallel_readers'] = parallel_threads
    config['data']['prefetch_buffer_size'] = prefetch_buffer_size 

    print(json.dumps(config, indent=4))

    config['hvd'] = FakeHvd()

    train_ds, val_ds = get_datasets(config)

    options = tf.data.Options()
    options.threading.private_threadpool_size = parallel_threads
    train_ds = train_ds.with_options(options)
    val_ds = val_ds.with_options(options)

    print("Datasets ready, creating network.")
    #########################################################################

    return train_ds, val_ds

#########################################################################
# Training an epoch
#########################################################################

def train_epoch(train_ds, val_ds, network, optimizer):
    # Here is our training loop!
    steps_per_epoch = int(1281167 / BATCH_SIZE)

    n = 0
    sum = 0.
    sum2 = 0.
    step = 1

    start = time.time() # Must place here to include the "train_ds.take(num_steps)" call
    for train_images, train_labels in train_ds.take(num_steps):
        loss, acc = training_step(network, optimizer, train_images, train_labels)
        end = time.time()

        total_time = end - start
        images_per_second = BATCH_SIZE / total_time
        if step > skip_steps: # skip the first measurement because it includes compile time
            sum += images_per_second
            sum2 += images_per_second * images_per_second
            n += 1
        
        print(f"Finished step {step}, loss={loss:.3f}, acc={acc:.3f}, batch={BATCH_SIZE}, time={total_time:1.3e} ({images_per_second:.3f} img/s).")
        step += 1
        start = time.time() # Time the next "train_ds.take(num_steps)" call
    
    mean_rate = sum / n
    stddev_rate = math.sqrt( sum2/n - mean_rate * mean_rate )
    
    return mean_rate, stddev_rate

#########################################################################
# Main: measure image throughput
#########################################################################
def main():

    print("Preparing data loader")
    train_ds, val_ds = prepare_data_loader()

    print("Checking sample images")
    example_images, example_labels = next(iter(train_ds.take(1)))
    print("Initial Image size: ", example_images.shape)
    
    print("Building network.")
    network = ResNet34()
    output = network(example_images)
    print("network output shape:", output.shape)
    print(network.summary())

    # We need an optimizer.  Let's use Adam:
    print("Entering training.")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    mean_rate, stddev_rate = train_epoch(train_ds, val_ds, network, optimizer)

    print("#######################################")
    print("Threads = %d, Prefetch = %d" % (parallel_threads, prefetch_buffer_size))
    print(f'mean image/s = {mean_rate:8.2f}   standard deviation: {stddev_rate:8.2f}')
    print("#######################################")

    with open(outfile, "a") as file:
        file.write(f"{prefetch_buffer_size},{parallel_threads},{mean_rate:1.6e},{stddev_rate:1.6e}\n")

if __name__ == "__main__":
    main()
