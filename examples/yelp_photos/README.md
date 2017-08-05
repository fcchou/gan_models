# How to Run the Photo Generator
To run this example, first download the [Yelp public dataset](https://www.yelp.com/dataset_challenge/dataset).
Note that you only need the photo data for this example.

These Yelp photos are of a variety of shapes and has a high resolution,
we need to first preprocess them by cropping and down-sampling. Run:

    python -m examples.yelp_photos.photo_preprocess --path-to-photos=./yelp_photos
  
The command above will create a `processed_photos.npy` file containing the processed photos.
Then we can train the GAN model. In this example we used the DCGAN architecture and improved WGAN algorithm:

    python -m examples.yelp_photos.train_photo_generator --input-path=processed_photos.npy --epochs=5 --batch-size=64 --iters=75
 
In this example the training took about 2 days on one single GTX 1070
