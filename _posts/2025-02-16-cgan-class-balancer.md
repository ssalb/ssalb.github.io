---
layout: post
title: "A Conditional GAN for Data Augmentation: A Cautionary Tale"
date: 2025-02-23 09:00:00
description: Balancing an image classification dataset using synthetic images
tags: GAN image-classification deep-learning keras
# categories: transformers
# thumbnail: assets/img/blog/2025-01-07-story-gen/gpu.webp
featured: false
---


In recent years, the use of synthetic data has become increasingly important for improving machine learning models, especially in situations
where data is scarce, sensitive, or simply when it becomes impractical to keep scaling real-world data. Major tech companies like
[Google, OpenAI](https://www.businessinsider.com/ai-synthetic-data-industry-debate-over-fake-2024-8), and particularly Nvidia with the
release of [Nemotron](https://blogs.nvidia.com/blog/nemotron-4-synthetic-data-generation-llm-training/) and
their [Omniverse platform](https://nvidianews.nvidia.com/news/nvidia-expands-omniverse-with-generative-physical-ai), among others, have been
investing heavily in this. It is also an active research area in academia, with [Stanford's Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)
probably being the first recent example that comes to mind for most people in the field.

This is the backdrop for this project, where I set out to answer the following question:

**For an image classification task, can generated images improve model performance on an imbalanced dataset?**

In other words, does rebalancing an imbalanced dataset by sampling synthetic images for minority classes improve classification accuracy—all
while staying on a private, consumer-grade budget (because I'm not a research lab 😜)?

To this end, I used an unbalanced version of the EuroSAT dataset, trained a conditional GAN to generate new synthetic samples conditioned on
a given class, and fine-tuned a pre-trained _EfficientNetB2_ on different versions of this dataset. I trained all models on a rather modest
instance from AWS SageMaker, while some other tasks were run on my personal PC. If you're interested, the code and scripts are all available
under [this GitHub repo](https://github.com/ssalb/cgan-class-balancer).

*Spoiler alert: not everything went as planned, but those unexpected twists made the journey all the more interesting!*

---

## Dataset Overview

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid 
            loading="eager" 
            path="assets/img/blog/2025-02-23-cgan-class/example_images.png" 
            class="img-fluid rounded z-depth-1" 
        %}
    </div>
</div>
<div class="caption">
    Examples from the EuroSAT dataset. Not every class is displayed.
</div>

[EuroSAT](https://github.com/phelber/EuroSAT) consists of 10 classes of land-use/land-cover from ESA's Sentinel-2 satellite images
[(Helber et al., 2019)](https://arxiv.org/abs/1709.00029), with 27,000+ total images (RGB or multispectral). Classes include:

- Annual Crop
- Forest
- Herbaceous Vegetation
- Highway
- Industrial Buildings
- Pasture
- Permanent Crop
- Residential Buildings
- River
- SeaLake

In particular I've used [this version](https://huggingface.co/datasets/blanchon/EuroSAT_RGB) from Hugging Face's Datasets, which only contains the RGB
images and already comes with a train-validation-test split.

The original dataset is well balanced, so I artificially reduced two classes–_Highway_ and _River_–by 85% in the training set, leaving the validation and
test sets untouched. I picked these two becuase they show quite distinct features, unlilke, say, _Forest_ or _SeaLake_, wich can often be very flat,
but they're also not as complex as _Industrial Buildings_ or _Residential Buildings_ can be.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid 
            loading="eager" 
            path="assets/img/blog/2025-02-23-cgan-class/Eurosat_training_distr.png" 
            class="img-fluid rounded z-depth-1" 
            zoomable=true
        %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid 
            loading="eager" 
            path="assets/img/blog/2025-02-23-cgan-class/Eurosat_unbalanced_distr.png" 
            class="img-fluid rounded z-depth-1" 
            zoomable=true
        %}
    </div>
</div>
<div class="caption">
    The class counts of the original training set (left) and the atrificially unbalanced one (right) with the classes <i>Highway</i> and <i>River</i>
    undersampled by 85%
</div>

## Experiment Design  

Throughout this experiment, I've used the [Keras 3](https://keras.io/about/) framework with TensorFlow as the backend. I've tried to keep my
code as "backend agnostic" as possible, and most models should actually work with other supported backends (currently PyTorch and JAX).
The only exception should be the custom training loop for the GAN. If you're interested in optimizers and hyperparameters, check out the code.

Here's the plan:

#### Baseline  

Fine-tune a pretrained _EfficientNetB2_ classifier using both the original and the unbalanced dataset, and use the classification accuracy
on the test set as the metric to compare different models. Train multiple models using different random seeds and calculate an average score for each.

#### Data Augmentation  

Build a conditional GAN model and train the generator to produce images of all 10 classes, conditioned on the class label as input. Use the
generator to create images of the two minority classes. Using the _unbalanced_ classifier mentioned above, pass these generated images through
it, keeping only those that the model correctly classifies as belonging to the same class as the intended one (GAN input). Generate 500 new
images for each class and build three new training sets from these:

- **augmented100**  
- **augmented300**  
- **augmented500**  

Each augments the unbalanced training set by 100, 300, and 500 images, respectively.

#### Evaluation  

Fine-tune multiple classifiers for each augmented dataset, calculate the group accuracy score, and compare it to the baseline. Additionally,
calculate the per-class group accuracy for the minority classes across all trained classifiers and compare them.

---

## The GAN Rabbit Hole 🕳️  

So far, so good. I had my dataset, experiment design, and infrastructure (first time using Pulumi) set up. My unbalanced dataset was ready,
and I had fine-tuned a classifier to establish a baseline. Time to generate some synthetic data!

I started with [this Keras example](https://keras.io/examples/generative/conditional_gan), modifying it for image dimensions and channels (RGB).
My first training run on the _unbalanced_ dataset wasn't good but produced exactly what I expected—blurry patches of green, brown, and blue, vaguely
resembling satellite imagery.

_Great!_ – I thought – _this is going in the right direction._ I tweaked hyperparameters systematically, adding a callback to generate samples
every 10 epochs. After many trials, I still had little more than color-matched patches, maybe some crop-like structures—if I squinted.
So, I explored deeper networks, different upsampling methods, and better class conditioning.  

That led me into researching more advanced GANs. I added an embedding layer for class labels, learned about projection discriminators
[(Miyato & Koyama, 2018)](https://arxiv.org/abs/1802.05637), and experimented with Wasserstein GANs with gradient penalty
[(WGAN-GP; Gulrajani et al., 2017)](https://arxiv.org/abs/1704.00028). Each change meant longer training runs, but at least I never had to
upgrade my instance 😅.  

Despite all the effort, nothing was working, my AWS bill was climbing, and I was running out of ideas. Then it hit me—EuroSAT is a well-known dataset. Surely,
someone had tried this before? Sure enough, I found [a paper](https://www.cs.swarthmore.edu/~llwin1/files/labelled_sat_images_dcgan_lwin.pdf) from (probably)
2022 Swarthmore students who had used a similar model. I replicated their setup and... nothing!  

But this was the turning point. I had proof that it _was_ possible, just not with my dataset. So, I finally did what I should’ve done earlier:
trained a GAN on the _full_ dataset. First try… **it worked!**  

#### The End of My GAN Learning Journey  

The issue was the dataset imbalance all along. I had tried class weights but never focused on balancing techniques like oversampling.  

At this point, I decided to use the GAN trained on the full dataset. Sure, it weakened the project's real-world applicability—the whole goal
was to handle imbalance with synthetic data—but solving that problem was an entirely new project. The right call was to finish answering my
original question and leave that challenge for another day.

On the bright side, I learned so much about GANs, conditioning them and stabilizing training, so no regrets!

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid 
            loading="eager" 
            path="assets/img/blog/2025-02-23-cgan-class/generated_highway.png" 
            class="img-fluid rounded z-depth-1" 
            zoomable=true
        %}
    </div>
</div>
<div class="caption">
    Generated samples of the <i>Highway</i> class.
</div>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid 
            loading="eager" 
            path="assets/img/blog/2025-02-23-cgan-class/generated_river.png" 
            class="img-fluid rounded z-depth-1" 
            zoomable=true
        %}
    </div>
</div>
<div class="caption">
    Generated samples of the <i>River</i> class.
</div>

---

## Final Results

At this point, I had spent way more time (and money on AWS) than I had planned. For this reason I decided to adapt my analysis: I would train
only 3 classifiers per setup. The issue was that this wouldn't be near enough samples for calculating proper group statistics, so I ended up
pulling out a trick.

#### Estimating Variance with Pooling

Let $$x^i_j$$ be the accuracy score of the $$j$$-th trained classifier ($$j \in [1,3]$$) for dataset $$i \in G$$, where $$G$$ is the set of
datasets (or groups) mentioned above. I'll also add to $$G$$ a variant that fine-tunes the classifier on the _unbalanced_ dataset with class
weights to compensate for imbalance.

Let's make two assumptions:  

- Three samples are enough to estimate the mean of each group (outrageous, I know).  
- All groups share the same (symetric) accuracy score distribution, just shifted by their mean.  

With this, we can subtract the group's mean accuracy $$\mu^i$$ from each sample, obtaining $$\tilde{x}^i_j = x^i_j - \mu^i$$.
By doing so, we've effectively centered all distributions around zero. Since we assumed they were the same distribution but shifted, we can now
"borrow power" from the other samples and calculate a shared variance as:

$$
\sigma^2 = \frac{1}{2|G|}\sum_{i\in G}\sum_{j=1}^{3}(\tilde{x}^i_j)^2
$$

This is of course a simplification, it is meant to help visualize differences in the mean results and not to rigorously account for all sources
of uncertainty.

#### Performance on the Test Set

Finally! Do these generated images improve the classifier's accuracy on the test set? Well, they do seem to help,
but not more than simply training on the unbalanced dataset while giving more weight to the minority classes.
But that result alone is a bit misleading. What caught me off guard was that when looking at the mean accuracy
on the minority classes only, adding synthetic images actually performs worse.

This result might seem strange at first—it certainly did to me—but after thinking it through, I have a hypothesis.
I suspect that, on the one hand, the generated samples lack variability and fail to capture all the details of the real data.
On the other hand, they still upsample the minority classes. The effect is that the more synthetic images in the dataset,
the more the model overfits on them, reducing per-class accuracy for these classes while smoothing out the decision boundaries
for the majority classes, improving their accuracy. The net effect is positive, which is why the overall accuracy improves.

If this hypothesis is correct—and considering that I trained the GAN on the full dataset—it suggests that effectively modeling
the synthetic data is critical for dataset augmentation to work this way. Proving this hypothesis, though, is a whole project in itself,
so I won’t attempt it today.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid 
            loading="eager" 
            path="assets/img/blog/2025-02-23-cgan-class/test_set_performance.png" 
            class="img-fluid rounded z-depth-1"
            zoomable=true
        %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid 
            loading="eager" 
            path="assets/img/blog/2025-02-23-cgan-class/per_class_performance.png" 
            class="img-fluid rounded z-depth-1"
            zoomable=true
        %}
    </div>
</div>
<div class="caption">
    Mean classification accuracy on the full test set (left) and only considering the minority classes <i>Highway</i> and <i>River</i> (right)
    for after fine-tuning on each dataset labeled in the x-axis.
</div>

#### Conclusion  

This project started as a straightforward experiment in data augmentation but quickly turned into a deep dive into GANs, dataset balancing,
and the unexpected ways synthetic data interacts with model training. While the generated images did improve overall accuracy, they didn't
outperform a simpler approach like class weighting—and, in fact, they hurt performance on the minority classes. That was an unexpected twist,
but one that makes sense in hindsight: more data isn’t always better if it doesn’t capture the full complexity of the real thing.

There are plenty of open questions left—like whether a better GAN model, or perhaps a difussion model, trained directly on the unbalanced
dataset could have made a difference. That’s a problem for another day. For now, this was a fun (if occasionally frustrating) exploration,
and I’ve learned a lot along the way. And at the very least, I now have a much better understanding of why dataset augmentation is trickier
than it seems.
