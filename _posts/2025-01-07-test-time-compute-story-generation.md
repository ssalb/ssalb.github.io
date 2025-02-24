---
layout: post
title: "Test-Time Compute: My Take on Story Generation"
date: 2025-01-07 09:00:00
description: Exploring test-time compute and beam search with a story generator app
tags: transformers llms test-time-compute beam-search
# categories: transformers
thumbnail: assets/img/blog/2025-01-07-story-gen/gpu.webp
featured: false
---

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blog/2025-01-07-story-gen/tree_diagram.webp" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<!-- # Test-Time Computing: My Take on Story Generation -->
We've seen incredible progress in large language models (LLMs) over the past few years, driven largely by scaling up model sizes and training data. But as noted in a [recent blog post by Hugging Face](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute), we're starting to hit some hard limits—training these massive models requires billion-dollar compute clusters, and we're also running out of high-quality training data.  

Enter test-time compute: instead of building ever-larger models, what if we let smaller models "think longer" about hard problems? Recent research shows this approach can be remarkably effective. In a [paper by Google DeepMind](https://arxiv.org/pdf/2408.03314) published last August, the team demonstrates that with smart test-time compute allocation, smaller models can actually outperform models 14 times their size on certain tasks. Probably the most famous examples are OpenAI's o1 and o3 models, but they're not the only ones that can do these tricks: Deep Mind’s results were also replicated by the Hugging Face team as described in the blog post mentioned above.  

I wanted to get more familiar with these concepts, so I decided to explore them with a simple and fun toy project: story generation. Could test-time computing help a modest LLM write better stories by carefully evaluating and refining its outputs?  

Well... yes! at least compared to the base model output. I have a demo under [this space](https://huggingface.co/spaces/ssalb/story_generator). It’ll be live for a few days, but feel free to clone it later and run it yourself. The source code is also available in [this GitHub repo](https://github.com/ssalb/story-beam-search).  

---
## The Building Blocks: Beam Search and Quality Metrics
My approach uses beam search, a method that explores multiple possible story variations simultaneously instead of generating just one. Think of it like a chess player considering different moves before committing to one. Concretely, it does this by expanding each partial story with a number of possible next paths (beams) and scoring them; it then keeps only the top-scoring beams (the beam width) and moves on to repeat this process for a determined number of steps. This way, the model stays focused on the most promising story paths while still branching out enough to discover interesting possibilities. But I needed a way for the system to judge which stories were “better,” so I implemented three quality metrics:

* **Coherence**: Do sentences flow naturally from one to the next? I evaluate this using cosine similarity between adjacent sentences.  
* **Fluency**: Does the text read smoothly and naturally? To approximate this, I used BERT to evaluate the probability of each generated token based on the previous tokens (by masking them). (Disclaimer: I really wanted to give [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) a try, so this biased my approach significantly. 😉)  
* **Genre alignment**: Does the story match its intended genre? I used a zero-shot classifier to predict whether a piece of text could be labeled by a predefined genre.

Writing a functional app took about 10% of the time at most, and the CI pipeline to get it running on a Hugging Face space another 10%. In contrast, about 40% of the time I invested was spent experimenting with these scoring algorithms. I tried different normalizations and ways of splitting the stories. For fluency, for example, I use the whole story as one unit. For genre alignment, I found it worked better by splitting stories into sentences and classifying them independently—this helped control alignment throughout the story.  

Fun fact: the genre alignment scorer turned out to be a decently good, low-effort prompt injection mechanism. I check the probability of the initial prompt being classified as "story" versus "prompt injection." If the "storyness" is too low or the "prompt injection" probability too high, I stop the process before generating anything. It’s by no means perfect, and I’ve definitely seen a few false positives, but it’s surprisingly effective for a quick demo.

## Making Beam Search Work for Creative Tasks
One interesting challenge emerged: basic beam search seems great at finding the "optimal" solution but not so great at generating diverse, creative outputs—my first attempts returned stories that were way too similar, sometimes differing by only a single word. Taking inspiration from the [blog post](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute) mentioned earlier (see the DVTS section), I modified the algorithm to maintain independent "beams" of stories that develop separately, rather than constantly comparing and filtering them all together. This helped preserve more variety in the outputs.  

## The Computing Challenge
Were you wondering where I spent the other 40% of my time? Well, here's where it gets interesting: while generating a single story from a language model takes just a couple of seconds on a GPU (I tested on a T4), the full test-time computation process is much more intensive. With a modest-sized (and not particularly impressive anymore) model like GPT2, the entire process takes 3–5 minutes. Scale up to Llama 3.2 1B (my original target model) and you'll get significantly better stories, but you'll also be looking at 30+ minutes of processing time! That's why I scaled it down for my demo.

I'm sure my code can be optimized further, but even after some thorough refactoring to process data in batches wherever possible, I only improved runtime by about 20%. The iterative steps just take a while—at least compared to what we’re used to when using LLMs directly. I believe researchers are putting effort into embedding these search algorithms in the network architectures themselves, but I’m not an expert in this field, so don’t quote me on that!

---
## What I Learned

Test-time computing offers a fascinating alternative to the “bigger is better” mindset that often dominates LLM and AI development. While it does introduce significant computational overhead, it enables smaller models to produce higher-quality outputs through careful evaluation and refinement.
Imagine how “small model, big compute” could also help with things like dialogue systems, or specialized domain tasks; anything really that would benefit from "thinking longer and in steps". 

Of course, we have to acknowledge the elephant in the room: more inference steps cost more time, so if you need super-quick responses, you might be better off with bigger models. In practice, it will probably be a combination of both–as we keep pushing the limits of model scaling, these methods for making smarter use of inference-time compute will likely become increasingly important—especially if new algorithms and hardware optimizations can streamline the process.

For story generation specifically, in this experiment beam search seems to be very effective at finding “optimal” stories according to our metrics—though maintaining creativity and diversity requires some clever tweaks to the standard approach. Perhaps other methods that would preference exploration over exploitation would work better. I do have a soft spot for Monte Carlo methods, so I'll try to find another ~~excuse~~ project to use a MC tree search next.

### Takeaways  
- **Small Models, Big Thinking**: With enough “thinking time,” a modest LLM can punch above its weight class.  
- **Balancing Creativity & Optimality**: Beam search seems to be better at finding an optimal solution. Other methods might preference exploration more.
- **Real-World Viability**: Extra processing time isn’t always ideal, but for certain applications it can be worth the trade-off.  

Feel free to tinker with the code, clone the project, or adapt it for your own experiments. I’d love to hear about your successes, hiccups, and any wild new ideas you come up with!
