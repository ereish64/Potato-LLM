# Potato LM

## Description
Run language models on potato hardware

## Background
The biggest problems with Large language models is that they are, well… Large. This makes them almost impossible to run on consumer hardware due to their incredible memory requirements. Making it so that most of us don’t even get to see Llama 3.1 70b in all its glory. If you want to run these models, and don't care how long it takes, then boy do I have the uber solution for you.

## How it works
Language models are composed of layers. These layers are contained within the weights file that (in this case) are downloaded from the huggingface repository. By extracting the weights into individual layers and running them one at a time, we can achieve a far smaller VRAM requirement.

## Performance
Getting a language model of 200Gb to run on a 12Gb GPU causes some performance degradation as you can probably imagine. The biggest bottleneck by far is transferring the layers one at a time to the GPU over the PCIE bus. Expect tokens per hour.

## Disk prefetch
Prefetch from disk is where the layers are loaded ahead of time from the disk into CPU RAM. This way, when the layer needs to be loaded into VRAM, the CPU can just transfer the layer rather than having to load it from disk first.

## VRAM prefetch
If your GPU has enough VRAM to fit 2 layers into memory, you can start loading the second layer while the first layer is still processing. This way once the GPU completes its current layer, the second one is ready to start processing.

## Example
Not interested in reading documentation? Me neither. Here’s an example of how this thing is used:

Break the weights into layers.