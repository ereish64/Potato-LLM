# Potato LM

## Description
Run language models on potato hardware

## Background
The biggest problems with Large language models is that they are, well… Large. This makes them almost impossible to run on consumer hardware due to their incredible memory requirements. Making it so that most of us don’t even get to see Llama in all of it's 70b glory. If you want to run these models, and don't care how long it takes, then boy do I have the uber solution for you.

## How it works
Language models are composed of layers. These layers are contained within the weights file that (in this case) are downloaded from the huggingface repository. By extracting the weights into individual layers and running them one at a time, we can achieve a far smaller VRAM requirement at the cost of the poor PCIE bus.

## Performance
Getting a language model of 200Gb to run on a 12Gb GPU causes some performance degradation as you can probably imagine. The biggest bottleneck by far is transferring the layers one at a time to the GPU over the PCIE bus. Running mainllamaramcache.py, 7b takes about 2 seconds for one token.

## Disk prefetch
Prefetch from disk is where the layers are loaded ahead of time from the disk into CPU RAM. This way, when the layer needs to be loaded into VRAM, the CPU can just transfer the layer rather than having to load it from disk first. These layers are unloaded from ram after being run on the GPU. They'll have to be loaded again the next time around unless you are running the ramcache version

## VRAM prefetch
If your GPU has enough VRAM to fit 2 layers into memory, you can start loading the second layer while the first layer is still processing. This way once the GPU completes its current layer, the second one is ready to start processing.

## Example
Not interested in reading documentation? Me neither. Here’s an example of how this thing is used:

Break the weights into layers:
Change the model path in save_llama_layers to the huggingface model folder you want to break into layers, the second argument is the output folder where the layers will be saved.
Then run process_layers.py

Run the model:
Change the model path in mainllamaramcache.py to the folder where the layers are saved.
Then run mainllamaramcache.py or llama2.py depending on whether you have enough CPU RAM to cache all of the layers

## Future work
Currently the model only supports Llama2. Still working on the Llama3 architecture.

LLAMA31 DOES NOT WORK YET