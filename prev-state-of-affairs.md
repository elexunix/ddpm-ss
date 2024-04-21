# This is an explanation of the current state of affairs
Date: 05.04.2024, the day of resuming the work on the project


## This commit

If you run code directly from this commit without any changes, you will run Sepformer-only model, and you will get SISDR 16.
Important: you should use batch_size=8 or similar in order to get good results from Sepformer, it won't work with small batches.
Sepformer model checkpoint is from here: `speechbrain/sepformer-wsj02mix`.

## What about SepFormer+Diffusion?

SepFormer+Diffusion can be easily run from this commit, for this you should modify `pipeline/model/ddpmss/sepdiff.py`.
You will get awful SISDR, but approximately the same perceived quality.
Diffusion model checkpoint is from here: `speechbrain/tts-diffwave-ljspeech`.

#### Explanation

I have examined the waveforms, the reason I believe is the following: the diffusion model generates realistic waveforms indeed, this is why the perceived quality is good; however, *it shifts the waveform here and there by a bit in different directions, by a few values, resulting in awful SISDR, which cannot handle such shifts adequately, as it expects exact correspondence between two vectors*. Additional discussion of this could be found in our chat with Max Kaledin, March 13.


# What about SepFormer+Diffusion+Fusion?

This model, I would say, works. It achieves a good SISDR, approximately 15. Recommended batch_size is maybe 3. Patience is required, SISDR of 14 is achieved only after training for 5k steps. Cf. chart from Feb 28 in my chat with Max Kaledin.
One might expect that this model should score strictly higher than previous ones, however, that is not the case. Why?

#### Conjectural explanation

The fusion model may be learning to *simply ignore the diffusion output, and use only sepformer output*.


## I want to run just Sepformer/Diffusion for minimal independent testing, not the whole pipeline, how to do that?
Check out the independent minimal examples folders:
SepFormer: sepformer-example
Diffusion: diffwave-example


## Future plans

Dig into the diffusion model deeper to extract the score function from it, as it is not immediately clear...
