# Generating Textual Stories from Short Video Clips
**A modular, zero-shot approach to sequential visual narrative**

MSc Computer Science, Sapienza University of Rome, Computer Vision course project (Prof. Marini) · Amirhossein Khishvand · 2026

📊 [Presentation](presentation.pdf) · 📓 [Notebook](03_context_provided.ipynb) · [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/video-clip-storytelling/blob/main/03_context_provided.ipynb)

## Overview
Turning a video into a story means linking separate visual moments into one continuous, chronological narrative. Models that treat frames independently tend to hallucinate or lose the temporal thread. This project builds a **lightweight, training-free pipeline** that does this with off-the-shelf models:

```
video clip ──► keyframes ──► BLIP-2 captions ──► LLaMA 3.3 70B ──► scene story
                                                    ▲                  │
                                                    └── previous scene story (context)
```

1. **Frame extraction**: scenes and keyframes from the [StoryFrames](https://huggingface.co/datasets/ingoziegler/StoryFrames) dataset
2. **Visual captioning**: `Salesforce/blip2-opt-2.7b` turns each frame into text
3. **Narrative synthesis**: LLaMA 3.3 70B (via the Groq API) weaves the captions into a story under strict constraints: no invented objects, no "camera/video" meta-talk, observational tone

This is a **late-fusion** design: fusion happens in text, not in embeddings.

## Relation to ImageChain
The project is inspired by [ImageChain](https://arxiv.org/abs/2502.19409) (Sánchez Villegas, Ziegler & Elliott, 2025). ImageChain fine-tunes a multimodal LLM to treat an image sequence as a multi-turn conversation. This project **simulates that multi-turn structure without any training**: the story generated for scene *t* is passed as context when writing scene *t+1*. The trade-off is computational efficiency (no fine-tuning, a 2.7B captioner) against visual semantic depth.

## Iterations
| Version | Change | Outcome |
|---|---|---|
| V1: zero-shot | Plain captions → story | Frequent hallucinations (e.g., describing the camera or video quality) |
| V2: few-shot constraints | Strict system prompt + examples | Stories focus on physical actions |
| **V3: context-aware memory** | Previous scene's story passed as context | Temporally coherent multi-scene narratives |

## Evaluation
LLM-as-a-judge (LLaMA 3.3 70B, temperature 0, JSON output) compares each scene's story with the human-annotated description on a 1–10 scale:
- **Semantic similarity**: faithfulness to the ground truth (inspired by ImageChain's SimRate)
- **Coherence**: logical, chronological flow
- **Style**: natural, observational writing

| Run | Semantic similarity | Coherence | Style |
|---|---:|---:|---:|
| V3, 10 stories / 20 scenes (notebook) | 2.50 | 7.75 | 7.10 |
| V3, larger subset (V4) | 2.32 | 7.71 | 6.79 |

![LLM-as-a-judge scores](judge_scores.png)

**Takeaway:** the pipeline writes highly coherent, well-styled narratives, but faithfulness to the human descriptions is low. The bottleneck is the captioner. The narrative LLM is "blind" and knows only what BLIP-2 extracts, so the details humans notice are lost before the story is written.

## Run it
1. Open the notebook in Colab with a GPU runtime (tested on a T4).
2. Get a free API key at [console.groq.com](https://console.groq.com) and add it under Colab **Secrets** as `GROQ_API_KEY`.
3. Run all cells.

## Limitations and future work
- Swap BLIP-2 for a stronger vision-language model to close the semantic-similarity gap.
- The same model family writes and judges the stories, which may bias the scores. Human evaluation or an independent judge would help.
- Add an OpenCV frame extractor so it works on arbitrary local `.mp4` files.

## References
- Sánchez Villegas, Ziegler & Elliott, *ImageChain: Advancing Sequential Image-to-Text Reasoning in Multimodal Large Language Models*, 2025, [arXiv:2502.19409](https://arxiv.org/abs/2502.19409) · [code](https://github.com/danaesavi/ImageChain)
- Li et al., *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*, ICML 2023, [arXiv:2301.12597](https://arxiv.org/abs/2301.12597)
- StoryFrames dataset: [huggingface.co/datasets/ingoziegler/StoryFrames](https://huggingface.co/datasets/ingoziegler/StoryFrames)
