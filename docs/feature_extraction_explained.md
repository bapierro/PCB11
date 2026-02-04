# CNN Feature Extraction Explained

This document explains how we extract features from images using a neural network.

## How It Works

```mermaid
flowchart TB
    subgraph Input
        A["Image (e.g., a cat photo)"]
    end
    
    subgraph CNN["Neural Network (ResNet50)"]
        direction TB
        B["Layer 1"]
        C["Layer 2"]
        D["Layer 3"]
        E["Layer 4"]
    end
    
    subgraph Pooling["Global Average Pooling"]
        F["Compress spatial info into a single vector"]
    end
    
    subgraph Output["Feature Vectors"]
        G1["Layer 1: 256 numbers"]
        G2["Layer 2: 512 numbers"]
        G3["Layer 3: 1024 numbers"]
        G4["Layer 4: 2048 numbers"]
    end
    
    A --> B --> C --> D --> E
    B --> F
    C --> F
    D --> F
    E --> F
    F --> G1
    F --> G2
    F --> G3
    F --> G4
```

## What Each Step Does

| Step | What Happens | Analogy |
|------|--------------|---------|
| **Input** | Feed an image to the network | Showing a photo to someone |
| **Layer 1-2** | Detect simple features | Noticing lines and textures |
| **Layer 3-4** | Detect complex features | Recognizing "this is a cat" |
| **Pooling** | Summarize each layer | Writing a brief description |
| **Output** | Numbers describing the image | A fingerprint for the image |

> [!TIP]
> **Why multiple layers?** Early layers see simple things (edges), late layers see complex things (objects). By extracting from multiple layers, we can compare how the brain processes images at different levels of abstraction.

## Why This Matters for RSA

Once we have these feature vectors, we can:
1. Compare how **similar** two images are according to each layer
2. Build a **similarity matrix** for all images
3. Compare the CNN's similarity judgments with **human brain data** (fMRI/MEG)

This helps us understand if the neural network "sees" images the same way humans do!
