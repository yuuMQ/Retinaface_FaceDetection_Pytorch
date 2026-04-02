'''
    Swin transformer from scratch:
    - image (H x W x 30)
    -> Patch Partition (partition windows)
    (H/4 x W/4 x 48)
    - Stage 1:
        -> Linear Embedding
        -> Swin Transformer Block (x2)
    (H/4 x W/4 x C)
    - Stage 2:
    Linear Embedding
        -> Patch Merging
        -> Swin Transformer Block (x2)
    (H/8 x W/8 x 2C)
    - Stage 3:
        -> Patch Merging
        -> Swin Transformer Block (x6)
    (H/16 x W/16 x 4C)
    - Stage 4:
        -> Patch Merging
        -> Swin Transformer Block (x2)
    (H/32 x W/32 x 8C)

    -----------------------------------
    - Swin transformer block: replace multi-head self attention (MSA)
     module in Transformer by module based on shifted windows - Window Attention
     - W-MSA
     - SW-MSA
     - MLP head
     - Layer Norm
     -----------------------------------
'''