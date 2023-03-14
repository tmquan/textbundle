from base_embedding import BaseEmbedding


class SentEmbedding(BaseEmbedding):
    def __init__(
        self,
        model_name='bert-large-cased',
        seed=42,
        **kwarg
    ) -> None:
        super().__init__(model_name, seed, **kwarg)  # type: ignore


def main():
    # import string
    import numpy as np
    np.set_printoptions(
        precision=5,
        threshold=5
    )
    se = SentEmbedding()
    sentences = ["Hello World!", "First sentence.", "Second sentence."]
    embeddings = se.embed_input(sentences)
    for idx, sentence in enumerate(sentences):
        print(sentence, embeddings[idx])

if __name__ == "__main__":
    main()

# Hello World!        [-0.38242 - 0.10765 - 0.32443 ... -0.17909 - 0.37729 - 0.68174]
# First sentence.     [0.07853 - 0.30699 - 0.25504 ...  0.03224 - 0.10972 - 0.07887]
# Second sentence.    [-0.18998 - 0.31491 - 0.05334 ... -0.01183 - 0.12456 - 0.27109]
