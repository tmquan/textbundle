from base_embedding import BaseEmbedding


class WordEmbedding(BaseEmbedding):
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
    we = WordEmbedding()
    words = ["Hello", "World"]
    embeddings = we.embed_input(words)
    for idx, word in enumerate(words):
        print(word, embeddings[idx])

if __name__ == "__main__":
    main()

# Hello     [0.67962 - 0.51435 - 0.15824 ... -0.16643 - 0.20161  0.37423]
# World     [-0.34038 - 0.8863 - 0.10977 ... -0.50555  0.33036  0.00201]
