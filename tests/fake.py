from typing import List

from langchain_community.embeddings import FakeEmbeddings


class FakeImageEmbeddings(FakeEmbeddings):
    """Fake embedding model for images."""

    def embed_image(self, image_path: str) -> List[float]:
        return self._get_embedding()