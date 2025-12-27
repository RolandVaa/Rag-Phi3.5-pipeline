def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    text = text.replace("\r\n", "\n")
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_size)
        chunks.append(text[i:j])
        if j >= n:
            break
        i = max(0, j - overlap)
    return chunks
