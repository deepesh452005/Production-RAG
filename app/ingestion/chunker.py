"""
Phase 1 chunker: naive fixed-size splitting with overlap.
Phase 2 will replace this with semantic/recursive chunking.
"""
# from app.config import settings


# def chunk_text(text: str, source: str) -> list[dict]:
#     """
#     Splits text into overlapping chunks.
#     Returns list of {"text": chunk, "source": source, "chunk_id": idx}
#     """
#     size = settings.chunk_size
#     overlap = settings.chunk_overlap
#     chunks = []

#     start = 0
#     idx = 0
#     while start < len(text):
#         end = start + size
#         chunk = text[start:end]
#         chunks.append({
#             "text": chunk,
#             "source": source,
#             "chunk_id": f"{source}_{idx}",
#         })
#         start += size - overlap
#         idx += 1

#     return chunks

"""
Phase 2 chunker: recursive chunking that respects natural text boundaries.

Instead of blindly cutting every N characters (Phase 1), this tries a
hierarchy of separators, largest structural unit first:
  1. paragraph breaks ("\n\n")
  2. line breaks ("\n")
  3. spaces (word boundaries)
  4. character-level (last resort, only if a single "word" is still too long)

This keeps whole sentences/paragraphs together whenever the size budget
allows, only forcing an unnatural cut when there's no other option.
"""
from app.config import settings

SEPARATORS = ["\n\n", "\n", " ", ""]  # ordered from largest to smallest unit


def _split_recursive(text: str, size: int, separators: list[str]) -> list[str]:
    """
    Recursively splits text on the first separator that actually helps.
    Returns pieces that are each <= size wherever possible.
    """
    if len(text) <= size:
        return [text] if text else []

    sep = separators[0]
    remaining_separators = separators[1:]

    if sep == "":
        # Last resort: hard character cut (only hit if a single "word" exceeds size)
        return [text[i:i + size] for i in range(0, len(text), size)]

    parts = text.split(sep)

    pieces = []
    current = ""
    for part in parts:
        candidate = (current + sep + part) if current else part
        if len(candidate) <= size:
            current = candidate
        else:
            if current:
                pieces.append(current)
            if len(part) <= size:
                current = part
            else:
                # this single part is still too big — recurse with a finer separator
                pieces.extend(_split_recursive(part, size, remaining_separators))
                current = ""
    if current:
        pieces.append(current)

    return pieces


def _add_overlap(pieces: list[str], overlap: int) -> list[str]:
    """
    Prepends a tail of the previous piece to each piece, so context
    isn't lost at boundaries (same goal as Phase 1's overlap, applied
    on top of natural-boundary pieces instead of raw character slices).
    """
    if overlap <= 0 or len(pieces) <= 1:
        return pieces

    result = [pieces[0]]
    for i in range(1, len(pieces)):
        prev_tail = pieces[i - 1][-overlap:]
        result.append(prev_tail + pieces[i])
    return result


def chunk_text(text: str, source: str) -> list[dict]:
    """
    Splits text into chunks along natural boundaries (paragraphs > lines >
    words > characters), then adds overlap between consecutive chunks.
    Returns list of {"text": chunk, "source": source, "chunk_id": idx}
    """
    size = settings.chunk_size
    overlap = settings.chunk_overlap

    raw_pieces = _split_recursive(text.strip(), size, SEPARATORS)
    pieces = _add_overlap(raw_pieces, overlap)

    chunks = []
    for idx, piece in enumerate(pieces):
        chunks.append({
            "text": piece,
            "source": source,
            "chunk_id": f"{source}_{idx}",
        })

    return chunks
