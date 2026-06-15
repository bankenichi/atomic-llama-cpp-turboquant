import sys

# Same-length byte swap: "gemma4-assistant" -> "gemma4_assistant".
# Because the byte length is identical, every GGUF tensor/data offset stays
# valid, so this is safe to do without re-serializing the file.

src, dst = sys.argv[1], sys.argv[2]
OLD = b"gemma4-assistant"
NEW = b"gemma4_assistant"
assert len(OLD) == len(NEW), "lengths must match for in-place-safe swap"

with open(src, "rb") as f:
    data = f.read()

count = data.count(OLD)
print(f"found {count} occurrence(s) of {OLD!r}")
data = data.replace(OLD, NEW)

with open(dst, "wb") as f:
    f.write(data)

print(f"wrote {dst} ({len(data)} bytes), replaced -> {NEW!r}")
