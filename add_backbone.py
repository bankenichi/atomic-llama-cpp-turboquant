"""
Add the `gemma4_assistant.n_embd_backbone` metadata key (missing from older
assistant GGUFs) by re-serializing the file with the repo's own GGUF copier.

Usage:
    python add_backbone.py <assistant_in.gguf> <assistant_out.gguf> <target_model.gguf | INT>

The 3rd arg is either the target model GGUF (the value is read from its
`gemma4.embedding_length`) or an explicit integer.
"""
import sys
import importlib.util
from pathlib import Path

repo = Path(__file__).parent
sys.path.insert(0, str(repo / "gguf-py"))
import gguf  # noqa: E402

# Reuse the tested copy_with_new_metadata / MetadataDetails from the repo script
gnm_path = repo / "gguf-py" / "gguf" / "scripts" / "gguf_new_metadata.py"
spec = importlib.util.spec_from_file_location("gnm", gnm_path)
gnm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gnm)

assistant_in, assistant_out, third = sys.argv[1], sys.argv[2], sys.argv[3]

if third.isdigit():
    n_bb = int(third)
else:
    tr = gguf.GGUFReader(third, "r")
    fld = tr.get_field("gemma4.embedding_length")
    if fld is None:
        sys.exit("! target model has no 'gemma4.embedding_length' key")
    n_bb = int(fld.contents())

print(f"n_embd_backbone = {n_bb}")

reader = gguf.GGUFReader(assistant_in, "r")
arch = gnm.get_field_data(reader, gguf.Keys.General.ARCHITECTURE)
print(f"arch = {arch}")

writer = gguf.GGUFWriter(assistant_out, arch=arch, endianess=reader.endianess)
align = gnm.get_field_data(reader, gguf.Keys.General.ALIGNMENT)
if align is not None:
    writer.data_alignment = align

new_md = {f"{arch}.n_embd_backbone": gnm.MetadataDetails(gguf.GGUFValueType.UINT32, n_bb)}
gnm.copy_with_new_metadata(reader, writer, new_md, [])
print(f"done -> {assistant_out}")
