# mkhead.py SRC_MTP TARGET_GGUF OUT QTYPE: copy the MTP gguf and add blk.N.nextn.shared_head_head.weight = the target
# output.weight, either copied raw (QTYPE=copy) or requantized from it (QTYPE=Q4_0, Q5_0, Q8_0, ...), for draft passes only.
# TARGET_GGUF is the target model file (or the shard of a split model) that holds output.weight.
import sys  # needs the gguf package: pip install gguf, or <llama.cpp>/gguf-py on PYTHONPATH
import numpy as np, gguf
src, tgt, out, qt = sys.argv[1:5]
r = gguf.GGUFReader(src)
head = next(t for t in gguf.GGUFReader(tgt).tensors if t.name == "output.weight")
il = next(int(t.name.split(".")[1]) for t in r.tensors if ".nextn." in t.name)
name = f"blk.{il}.nextn.shared_head_head.weight"
assert all(t.name != name for t in r.tensors), "already has a draft head"
if qt == "copy":
    data, ttype = np.array(head.data), head.tensor_type
else:
    ttype = gguf.GGMLQuantizationType[qt]
    f = gguf.quants.dequantize(head.data, head.tensor_type)
    data = gguf.quants.quantize(f.astype(np.float32), ttype); del f
arch = r.fields["general.architecture"].contents()
w = gguf.GGUFWriter(out, arch=arch, endianess=r.endianess)
for field in r.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
        continue
    vt = field.types[0]; st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    w.add_key_value(field.name, field.contents(), vt, sub_type=st)
for t in r.tensors:
    w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.add_tensor_info(name, data.shape, data.dtype, data.nbytes, ttype)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t in r.tensors:
    w.write_tensor_data(t.data, tensor_endianess=r.endianess)
w.write_tensor_data(data, tensor_endianess=r.endianess)
w.close()
print(out, name, ttype.name, data.shape, round(data.nbytes / 2**20), "MiB")
