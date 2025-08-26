#!/usr/bin/env python3
# fix_opset_header_to19.py  (可直接執行)
import onnx, os, sys, shutil

SRC = "opset19_fp16_fixed.onnx"           # 改成你要修的檔名（如果不同請修改）
DST = SRC.replace(".onnx", "_op19_only.onnx")
BAK = SRC + ".orig.bak"

if not os.path.exists(SRC):
    print("Source not found:", SRC)
    sys.exit(1)

print("Backing up original file ->", BAK)
# 先用 copy 作備份（避免在內存中修改後再保存覆蓋備份）
if not os.path.exists(BAK):
    shutil.copy2(SRC, BAK)

print("Loading", SRC)
model = onnx.load(SRC)

print("Original opset_import:", [(o.domain, o.version) for o in model.opset_import])

# 建立新的 opset dict（domain -> version）
new_opset = {}
for op in model.opset_import:
    dom = op.domain if op.domain is not None else ""
    # 若是 ai.onnx (domain == "" 或 "ai.onnx")，強制改為 19
    if dom == "" or dom == "ai.onnx":
        new_opset[""] = 19
    else:
        # 若已有相同 domain，保留最高版本（保守處理）
        if dom in new_opset:
            new_opset[dom] = max(new_opset[dom], op.version)
        else:
            new_opset[dom] = op.version

# 若沒有 ai.onnx entry，就補一個 19
if "" not in new_opset:
    new_opset[""] = 19

# 刪除原有 repeated field（注意用 del slice）
del model.opset_import[:]

# 重新加入新的 opset_import 條目
for dom, ver in new_opset.items():
    new = model.opset_import.add()
    # domain 用空字串表示 ai.onnx
    new.domain = dom
    new.version = ver

print("New opset_import:", [(o.domain, o.version) for o in model.opset_import])

# 儲存成新檔
print("Saving fixed model ->", DST)
onnx.save(model, DST)

# run checker
print("Running onnx.checker.check_model ...")
try:
    onnx.checker.check_model(model)
    print("onnx.checker: PASSED")
except Exception as e:
    print("onnx.checker: FAILED ->", e)
    print("Saved file but checker failed; model may use features > opset19.")
    # 仍舊保留輸出檔以便分析

print("Done. Try loading with ONNX Runtime (CPU/CUDA).")
