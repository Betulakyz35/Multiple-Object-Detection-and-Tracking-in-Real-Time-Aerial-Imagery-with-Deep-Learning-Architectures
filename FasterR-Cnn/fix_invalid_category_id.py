import json
import os

JSON_PATHS = {
    "visdrone": {
        "path": "visdrone_val_coco.json",
        "valid_ids": list(range(1, 11))  # 1–10
    },
    "dota": {
        "path": "dota_val_coco.json",
        "valid_ids": list(range(11, 26))  # 11–25
    }
}

def fix_json(json_path, valid_ids, output_path):
    print(f"\nKontrol ediliyor: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    total_ann = len(data["annotations"])
    fixed_ann = []
    removed = 0

    for ann in data["annotations"]:
        cid = ann["category_id"]
        if cid in valid_ids:
            fixed_ann.append(ann)
        else:
            removed += 1
            print(f"⚠️ Hatalı category_id bulundu: {cid} (image_id={ann['image_id']})")

    data["annotations"] = fixed_ann

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"✅ {removed} geçersiz annotation kaldırıldı.")
    print(f"💾 Yeni dosya kaydedildi: {output_path}")


def main():
    for name, config in JSON_PATHS.items():
        path = config["path"]
        valid_ids = config["valid_ids"]

        if not os.path.exists(path):
            print(f"⛔ Dosya bulunamadı: {path}")
            continue

        out_path = path.replace(".json", "_fixed.json")
        fix_json(path, valid_ids, out_path)


if __name__ == "__main__":
    main()
