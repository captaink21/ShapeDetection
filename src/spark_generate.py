from pyspark.sql import SparkSession
import os
import json
from task_1 import ShapeGeneratorCV

def generate_partition(index, iterator, out_dir="spark_data"):

    gen = ShapeGeneratorCV(img_size=256, out_dir=None, save_to_disk=False)
    os.makedirs(out_dir, exist_ok=True)

    for i, _ in enumerate(iterator):
        img, anns = gen.generate_image(i + index*1000, return_data=True)
        img_path = os.path.join(out_dir, f"{i+index*1000:06d}.png")
        ann_path = img_path.replace(".png", ".json")

        import cv2
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(anns, f, ensure_ascii=False, indent=2)

        yield img_path

if __name__ == "__main__":
    spark = SparkSession.builder.appName("ShapesGenerator").getOrCreate()

    rdd = spark.sparkContext.parallelize(range(10000), numSlices=10)

    result = rdd.mapPartitionsWithIndex(generate_partition).collect()

    print(f"Сгенерировано {len(result)} изображений")
    spark.stop()
