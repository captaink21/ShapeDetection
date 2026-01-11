import os
import json
import random
import numpy as np
import cv2


class ShapeGeneratorCV:
    def __init__(self, img_size=256, out_dir="data", seed=None,
                 allowed_shapes=None, must_include=None, save_to_disk=True):

        self.img_size = img_size
        self.out_dir = out_dir
        self.save_to_disk = save_to_disk
        if save_to_disk:
            self.img_dir = os.path.join(out_dir, "images")
            self.ann_dir = os.path.join(out_dir, "annotations")
            os.makedirs(self.img_dir, exist_ok=True)
            os.makedirs(self.ann_dir, exist_ok=True)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.all_shapes = ["triangle", "rhombus", "circle", "hexagon"]

        self.allowed_shapes = allowed_shapes if allowed_shapes is not None else self.all_shapes
        self.must_include = must_include if must_include is not None else []

    def _random_color(self, exclude=None):
        while True:
            color = tuple(np.random.randint(0, 256, size=3).tolist())
            if color != exclude:
                return color

    def _get_polygon_points(self, shape_type, w, h):
        if shape_type == "triangle":
            return np.array([[w // 2, 0], [0, h], [w, h]], np.int32)
        elif shape_type == "rhombus":
            return np.array([[w // 2, 0], [0, h // 2], [w // 2, h], [w, h // 2]], np.int32)
        elif shape_type == "hexagon":
            pts = []
            for theta in np.linspace(0, 2 * np.pi, 6, endpoint=False):
                x = w / 2 + (w / 2) * np.cos(theta)
                y = h / 2 + (h / 2) * np.sin(theta)
                pts.append((int(x), int(y)))
            return np.array(pts, np.int32)
        return None

    def _draw_shape(self, shape_type, w, h, color, angle):
        temp = np.zeros((h, w, 3), dtype=np.uint8)

        if shape_type == "circle":
            cv2.circle(temp, (w // 2, h // 2), min(w, h) // 2, color, -1)
        else:
            pts = self._get_polygon_points(shape_type, w, h)
            if pts is not None:
                cv2.fillPoly(temp, [pts], color)

        (hh, ww) = temp.shape[:2]
        center = (ww // 2, hh // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int((hh * sin) + (ww * cos))
        new_h = int((hh * cos) + (ww * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        rotated = cv2.warpAffine(
            temp, M, (new_w, new_h), borderValue=(0, 0, 0))

        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(gray)
        if coords is None:
            return None, None
        x, y, rw, rh = cv2.boundingRect(coords)
        return rotated, (x, y, rw, rh)

    def _check_overlap(self, bbox, existing_boxes):
        x1, y1, w1, h1 = bbox
        for (x2, y2, w2, h2) in existing_boxes:
            if not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1):
                return True
        return False

    def generate_image(self, idx=0, return_data=False):
        while True:
            bg_color = self._random_color()
            image = np.full((self.img_size, self.img_size, 3),
                            bg_color, dtype=np.uint8)

            n_shapes = random.randint(1, 5)
            annotations = []
            existing_boxes = []

            # для 3 задания
            for must_shape in self.must_include:
                w = random.randint(25, 150)
                h = random.randint(25, 150)
                angle = random.randint(0, 359)
                rotated, bbox = self._draw_shape(
                    must_shape, w, h, self._random_color(
                        exclude=bg_color), angle
                )
                if rotated is None or bbox is None:
                    continue
                bx, by, rw, rh = bbox
                if rw < 25 or rh < 25 or rw > 150 or rh > 150:
                    continue
                x = random.randint(0, self.img_size - rw)
                y = random.randint(0, self.img_size - rh)
                if x + rw > self.img_size or y + rh > self.img_size:
                    continue

                roi = rotated[by:by + rh, bx:bx + rw]
                mask = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, mask_bin = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
                roi_bg = image[y:y + rh, x:x + rw]
                roi_fg = cv2.bitwise_and(roi, roi, mask=mask_bin)
                roi_bg = cv2.bitwise_and(
                    roi_bg, roi_bg, mask=cv2.bitwise_not(mask_bin))
                image[y:y + rh, x:x + rw] = cv2.add(roi_bg, roi_fg)

                existing_boxes.append((x, y, rw, rh))
                annotations.append({
                    "id": len(annotations) + 1,
                    "name": must_shape,
                    "x": x, "y": y, "w": rw, "h": rh,
                    "angle": angle
                })

            for i in range(n_shapes):
                shape_type = random.choice(self.allowed_shapes)
                w = random.randint(25, 150)
                h = random.randint(25, 150)
                angle = random.randint(0, 359)
                for attempt in range(100):
                    x = random.randint(0, self.img_size - w)
                    y = random.randint(0, self.img_size - h)
                    rotated, bbox = self._draw_shape(
                        shape_type, w, h, self._random_color(
                            exclude=bg_color), angle
                    )
                    if rotated is None or bbox is None:
                        continue
                    bx, by, rw, rh = bbox
                    if rw < 25 or rh < 25 or rw > 150 or rh > 150:
                        continue
                    if x + rw > self.img_size or y + rh > self.img_size:
                        continue
                    if self._check_overlap((x, y, rw, rh), existing_boxes):
                        continue

                    roi = rotated[by:by + rh, bx:bx + rw]
                    mask = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    _, mask_bin = cv2.threshold(
                        mask, 1, 255, cv2.THRESH_BINARY)
                    roi_bg = image[y:y + rh, x:x + rw]
                    roi_fg = cv2.bitwise_and(roi, roi, mask=mask_bin)
                    roi_bg = cv2.bitwise_and(
                        roi_bg, roi_bg, mask=cv2.bitwise_not(mask_bin))
                    image[y:y + rh, x:x + rw] = cv2.add(roi_bg, roi_fg)

                    existing_boxes.append((x, y, rw, rh))
                    annotations.append({
                        "id": len(annotations) + 1,
                        "name": shape_type,
                        "x": x, "y": y, "w": rw, "h": rh,
                        # "angle": angle
                    })
                    break

            if annotations:
                break

        if return_data:
            return image, annotations

        if self.save_to_disk:
            img_path = os.path.join(self.img_dir, f"{idx:03d}.png")
            ann_path = os.path.join(self.ann_dir, f"{idx:03d}.json")
            cv2.imwrite(img_path, image)
            with open(ann_path, "w", encoding="utf-8") as f:
                json.dump(annotations, f, indent=2, ensure_ascii=False)

    def generate_dataset(self, n=100):
        for i in range(1, n + 1):
            self.generate_image(i)
        print(
            f"Сгенерировано {n} изображений и JSON-аннотаций в {self.out_dir}")


if __name__ == "__main__":
    gen = ShapeGeneratorCV(out_dir="dataset", seed=42)
    gen.generate_dataset(n=100)


