from pathlib import Path
import time
import shutil
import tempfile

import numpy as np

# Note: duplication defenition in evaluation_script/main.py
__GT_BOX_DTYPE__ = np.dtype([
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('dx', np.float32),
    ('dy', np.float32),
    ('dz', np.float32),
    ('heading', np.float32),
    ('class', np.float32),
])

# Note: duplication defenition in evaluation_script/main.py
def tmp_dir(timestamp: int, name: str) -> Path:
    return Path(tempfile.gettempdir()) / f'eccv_{timestamp}_{name}'



def main():
    annotation_file = Path(__file__).parent.parent / 'submission.zip'
    gen_annotation_file = Path(__file__).parent.parent / 'submission1.zip'
    timestamp = int(time.time() * 1e6) # used for run unique folder name

    print(f"Unzip annotation file '{annotation_file}'")
    tmp_annotations_dir = tmp_dir(timestamp, annotation_file.name)
    shutil.unpack_archive(str(annotation_file), str(tmp_annotations_dir))

    print(f"Gen updated annotation file")
    tmp_gen_annotations_dir = tmp_dir(timestamp, annotation_file.name + "_gen")
    tmp_gen_annotations_dir.mkdir()

    annotation_files = sorted(tmp_annotations_dir.glob('*.bin'))
    print(f'{len(annotation_files)} gt files: {[str(f) for f in annotation_files]}')
    # generate gt for all files except last, with 0.8 dx, 2 missing annotation and 1 extra annotation.
    for ann_file in annotation_files[:-1]:
        print(f'gt file: {ann_file}')
        boxes = np.fromfile(ann_file, dtype=__GT_BOX_DTYPE__)
        boxes['dx'] *= 0.8
        out_path = tmp_gen_annotations_dir / ann_file.name
        # remove 2 annotation (for fa)
        boxes = boxes[:-2]
        # # add synthetic annotation (for fp)
        extra_boxes = np.array([10, 500, 5, 8, 4, 2, 0, 0], dtype=np.float32)
        extra_boxes.dtype = __GT_BOX_DTYPE__
        boxes = np.hstack((boxes, extra_boxes))

        print(f'dumping {out_path}')
        with open(out_path, 'wb') as f:
            f.write(boxes.tobytes())

    shutil.make_archive(gen_annotation_file.stem, 'zip', tmp_gen_annotations_dir)


    print("# Cleanups")
    if tmp_annotations_dir.exists():
        print(f"delete '{tmp_annotations_dir}'")
        shutil.rmtree(tmp_annotations_dir)
    if tmp_gen_annotations_dir.exists():
        print(f"delete '{tmp_gen_annotations_dir}'")
        shutil.rmtree(tmp_gen_annotations_dir)


if __name__ == "__main__":
    main()