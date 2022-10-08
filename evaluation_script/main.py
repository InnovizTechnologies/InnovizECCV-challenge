import random
import tempfile
from pathlib import Path
import shutil
import time
import numpy as np


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

class IOUBox:    
    """
    source: https://stackoverflow.com/questions/44797713/calculate-the-area-of-intersection-of-two-rotated-rectangles-in-python
    One diff is the counter clock wise rotation used for contour calculation. see 'def contour'
    """
    def __init__(self, x, y, dx, dy, heading):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.heading = heading

    def contour(self):
        """
        If rotate_clockwise is True, positive angles are rotated clockwise
        * For real-world top view (x=north, y=west) rotation around z (from x to y) is ccw
        * For image coordinates top view (x=east, y=south) rotation around z (from x to y) is cw
        """        
        import shapely.geometry
        import shapely.affinity

        w = self.dx
        h = self.dy
        c = shapely.geometry.box(-w/2.0, -h/2.0, w/2.0, h/2.0)
        rc = shapely.affinity.rotate(c, -self.heading)
        return shapely.affinity.translate(rc, self.x, self.y)

    def intersection(self, other):
        return self.contour().intersection(other.contour())

    def intersection_area(self, other):
        return self.intersection(other).area

    @property
    def area(self):
        return self.dx * self.dy

    def iou(self, other):
        intersection_area = self.intersection_area(other)
        return intersection_area / (self.area + other.area - intersection_area + 1e-9)

    @staticmethod
    def from_numpy(nb):
        return IOUBox(x=nb['x'], y=nb['y'], dx=nb['dx'], dy=nb['dy'], heading=nb['heading'])

def tmp_dir(timestamp: int, name: str) -> Path:
    return Path(tempfile.gettempdir()) / f'{timestamp}_{name}'


def calc_xy_iou(target_boxes: np.array, ref_boxes: np.array):
    """ to each box best ref box match is found using xy iou as metric.
    Args:
        tgt_boxes (np.array): target boxes, to each box best match from ref boxes is found
        ref_boxes (np.array): ref boxes, to each box best match from ref boxes is found

    Returns:
        np.array: numpy array in size of tgt_boxes
    """
    # https://stackoverflow.com/questions/44797713/calculate-the-area-of-intersection-of-two-rotated-rectangles-in-python
    # create a result matrix in size of inputs
    results = np.empty((len(target_boxes), len(ref_boxes)), dtype=float)
    for ri, tgt_box in enumerate(target_boxes):
        for ci, ref_box in enumerate(ref_boxes):
            results[ri, ci] = (IOUBox.from_numpy(tgt_box)).iou(IOUBox.from_numpy(ref_box))
    
    # for each gt use iou of detection with max iou (best match)
    ious = np.max(results, axis=1)

    return ious


def calc_xy_iou_from_files(target_file: str or Path, ref_file: str or Path):
    """ to each box at target_file, best ref box match in ref_file  is found using xy iou as metric.
    """
    target_boxes = np.fromfile(target_file, dtype=__GT_BOX_DTYPE__)
    try:
        ref_boxes = np.fromfile(ref_file, dtype=__GT_BOX_DTYPE__)
    except Exception as ex:
        print(f"Failed reading submission file: '{ref_file}'. adding 0 each gt to results compute. Exception: {ex}")
        return np.zeros(len(target_boxes), dtype=float)
    
    print(f'calculating metrics. {len(target_boxes)} target boxes, {len(ref_boxes)} ref boxes')
    ious = calc_xy_iou(target_boxes, ref_boxes)
    print(f'ious: {ious}')

    return ious 


def zero_iou_from_gt_file(gt_file):
    gt_boxes = np.fromfile(gt_file, dtype=__GT_BOX_DTYPE__)
    return np.zeros(len(gt_boxes), dtype=float)


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """
    print("# Install external packages")
    import os
    ecode = os.system("python -m pip install shapely cryptography")
    assert ecode == 0
    
    output = {}
    print(f"# Evaluating for '{phase_codename}' Phase")
    print(f"test_annotation_file '{test_annotation_file}'")
    print(f"user_submission_file '{user_submission_file}'")

    # read input
    print("# Read inputs")
    test_annotation_file = Path(test_annotation_file)
    user_submission_file = Path(user_submission_file)
    assert test_annotation_file.exists()
    assert user_submission_file.exists()
    timestamp = int(time.time() * 1e6) # used for run unique folder name

    tmp_enc_dir = None
    if test_annotation_file.suffix == '.enc':
        print("# Decrypt test annotation file")
        from cryptography.fernet import Fernet
        key = (Path(__file__).parent / 'key.txt').read_bytes()
        tmp_enc_dir = tmp_dir(timestamp, test_annotation_file.name)
        tmp_enc_dir.mkdir()

        # decrypt
        f = Fernet(key)
        data = test_annotation_file.read_bytes()
        data_dec = f.decrypt(data)

        # dump zip file
        test_annotation_file = tmp_enc_dir / test_annotation_file.name.replace('.enc', '.zip')
        with open(test_annotation_file, 'wb') as f:
            f.write(data_dec)

    print(f"Unzip annotation file '{test_annotation_file}'")
    tmp_annotations_dir = tmp_dir(timestamp, test_annotation_file.name)
    shutil.unpack_archive(str(test_annotation_file), str(tmp_annotations_dir))

    print(f"Unzip submission file '{user_submission_file}'")
    tmp_submission_dir = tmp_dir(timestamp, user_submission_file.name)
    shutil.unpack_archive(str(user_submission_file), str(tmp_submission_dir))

    # run evaluation frame by frame
    print("# Run evaluation")
    gt_files = sorted(tmp_annotations_dir.glob('*.bin'))
    print(f'{len(gt_files)} gt files: {[str(f) for f in gt_files]}')
    all_gt_xy_iou = []
    all_det_xy_iou = []
    for gt_file in gt_files:
        print(f"gt file: '{gt_file}'")
        submission_file = tmp_submission_dir / gt_file.name
        if not submission_file.exists():
            print(f"submission file missing: '{submission_file}', adding 0 for each gt to results compute.")
            gt_xy_ious = zero_iou_from_gt_file(gt_file)
            det_xy_iou = zero_iou_from_gt_file(gt_file)
        else:
            print(f"submission file: '{submission_file}'")
            print('calc gt vs det xy ious')
            gt_xy_ious = calc_xy_iou_from_files(gt_file, submission_file)
            print('calc det vs get xy ious')
            det_xy_iou = calc_xy_iou_from_files(submission_file, gt_file)
        
        all_gt_xy_iou += gt_xy_ious.tolist()
        all_det_xy_iou += det_xy_iou.tolist()
        print(f'all_gt_xy_iou len - {len(all_gt_xy_iou)}')
        print(f'all_det_xy_iou len - {len(all_det_xy_iou)}')

    avg_gt_xy_iou = np.mean(all_gt_xy_iou)
    avg_det_xy_iou = np.mean(all_det_xy_iou)
    avg_xy_iou = 0.5 * avg_gt_xy_iou + 0.5 * avg_det_xy_iou

    print(f'# AVG_GT_VS_DET_XY_IOU: {avg_gt_xy_iou}')
    print(f'# AVG_DET_VS_GT_XY_IOU: {avg_det_xy_iou}')
    print(f'# AVG_XY_IOU: {avg_xy_iou}')

    output["result"] = [
        {
            f"{phase_codename}_split": {
                "AVG_XY_IOU": avg_xy_iou,
            }
        }
    ]
    # To display the results in the result file
    output["submission_result"] = output["result"][0][f"{phase_codename}_split"]
    print("# Output")
    print(output)
    
    print("# Cleanups")
    if tmp_annotations_dir.exists():
        print(f"delete '{tmp_annotations_dir}'")
        shutil.rmtree(tmp_annotations_dir)
    if tmp_submission_dir.exists():
        print(f"delete '{tmp_submission_dir}'")
        shutil.rmtree(tmp_submission_dir)
    if tmp_enc_dir is not None and tmp_enc_dir.exists():
        print(f"delete '{tmp_enc_dir}'")
        shutil.rmtree(tmp_enc_dir)

    print(f"# Completed evaluation for '{phase_codename}' Phase")
    return output
