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


def calc_xy_iou(gt_boxes: np.array, det_boxes: np.array):
    # https://stackoverflow.com/questions/44797713/calculate-the-area-of-intersection-of-two-rotated-rectangles-in-python
    print(f'calculating metrics. {len(gt_boxes)} gt_boxes, {len(det_boxes)} detected boxes')
    # create a result matrix in size of inputs
    results = np.empty((len(gt_boxes), len(det_boxes)), dtype=float)
    for ri, gt_box in enumerate(gt_boxes):
        for ci, det_box in enumerate(det_boxes):
            results[ri, ci] = (IOUBox.from_numpy(gt_box)).iou(IOUBox.from_numpy(det_box))
    
    # for each gt use iou of detection with max iou (best match)
    ious = np.max(results, axis=1)
    print(f'ious: {ious}')

    return ious


def calc_xy_iou_from_files(gt_file, submission_file):
    gt_boxes = np.fromfile(gt_file, dtype=__GT_BOX_DTYPE__)
    detected_boxes = np.fromfile(submission_file, dtype=__GT_BOX_DTYPE__)
    return calc_xy_iou(gt_boxes, detected_boxes)


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
    output = {}
    print(f"# Evaluating for '{phase_codename}' Phase")

    # read input
    print("# Read inputs")
    test_annotation_file = Path(test_annotation_file)
    user_submission_file = Path(user_submission_file)
    assert test_annotation_file.exists()
    assert user_submission_file.exists()
    print(f"Unzip gt '{test_annotation_file}'")
    timestamp = int(time.time() * 1e6) # used for run unique folder name
    tmp_gt_dir = tmp_dir(timestamp, test_annotation_file.name)
    shutil.unpack_archive(str(test_annotation_file), str(tmp_gt_dir))

    print(f"Unzip gt '{user_submission_file}'")
    tmp_submission_dir = tmp_dir(timestamp, user_submission_file.name)
    shutil.unpack_archive(str(user_submission_file), str(tmp_submission_dir))

    # run evaluation frame by frame
    print("# Run evaluation")
    all_xy_iou = []
    for gt_file in sorted(tmp_gt_dir.glob('*.bin')):
        print(f"gt file: '{gt_file}'")
        submission_file = tmp_submission_dir / gt_file.name
        if not submission_file.exists():
            # todo: add to results [0] * gt's
            print(f"submission file missing: '{submission_file}', adding 0 to results compute.")
        else:
            print(f"submission file: '{submission_file}'")
            xy_ious = calc_xy_iou_from_files(gt_file, submission_file)
            all_xy_iou += xy_ious.tolist()
            print(f'all_xy_iou len - {len(all_xy_iou)}')
    
    avg_xy_iou = np.mean(all_xy_iou)
    print(f'# AVG_XY_IOU: {avg_xy_iou}')

    output["result"] = [
        {
            "test_split": {
                "AVG_XY_IOU": avg_xy_iou,
            }
        }
    ]
    # To display the results in the result file
    output["submission_result"] = output["result"][0]["test_split"]

    print("# Cleanups")
    print(f"delete '{tmp_gt_dir}'")
    shutil.rmtree(tmp_gt_dir)
    print(f"delete '{tmp_submission_dir}'")
    shutil.rmtree(tmp_submission_dir)

    print(f"# Completed evaluation for '{phase_codename}' Phase")
    return output
