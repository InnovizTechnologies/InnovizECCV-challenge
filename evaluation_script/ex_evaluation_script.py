from main import evaluate

from pathlib import Path


def run_dev_set():
    root = Path(__file__).parent.parent
    test_annotation_file = root / 'annotations' / 'test_annotations_devsplit.zip'
    user_submission_file = root / 'submission.zip'
    phase_codename = 'dev'
    resutls = evaluate(test_annotation_file, user_submission_file, phase_codename)


def run_eval_set():
    root = Path(__file__).parent.parent
    test_annotation_file = root / 'annotations' / 'innoviz_2022-09-23_eval_gt.zip.enc'
    user_submission_file = root / 'annotations' / 'innoviz_2022-09-23_eval_gt.zip'
    phase_codename = 'test'
    resutls = evaluate(test_annotation_file, user_submission_file, phase_codename)


if __name__ == "__main__":
    # run_dev_set()
    run_eval_set()
