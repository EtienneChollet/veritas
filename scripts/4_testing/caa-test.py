import gc
import time
from veritas.data import CAAPrediction

def main(case, roi, model_n, redundancy, subpatch_ids, model_exp_name='models'):
    t1 = time.time()
    predictor = CAAPrediction(
        case=case,
        roi=roi,
        model_n=model_n,
        redundancy=redundancy,
        subpatch_ids=subpatch_ids,
        model_exp_name=model_exp_name
        )
    print('Starting predictions:', time.time() - t1)
    predictor.predict()
    print('Done predicting:', time.time() - t1)
    predictor.save_niftis()
    print('Done. Total time [s]:', time.time() - t1)

# Ablation models ('models')
#versions = [
#    1, 11, 111, 2, 22, 222, 3, 33, 333, 4, 44, 444, 5, 55, 555, 6, 66, 666,
#    7, 77, 777, 8, 88, 888, 101, 102, 103
#]
# Caroline models ('caroline_models')
# QT for these --> [0.01, 0.99]
#versions = [1, 11, 111, 1111, 11111, 2, 22, 222, 2222, 22222, 3, 33, 333, 3333, 33333]
#versions = [3, 333, 3333]
# CCO models ('cco_models')

if __name__ == '__main__':
    case_roi = {
        'caa6-frontal' : ['caa6', 'frontal'],
        'caa6-occipital' : ['caa6', 'occipital'],
        'caa26-frontal' : ['caa26', 'frontal'],
        'caa26-occipital' : ['caa26', 'occipital'],
    }
    patches_ = {
        'caa6-frontal' : ['3', '4'],
        'caa6-occipital' : ['4', '5'],
        'caa26-frontal' : ['0', '1'],
        'caa26-occipital' : ['3', '8'],
    }

    model_exp_name = 'simple_complex'
    versions = [1601, 1602, 1603, 1604, 1605]

    print('Starting Tests!!!')
    for version in versions:
        for key in case_roi.keys():
            case, roi = case_roi[key]
            patches = patches_[key]
            main(case, roi, version, redundancy=2, subpatch_ids=patches, model_exp_name=model_exp_name)
            gc.collect()
            