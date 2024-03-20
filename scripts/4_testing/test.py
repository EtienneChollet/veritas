import time
import veritas
import torch
import argparse


if __name__ == "__main__":   

    parser = argparse.ArgumentParser(description='Test UNet model that has been trained in default models dir.')

    parser.add_argument('--version', type=str, default=1,
                    help='model version to test (default: 1). can test many versions seperated by commas.')
    parser.add_argument('--volume', type=str, default='/autofs/cluster/octdata2/users/epc28/data/caroline_data/I46_Somatosensory_20um_crop.nii',
                        help='path to volume you want to predict on.')
    parser.add_argument('--patch-size', type=int, default=256,
                        help='size of UNet (and size of sliding prediction patch)')
    parser.add_argument('--step-size', type=int, default=64,
                        help='step size (in vx) between adjacent prediction patches.')
    
    args = parser.parse_args()
    versions = args.version
    versions = versions.split(',')
    volume = args.volume
    patch_size = args.patch_size
    step_size = args.step_size

    #volume = '/autofs/cluster/octdata2/users/epc28/data/caroline_data/I46_Somatosensory_20um_crop.nii'
    #volume = '/autofs/cluster/octdata2/users/epc28/data/CAA/caa17/occipital/caa17_occipital.nii'
    #volume = '/autofs/cluster/octdata/users/cmagnain/190312_I46_SomatoSensory/I46_Somatosensory_20um_averaging_new.nii'
    #versions = [1, 2, 3, 4]

    for v in versions:
        t1 = time.time()
        with torch.no_grad():
            unet = veritas.Unet(
                model_dir='models',
                version_n=v,
                device='cuda'
                )
            
            unet.load(type='best', mode='test')

            prediction = veritas.RealOctPredict(
                input=volume,
                trainee=unet.trainee,
                dtype=torch.float32,
                device='cuda',
                patch_size=patch_size,
                step_size=step_size,
                normalize=True,
                pad_it=True,
                padding_method='reflect',
                normalize_patches=True,
                )
            
            prediction.predict_on_all()
            out_path = f"{unet.version_path}/predictions"
            prediction.save_prediction(dir=out_path)
            t2 = time.time()
            print(f"Process took {round((t2-t1)/60, 2)} min")
