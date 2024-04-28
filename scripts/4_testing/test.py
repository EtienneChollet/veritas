import gc
import time
import veritas
import torch
import argparse

def main(args):
    versions = args.version
    versions = versions.split(',')
    input = args.input
    inputs = input.split(',')
    patch_size = args.patch_size
    redundancy = args.redundancy
    checkpoint = args.checkpoint

    for v in versions:
        for i in inputs:
            t1 = time.time()
            with torch.no_grad():
                unet = veritas.Unet(
                    model_dir='models',
                    version_n=v,
                    device='cuda'
                    )
                
                unet.load(type=checkpoint, mode='test')

                prediction = veritas.data.RealOctPredict(
                    input=i,
                    trainee=unet.trainee,
                    dtype=torch.float32,
                    device='cuda',
                    patch_size=patch_size,
                    redundancy=redundancy,
                    normalize=False,
                    pad_it=True,
                    padding_method='reflect',
                    normalize_patches=True,

                    )
                prediction.predict_on_all()
                out_path = f"{unet.version_path}/predictions"
                prediction.save_prediction(dir=out_path)
                t2 = time.time()
                print(f"Process took {round((t2-t1)/60, 2)} min")
                print('#' * 30, '\n')
                del unet
                del prediction
                torch.cuda.empty_cache()
                gc.collect()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test UNet model that has been trained in default models dir.")

    parser.add_argument('-v', '--version', type=str, default='1',
                        help='Model version to test (default: 1). Can test many versions seperated by commas.')
    parser.add_argument('-i', '--input', type=str, required=False,
                        default='/autofs/cluster/octdata2/users/epc28/data/caroline_data/I46_Somatosensory_20um_crop.nii',
                        help='Path to NIfTI data. Can test many input files seperated by commas.')
    parser.add_argument('-p', '--patch_size', type=int, default=128,
                        help='Size of the Unet input layer and desired patch size (default: 128).')
    parser.add_argument('-r', '--redundancy', type=int, default=3,
                        help='Redundancy factor for prediction overlap (default: 3).')
    parser.add_argument('-c', '--checkpoint', type=str, default='best',
                        help='Checkpoint to load. "best" or "last".')
    
    args = parser.parse_args()
    main(args)