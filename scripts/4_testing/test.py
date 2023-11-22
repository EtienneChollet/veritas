import time
import veritas
import torch
import os

#os.environ['TORCH_USE_CUDA_DSA'] = '1'
#os.environ['CUDA_LAUNCH_BLOCKING']='1'
#os.environ['PYTORCH_CUDA_ALLOC_CONF']='garbage_collection_threshold:0.9,max_split_size_mb:512'

if __name__ == "__main__":    
    print(f"CUDA available: {torch.cuda.is_available()}")
    t1 = time.time()

    #volume = '/cluster/micro/recon/191124_HK001_CaudalMedulla/Process_new/HK001-CaudalMedulla_20um_averaging.nii'
    # CAA Case
    volume = '/autofs/space/omega_001/users/caa/CAA26_Occipital/Process_caa26_occipital/mus/mus_mean_20um-iso.nii'
    # Caroline's volume
    #volume = '/autofs/cluster/octdata2/users/epc28/data/caroline_data/I46_Somatosensory_20um_crop.nii'
    with torch.no_grad():
        unet = veritas.Unet(version_n=8)
        unet.load(type='best')

        prediction = veritas.RealOctPredict(
            input=volume,
            trainee=unet.trainee,
            dtype=torch.float32,
            device='cuda',
            patch_size=256,
            step_size=64,
            normalize=True,
            pad=True,
            verbose=True,
            )
        
        prediction.predict_on_all()
        out_path = f"{unet.version_path}/predictions"
        prediction.save_prediction(dir=out_path)

        t2 = time.time()
        print(f"Process took {round((t2-t1)/60, 2)} min")