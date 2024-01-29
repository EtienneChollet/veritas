import time
import veritas
import torch

#os.environ['TORCH_USE_CUDA_DSA'] = '1'
#os.environ['CUDA_LAUNCH_BLOCKING']='1'
#os.environ['PYTORCH_CUDA_ALLOC_CONF']='garbage_collection_threshold:0.9,max_split_size_mb:512'

if __name__ == "__main__":    
    print(f"CUDA available: {torch.cuda.is_available()}")
    t1 = time.time()

    volume = '/autofs/cluster/octdata2/users/epc28/data/CAA/caa25/occipital/caa25_occipital.nii'

    with torch.no_grad():
        unet = veritas.Unet(
            model_dir='models',
            version_n=8,
            device='cuda'
            )
        
        unet.load(type='last')

        prediction = veritas.RealOctPredict(
            input=volume,
            trainee=unet.trainee,
            dtype=torch.float32,
            device='cpu',
            patch_size=256,
            step_size=64,
            normalize=True,
            pad_it=True,
            padding_method='reflect',
            )
        
        #for coord_pair in prediction.complete_patch_coords:
        #   print(coord_pair[2])
        #print(prediction.complete_patch_coords)

        prediction.predict_on_all()
        out_path = f"{unet.version_path}/predictions"
        prediction.save_prediction(dir=out_path)
        t2 = time.time()
        print(f"Process took {round((t2-t1)/60, 2)} min")