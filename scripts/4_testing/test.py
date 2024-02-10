import time
import veritas
import torch

if __name__ == "__main__":    
    t1 = time.time()
    volume = '/autofs/cluster/octdata2/users/epc28/data/caroline_data/I46_Somatosensory_20um_crop.nii'
    #volume = '/autofs/cluster/octdata2/users/epc28/data/CAA/caa26/occipital/caa26_occipital.nii'
    #volume = '/autofs/cluster/octdata/users/cmagnain/190312_I46_SomatoSensory/I46_Somatosensory_20um_averaging_new.nii'

    with torch.no_grad():
        unet = veritas.Unet(
            model_dir='lets_get_small_vessels',
            version_n=1,
            device='cuda'
            )
        
        unet.load(type='last', mode='test')

        prediction = veritas.RealOctPredict(
            input=volume,
            trainee=unet.trainee,
            dtype=torch.float32,
            device='cuda',
            patch_size=64,
            step_size=16,
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