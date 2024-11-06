
from cegalprizm.pycoderunner import WorkflowDescription,DomainObjectsEnum, MeasurementNamesEnum

# Start: PWR Description
from cegalprizm.pycoderunner import WorkflowDescription,DomainObjectsEnum, MeasurementNamesEnum

pwr_description = WorkflowDescription(name="Angle stack conditioning",
                                      category="GeoLab (seismic processing)",
                                      description="Automated residual Moveout Correction of angle stacks with a pretrained CNN. Outputs new conditioned offset volumes. Can improve resolution particularily on far offset stacks",
                                      authors="Owen Huff",
                                      version="1.0")


# parameter 1: seismic volume from project
pwr_description.add_object_ref_parameter(name='n_seismic',object_type=DomainObjectsEnum.Seismic3D,label='near',description='near offset stack (near, mid and far must be same size')

# parameter 2: 
pwr_description.add_object_ref_parameter(name='m_seismic',object_type=DomainObjectsEnum.Seismic3D,label='mid',description='mid offset stack (near, mid and far must be same size')

# parameter 3: 
pwr_description.add_object_ref_parameter(name='f_seismic',object_type=DomainObjectsEnum.Seismic3D,label='far',description='far offset stack (near, mid and far must be same size')


# End: PWR Description

#%%
from cegalprizm.pythontool import PetrelConnection, SurfaceAttribute
from typing import cast
ptp = PetrelConnection()

print('load seismic input volume and create empty output volumes...')
near=ptp.get_petrelobjects_by_guids([parameters['n_seismic']])[0]
mid=ptp.get_petrelobjects_by_guids([parameters['m_seismic']])[0]
far=ptp.get_petrelobjects_by_guids([parameters['f_seismic']])[0]

output_seismic_name='Conditioned_near'
output_near=near.clone(output_seismic_name,copy_values=False)  # copy value False is faster and will give empty cube

output_seismic_name='Conditioned_mid'
output_mid=near.clone(output_seismic_name,copy_values=False)  # copy value False is faster and will give empty cube

output_seismic_name='Conditioned_far'
output_far=near.clone(output_seismic_name,copy_values=False)  # copy value False is faster and will give empty cube



#%%
import numpy as np 
import tensorflow as tf

#%%
# full loading, prediction, and writing function
def condition_angle_stacks(model_h5, t_max, near, mid, far, output_near, output_mid, output_far):
    
    
    
    print('load the cnn model...')
    # load trained model
    model = tf.keras.models.load_model(model_h5, compile=False)
    
    size=20
    t_range=(0,near.extent[2]-1)
    
    for il in range(0, near.extent[0]-size, size):
        il_range=(il,il+size-1)
        print('printing inline progess...')
        print(il)
        for xl in range(0, near.extent[1]-size, size):
            xl_range=(xl,xl+size-1)
            
        
            xl_range=(xl,xl+size)  #(0,near.extent[1]-1)
            il_range=(il, il+size) #(0,near.extent[0]-1)
            
            near_orig = near.chunk(il_range, xl_range, t_range).as_array()
            mid_orig = mid.chunk(il_range, xl_range, t_range).as_array()
            far_orig = far.chunk(il_range, xl_range, t_range).as_array()
            #seismic_chunk = np.rot90(seismic_chunk, 1, (0, 2))
            
            # retain original time length of data
            t_orig = near_orig.shape[2]
            
            # if the original seismic data has fewer time samples than the tmax of the model
            if t_orig < t_max:
                
                # pad the data with zeros at the end to make it the same length tmax
                zeros_to_add_before = t_max - t_orig
                zero_array_before = np.zeros((near_orig.shape[0], near_orig.shape[1], zeros_to_add_before))
                near_orig = np.concatenate((near_orig, zero_array_before), axis=2)
                mid_orig = np.concatenate((mid_orig, zero_array_before), axis=2)
                far_orig = np.concatenate((far_orig, zero_array_before), axis=2)
                
            # if the original data has more samples (or the same) as tmax
            else:
        
                near_orig = near_orig[:, :, :t_max]
                mid_orig = mid_orig[:, :, :t_max]
                far_orig = far_orig[:, :, :t_max]
        
            # initialize arrays for cnn prediction
            near_cnn = np.zeros(near_orig.shape)
            mid_cnn = np.zeros(mid_orig.shape)
            far_cnn = np.zeros(far_orig.shape)
            
            # iterate over crossline number
            for xline in range(near_orig.shape[1]):
                # iterate over inline number
                for inline in range(near_orig.shape[0]):
        
                    # combine near,mid,far into stacked array, and reshape for tensorflow format
                    orig_stacks = np.stack((near_orig[inline, xline, :], mid_orig[inline, xline, :], far_orig[inline, xline, :]), axis=1)
                    orig_stacks = np.reshape(orig_stacks, (1, t_max, 3, 1))
        
                    # make prediction with model, and extract it from tensorflow format
                    pred_stacks = model.predict(orig_stacks, verbose=False)
                    pred_stacks = pred_stacks[0,:,:]      
        
                    # fill in cnn prediction arrays
                    near_cnn[inline, xline, :] = pred_stacks[:,0]
                    mid_cnn[inline, xline, :] = pred_stacks[:,1]
                    far_cnn[inline, xline, :] = pred_stacks[:,2]
                    
            # normalize CNN-predicted amplitudes to same level as average of original amplitudes
            
            
            norm_near = np.mean(np.abs(near_orig)) / np.mean(np.abs(near_cnn))
            norm_mid = np.mean(np.abs(mid_orig)) / np.mean(np.abs(mid_cnn))
            norm_far = np.mean(np.abs(far_orig)) / np.mean(np.abs(far_cnn))
        
            near_cnn *= norm_near
            mid_cnn *= norm_mid
            far_cnn *= norm_far
            
            
            #print(' ')
            #print('DONE MAKING PREDICTIONS')
            #print(' ')
        
            # if the original seismic data had more time samples than the tmax of the model
            if t_orig > t_max:
                # calculate number of zeros to add back to array, for writing back to initial segy geometry
                zeros_to_add_after = t_orig - t_max
        
                # pad the prediction arrays past 1248 samples with zeros
                zero_array_after = np.zeros((near_orig.shape[0], near_orig.shape[1], zeros_to_add_after))
                near_cnn_pad = np.concatenate((near_cnn, zero_array_after), axis=2)
                mid_cnn_pad = np.concatenate((mid_cnn, zero_array_after), axis=2)
                far_cnn_pad = np.concatenate((far_cnn, zero_array_after), axis=2)
                
            # if the original seismic data had fewer time samples (or the same) than the tmax of the model    
            else:
                
                near_cnn_pad = near_cnn[:, :, :t_orig]
                mid_cnn_pad = mid_cnn[:, :, :t_orig]
                far_cnn_pad = far_cnn[:, :, :t_orig]
                
                
            # write near, mid, and far segy prediction files
            # Y = np.rot90(Y, -1, (0, 2))
            output_near.chunk(il_range, xl_range, t_range).set(near_cnn_pad)
            output_mid.chunk(il_range, xl_range, t_range).set(mid_cnn_pad)
            output_far.chunk(il_range, xl_range, t_range).set(far_cnn_pad)
                                                                  
                                                                  
            
            print('')
    
    return 'DONE'


#%%

t_max = 1248

# set these as the paths to the model file, the near/mid/far input segys, and desired name/path for writing results
model_h5 = 'unet-resnet-102030.h5'

condition_angle_stacks(model_h5, t_max, near, mid, far, output_near, output_mid, output_far)



#%%

