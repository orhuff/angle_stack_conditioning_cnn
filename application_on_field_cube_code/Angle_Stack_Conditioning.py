
import numpy as np 
import tensorflow as tf
import segyio
from shutil import copyfile


# # FUNCTIONS


# segy copying and writing function
def block_write_mmap(output_segy, input_segy, data_cube, mmap=True):
    copyfile(input_segy, output_segy)

    with segyio.open(output_segy, "r") as segy_src:
        n_xl = len(segy_src.xlines)
        n_il = len(segy_src.ilines)
        n_smpl = len(segy_src.samples)
        expected_shape = (n_il, n_xl, n_smpl)
    with segyio.open(output_segy, "r+", ignore_geometry=True) as segy_dst:
        if mmap:
            mapped_success = segy_dst.mmap()
        if data_cube.shape != expected_shape:
            raise ValueError(
                f"dataset has shape {data_cube.shape} which is not the expected shape {expected_shape}")
        data_cube = np.ascontiguousarray(data_cube, 'float32')
        segy_dst.trace.raw[:] = data_cube.reshape((n_xl * n_il, n_smpl))
        print('Done writing ' + output_segy)
        return mapped_success


# full loading, prediction, and writing function
def condition_angle_stacks(model_h5, t_max, near_inp_segy, mid_inp_segy, far_inp_segy, near_out_segy, mid_out_segy, far_out_segy):
    
    # load original data as numpy arrays
    near_orig = segyio.tools.cube(near_inp_segy)
    mid_orig = segyio.tools.cube(mid_inp_segy)
    far_orig = segyio.tools.cube(far_inp_segy)
    
    # load trained model
    model = tf.keras.models.load_model(model_h5, compile=False)

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
        counter = 1
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

            # print out progress every 400 CDP locations
            if counter % 400 == 0:
                print('XL = '+str(xline+1)+' / '+str(near_orig.shape[1])+'  |  IL = '+ str(inline+1)+' / '+str(near_orig.shape[0]))
            counter += 1
    
    # normalize CNN-predicted amplitudes to same level as average of original amplitudes
    norm_near = np.mean(np.abs(near_orig)) / np.mean(np.abs(near_cnn))
    norm_mid = np.mean(np.abs(mid_orig)) / np.mean(np.abs(mid_cnn))
    norm_far = np.mean(np.abs(far_orig)) / np.mean(np.abs(far_cnn))

    near_cnn *= norm_near
    mid_cnn *= norm_mid
    far_cnn *= norm_far
    
    print(' ')
    print('DONE MAKING PREDICTIONS')
    print(' ')

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
    block_write_mmap(near_out_segy, near_inp_segy, near_cnn_pad, mmap=True)
    block_write_mmap(mid_out_segy, mid_inp_segy, mid_cnn_pad, mmap=True)
    block_write_mmap(far_out_segy, far_inp_segy, far_cnn_pad, mmap=True)
    print('')
    
    return 'ALL FILES WRITTEN'


# # RUN CODE


t_max = 1248

# set these as the paths to the model file, the near/mid/far input segys, and desired name/path for writing results
model_h5 = 'unet-resnet-102030.h5'

near_inp_segy = 'near_cube.sgy'
mid_inp_segy = 'mid_cube.sgy'
far_inp_segy = 'far_cube.sgy'

near_out_segy = 'near_cube_CNN.sgy'
mid_out_segy = 'mid_cube_CNN.sgy'
far_out_segy = 'far_cube_CNN.sgy'

# call function

condition_angle_stacks(model_h5, t_max, near_inp_segy, mid_inp_segy, far_inp_segy, near_out_segy, mid_out_segy, far_out_segy)






