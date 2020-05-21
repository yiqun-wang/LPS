# LPS

This code implements a deep learning method on **Local Point Signature** for 3D surface shape matching described in our CVPR 2019 paper:

["A Robust Local Spectral Descriptor for Matching Non-Rigid Shapes with Incompatible Shape Structures"](http://openaccess.thecvf.com/content_CVPR_2019/html/Wang_A_Robust_Local_Spectral_Descriptor_for_Matching_Non-Rigid_Shapes_With_CVPR_2019_paper.html) 

by Yiqun Wang, Jianwei Guo, Dong-Ming Yan, Kai Wang, Xiaopeng Zhang.

[Project Page](http://www.nlpr.ia.ac.cn/ivc/project/specmathcing/)

Please consider citing the above paper if you use the code/program (or part of it). 


## Usage	  
	  
There are three folders here. The "cpp" calls the "matlab" for GI generation. The "python" is used for network training and testing.

The usage is as follows:

1. Compile matlab project: 

		MCC matlab "mcc -W cpplib:libcompcur -T link:lib compute_curvature.m". We got the libcompcur.dll that is going to be added to the CPP project.


2. Build cpp solution: this code is to generate geometry images. You can run this step in your local desktop.

		Modify CMakeLists: 
			add include_directories and link_directories for openmesh and matlab runtime
		Cmake
		Build solution
		Modify config.ini for mesh_dir(directory of OFF models) gi_dir(directory of geometry images) and kpi_dir(directory of key points, you can skip it for dense matching)
			edit other paras such as gi_size(NxN of gi), rotation_num and radius_list_p(the ratio of geodesic diameter).
		Add "libcompcur.dll" to folder with GIGen.exe
		Run "GIGen.exe config.ini" to generate GI

	
3. Python project: this code is to train and test network. You should copy the geometry images generated in last step into the server.
	
	3.1 Train network:

		run "classify_gi_by_pidx_and_split.py" to classify dataset by point index for generating Tfrecord
			source_dir is the folder of geometry images(rotation_num=12) for training, destination_dir is the generated folder after splitting
		run "tfr_gen.py" to generate Tfrecord
			gi_dir is the same as destination_dir, tfr_dir is the generated folder of Tfrecord.
		run "train_softmax6890.py" to pretrain a classification network
			tfr_dir needs to be specified
		run "train_mincv_perloss6890_256.py" to train the triplet network by restoring a pre-trained classification model
			Tfr_dir needs to be specified
		
	3.2 Test to generate descriptor:
	
		run "descGen.py" to generate descriptor using test dataset
			gi_dir is the folder of geometry images(rotation_num=1) for testing, desc_dir is the generated folder of descriptor.
	

## License

This program is free software; you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation; either version 2 of 
the License, or (at your option) any later version. 
