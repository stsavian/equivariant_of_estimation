
dest_folder_name=ABLATION_ch_networks
dest_path="/home/ssavian/training/plots_ironspeed/${dest_folder_name}"
#########EVALUATE FLOWNET ##################



model_path=/home/ssavian/training/trained_models/main_mirror_baseline/model_best.pth.tar
model_name=flownetC_mir

#bash test_flownetc_v2.sh -a "${model_name}" -b "${model_path}" -c "${dest_path}" -d "sintel kaleidoscope" -e "true"
bash test_flownetc_v2.sh -a "${model_name}_ev_m" -b "${model_path}" -c "${dest_path}" -d "sintel kaleidoscope" -e "false"

model_path=/home/ssavian/training/trained_models/main_no_mirror/model_best.pth.tar
model_name=flownetC_no_mir

#bash test_flownetc_v2.sh -a "${model_name}" -b "$model_path" -c "${plots_pth}" -d "sintel kaleidoscope" -e "true"
bash test_flownetc_v2.sh -a "${model_name}_ev_m" -b "${model_path}" -c "${dest_path}" -d "sintel kaleidoscope" -e "false"




############EVALUATE IRR #############

##TEST IRR
irr_things_pth=/home/ssavian/optical_flow_networks/irr/saved_check_point/pwcnet/IRR-PWC_things3d/checkpoint_best.ckpt
irr_chairs_pth=/home/ssavian/optical_flow_networks/irr/saved_check_point/pwcnet/IRR-PWC_flyingchairsOcc/checkpoint_best.ckpt
#irr_kitti_pth=/home/ssavian/optical_flow_networks/irr/saved_check_point/pwcnet/IRR-PWC_kitti/checkpoint_best.ckpt
#irr_sintel_ft_pth=/home/ssavian/optical_flow_networks/irr/saved_check_point/pwcnet/IRR-PWC_sintel/checkpoint_best.ckpt

model_name=irr_chairs
model_path=$irr_chairs_pth
#bash test_irr_v2.sh -a "${model_name}" -b "${model_path}" -c "${dest_path}" -d "sintel kaleidoscope" -e "true"
bash test_irr_v2.sh -a "${model_name}_ev_m" -b "${model_path}" -c "${dest_path}" -d "sintel kaleidoscope" -e "false"


model_name=irr_things
model_path=$irr_things_pth
#bash test_irr_v2.sh -a "${model_name}" -b "${model_path}" -c "${dest_path}" -d "sintel kaleidoscope" -e "true"
bash test_irr_v2.sh -a "${model_name}_ev_m" -b "${model_path}" -c "${dest_path}" -d "sintel kaleidoscope" -e "false"

############EVALUATE DDFlow #############


model_name=DDFlow
model_path=none
fake_input_size="436 1024 3"
#bash test_ddflow_v2.sh -a "${model_name}" -b "${model_path}" -c "${dest_path}" -d "sintel" -e "true" -f "$fake_input_size"
bash test_ddflow_v2.sh -a "${model_name}_ev_m" -b "${model_path}" -c "${dest_path}" -d "sintel" -e "false"  -f "$fake_input_size"


fake_input_size="512 512 3"
#bash test_ddflow_v2.sh -a "${model_name}" -b "${model_path}" -c "${dest_path}" -d "kaleidoscope" -e "true"  -f "$fake_input_size"
bash test_ddflow_v2.sh -a "${model_name}_ev_m" -b "${model_path}" -c "${dest_path}" -d "kaleidoscope" -e "false"  -f "$fake_input_size"