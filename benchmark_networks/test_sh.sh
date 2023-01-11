#path=/home/ssavian/training
#echo path
#path1=$path
#echo $path1

folder_pth='/Users/stefano/Desktop/microtec'
results_pth="/home/ssavian/training/plots_ironspeed/${name}"
results_summary_pth="/home/ssavian/training/plots_ironspeed/${name}_summary"
plots_pth="/home/ssavian/training/plots_ironspeed/${name}_plots"
echo $absolute_path
#echo $train_pth
folder_pth='/scratch/ssavian/transfer/MODELS_chairs/FWDS_mir'
for filename in "$folder_pth"/*; do
  #echo  $filename
  path=$FWDS_mir_root/$filename/$filename.pth
  echo "$filename/$(basename $filename).pth"
done