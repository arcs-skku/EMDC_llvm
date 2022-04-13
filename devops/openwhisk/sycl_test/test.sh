DOCKER_IMAGE=sunginh/emdc-oneapi-base-cuda:v1.0
ACTION_NAME=sycl_hw
ACTION=Action.zip

#rm $ACTION
#rm exec

SRC=main.cpp
#clang++ -fsycl -fsycl-targets=spir64_x86_64,spir64_fpga,nvptx64-nvidia-cuda $SRC -o exec.exe
zip $ACTION exec exec.exe

wsk action delete $ACTION_NAME
wsk action create -i $ACTION_NAME --docker $DOCKER_IMAGE $ACTION
wsk action invoke -i $ACTION_NAME -b --result --param SYCL_DEVICE_FILTER cuda
