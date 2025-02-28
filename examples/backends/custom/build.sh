rm -rf build
mkdir build && cd build

cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_BACKEND_REPO_TAG=r23.12 -DTRITON_CORE_REPO_TAG=r23.12 -DTRITON_COMMON_REPO_TAG=r23.12 ..
make install

mkdir -p /opt/tritonserver/backends/projection_logic
cp --preserve=links install/backends/recommended/libtriton_recommended.so /data/repos/viztel-shopassist-prod/triton-model-dir/projection_logic/libtriton_projection_logic.so
cp --preserve=links install/backends/recommended/libtriton_recommended.so /opt/tritonserver/backends/projection_logic/libtriton_projection_logic.so
