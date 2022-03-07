cmake_minimum_required(VERSION 3.15.0)
project(TorchDGLGraph)

# find_package(PythonInterp REQUIRED)
find_package(Torch REQUIRED)

find_library(DGL_LIB Name dgl PATHS ${CMAKE_SOURCE_DIR}/third_party/dgl/build/)

include_directories("include")
include_directories(third_party/dgl/src)
include_directories(third_party/dgl/include)
include_directories(third_party/dgl/third_party/dlpack/include)
include_directories(third_party/dgl/third_party/dmlc-core/include)

# Define our library target
# add_library(TorchDGLGraph SHARED src/torch_dgl_graph.cc)
add_library(adjmatrix SHARED src/adjmatrix.cc
            src/utils.cc
            src/spmm.cc
            src/edge_softmax.cc
            src/sddmm.cc
            )
add_executable(test src/test.cc)
set(CMAKE_CXX_STANDARD 14)

# Link against LibTorch
# target_link_libraries(TorchDGLGraph "${TORCH_LIBRARIES}" "${DGL_LIB_PATH}")
target_link_libraries(adjmatrix "${TORCH_LIBRARIES}")
target_link_libraries(adjmatrix "${DGL_LIB}")
target_link_libraries(test "${TORCH_LIBRARIES}" "${DGL_LIB}")

# add_executable(inference src/infer.cc)
message(STATUS "DGLLIB: ${DGL_LIB}")
message(STATUS "Torch: ${TORCH_LIBRARIES}")
# # target_link_libraries(inference "${DGL_LIB_PATH}")
# target_link_libraries(inference -Wl,--no-as-needed TorchDGLGraph)
# target_link_libraries(inference "${DGL_LIB_PATH}")
# target_link_libraries(inference "${TORCH_LIBRARIES}")
# target_link_libraries(inference "${DGL_LIB_PATH}")
