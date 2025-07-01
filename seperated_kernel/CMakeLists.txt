# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
cmake_minimum_required(VERSION 3.1)

# Enable C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Define project name
set(TARGET_NAME mixgemm_plugins)
project(${TARGET_NAME})

set(CMAKE_VERBOSE_MAKEFILE 1)

# Compile options
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -pthread  -std=c++20 -fopenmp -lgomp -lcuda -DENABLE_BF16 -DNDEBUG")

set(CMAKE_C_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -lstdc++")

set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})

set(CMAKE_BUILD_TYPE release)

find_package(CUDA REQUIRED)
message(STATUS "CUDA library status:")
message(STATUS "    config: ${CUDA_HOME}")
message(STATUS "    version: ${CUDA_VERSION}")
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
message(STATUS "---------CUDA_HOME--------------")
message(STATUS ${CUDA_HOME})


include_directories(${CUDA_INCLUDE_DIRS})

 
# Declare the target library.
add_library(
  ${TARGET_NAME} SHARED
  cpp_bindings.cpp
 )

add_subdirectory(kernel)





target_link_libraries(
  ${TARGET_NAME} PUBLIC  cuda kernel )

if(NOT MSVC)
  set_property(TARGET ${TARGET_NAME} PROPERTY LINK_FLAGS "-Wl,--no-undefined")
endif()



