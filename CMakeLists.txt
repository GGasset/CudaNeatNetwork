cmake_minimum_required(VERSION 3.28)

# In linux add CPP13 compilation flag. I recommend homebrew to install gcc@13
# In windows compile inside Visual Studio.

function(add_prefix prefix_string in_list_name out_list_name)
	SET(tmp_output)
	foreach(in_list_file_iter ${${in_list_name}})

		SET(tmp_prefixed "${prefix_string}${in_list_file_iter}")
		list(APPEND tmp_output ${tmp_prefixed})
	endforeach()
	SET(${out_list_name} ${tmp_output} PARENT_SCOPE)
	SET(${out_list_name} ${tmp_output})
	message(STATUS "${out_list_name}: ${${out_list_name}}")
endfunction()

function(add_prefix_inplace prefix_string list_name)
	ADD_PREFIX(${prefix_string} ${list_name} ${list_name})
	SET(${list_name} ${${list_name}} PARENT_SCOPE)
endfunction()

# Add files for compilation

SET(NETWORK_INCLUDE_DIRECTORIES)
list(APPEND NETWORK_INCLUDE_DIRECTORIES Separate_headers/ Separate_headers/hyperparameters/)


# Basic Files
SET(FUNCTIONALITY_FILES
  cuda_functionality.cu
  functionality.cpp
 )
SET(FUNCTIONALITY_FOLDER Functionality)
ADD_PREFIX_INPLACE(${FUNCTIONALITY_FOLDER}/ FUNCTIONALITY_FILES)
list(APPEND NETWORK_INCLUDE_DIRECTORIES ${FUNCTIONALITY_FOLDER})

SET(CONNECTION_FILES
  linear_functions.cu
  connection_gradients.cu

  IConnections.cu
  DenseConnections.cu
  NeatConnections.cu
 )
SET(CONNECTIONS_FOLDER Connections)
ADD_PREFIX_INPLACE(${CONNECTIONS_FOLDER}/ CONNECTION_FILES)
list(APPEND NETWORK_INCLUDE_DIRECTORIES ${CONNECTIONS_FOLDER})

SET(OPTIMIZER_FILES
   #Optimizer_init.cu
   #IOptimizer.cu
   #Adam.cu

  Optimizers.cu
 )
SET(OPTIMIZER_FOLDER Optimizers)
ADD_PREFIX_INPLACE(${OPTIMIZER_FOLDER}/ OPTIMIZER_FILES)

SET(LAYER_FILES
  derivatives.cu
  gradients.cu
  ILayer.cu
  LSTMLayer.cu
  NeuronLayer.cu
  neuron_operations.cu

  ${OPTIMIZER_FILES}
 )
SET(LAYER_FOLDER Layers)
ADD_PREFIX_INPLACE(${LAYER_FOLDER}/ LAYER_FILES)
list(APPEND NETWORK_INCLUDE_DIRECTORIES ${LAYER_FOLDER})
list(APPEND NETWORK_INCLUDE_DIRECTORIES ${LAYER_FOLDER}/${OPTIMIZER_FOLDER})

SET(COST_FUNCTIONS_FILES
  costs.cu
  GAE.cu
  PPO.cu
  regularization.cu
 )
SET(COST_FUNCTIONS_FOLDER Costs)
ADD_PREFIX_INPLACE(${COST_FUNCTIONS_FOLDER}/ COST_FUNCTIONS_FILES)
list(APPEND NETWORK_INCLUDE_DIRECTORIES ${COST_FUNCTIONS_FOLDER})

SET(NON_FOLDER_FILES
  NN.cu
  NN_constructor.cu
 )

SET(NECESSARY_COMPILATION_FILES
  ${FUNCTIONALITY_FILES}
  ${CONNECTION_FILES}
  ${LAYER_FILES}
  ${COST_FUNCTIONS_FILES}
  ${NON_FOLDER_FILES}
 )

# Optional files

SET(SOCKET_FILES

  NN_socket_interpreter.cu

  unix_sock_interop.cu

  SinglyLinkedListNode.h
  HashTable.h
 )
SET(SOCKET_FOLDER socket)
list(APPEND NETWORK_INCLUDE_DIRECTORIES ${SOCKET_FOLDER})

SET(GCC_COVERAGE_COMPILE_FLAGS "-lm -lstdc++ -fanalyzer")
SET(GCC_COVERAGE_LINK_FLAGS    "-lm -lstdc++")

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")

set (CMAKE_CXX_STANDARD 20)

set(CMAKE_CUDA_COMPILER "nvcc")
#set(CMAKE_CXX_COMPILER "gcc")

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lcurand")

# -rdc=true
if (UNIX)
	#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin='g++-13'")
endif()
if (DEBUG)
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -O0")
	SET(GCC_COVERAGE_COMPILE_FLAGS "${GCC_COVERAGE_COMPILE_FLAGS} -O0")
	add_compile_definitions("DEBUG")
	message("-- Config debug")
else()
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")
	SET(GCC_COVERAGE_COMPILE_FLAGS "${GCC_COVERAGE_COMPILE_FLAGS} -O2")
	message("-- Added optimization")
endif()

set(CMAKE_CUDA_ARCHITECTURES "native")

project(NN LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS}")
if(DETERMINISTIC)
	add_compile_definitions(DETERMINISTIC)
	message("-- Deterministic value generation active")
endif()
add_compile_definitions(INCLUDE_BACKEND)
if(target STREQUAL "static")
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fPIC")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fPIC")
	message("-- Static Lib added")
	add_library(${PROJECT_NAME} STATIC ${NECESSARY_COMPILATION_FILES})
	add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
				   COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --cyan
	 "Built Static Lib")
elseif(target STREQUAL "socket")
	message("-- Socket config added")
	if (DEBUG)
		add_compile_definitions(log_positive)
	endif()

	add_executable(${PROJECT_NAME} ${NECESSARY_COMPILATION_FILES} ${SOCKET_FILES})
	if(WIN32)
		target_link_libraries(${PROJECT_NAME} PUBLIC wsock32 ws2_32)
	endif()

	add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --cyan
	"Built socket config")
else()
	message("-- Exe for testing added")
	add_executable(${PROJECT_NAME} kernel.cu ${NECESSARY_COMPILATION_FILES})
endif()

#target_link_libraries(${PROJECT_NAME} PUBLIC -lcurand)
target_link_libraries(${PROJECT_NAME} PUBLIC curand)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "native")
target_include_directories(${PROJECT_NAME} PRIVATE
 ${NETWORK_INCLUDE_DIRECTORIES} ./)
if (DEBUG)
	add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
				   COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --cyan
				   "Built debug")
endif()
