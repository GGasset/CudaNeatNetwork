cmake_minimum_required(VERSION 3.28)

# In linux add CPP13 compilation flag. I recommend homebrew to install gcc@13
# In windows compile inside Visual Studio.

# Add files for compilation
SET(COMPILATION_FILES 
		connection_gradients.cu 
		costs.cu
		cuda_functionality.cu
		DenseConnections.cu
		derivatives.cu
		gradients.cu
		IConnections.cu
		ILayer.cu
		linear_functions.cu
		LSTMLayer.cu
		NeatConnections.cu
		NeuronLayer.cu
		neuron_operations.cu
		NN.cu
		NN_constructor.cu
		GAE.cu
		Optimizers.cu

		functionality.cpp
	)
SET(SOCKET_FILES
	
		NN_socket_interpreter.cu

		unix_sock_interop.cu

		SinglyLinkedListNode.h
		HashTable.h
	)

# Add linking to stdlib
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
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin='g++-13'")
endif()
if (DEBUG)
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
	add_compile_definitions("DEBUG")
	message("-- Config debug")
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
	   add_library(${PROJECT_NAME} STATIC ${COMPILATION_FILES})
	add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --cyan
		   "Built Static Lib")
elseif(target STREQUAL "socket")
	message("-- Socket config added")
	if (DEBUG)
		add_compile_definitions(log_positive)
	endif()

	add_executable(${PROJECT_NAME} ${COMPILATION_FILES} ${SOCKET_FILES})
	if(WIN32)
		target_link_libraries(${PROJECT_NAME} PUBLIC wsock32 ws2_32)
	endif()

	add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
	   COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --cyan
	   "Built socket config")
else()
	message("-- Exe for testing added")
	add_executable(${PROJECT_NAME} kernel.cu ${COMPILATION_FILES})
endif()

#target_link_libraries(${PROJECT_NAME} PUBLIC -lcurand)
target_link_libraries(${PROJECT_NAME} PUBLIC curand)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "native")
if (DEBUG)
	add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --cyan
                   "Built debug")
endif()
