# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/luca/Documents/CLDL-CUDA

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/luca/Documents/CLDL-CUDA/build/test_files

# Include any dependencies generated for this target.
include test_files/CMakeFiles/testing_input_prop.dir/depend.make

# Include the progress variables for this target.
include test_files/CMakeFiles/testing_input_prop.dir/progress.make

# Include the compile flags for this target's objects.
include test_files/CMakeFiles/testing_input_prop.dir/flags.make

test_files/CMakeFiles/testing_input_prop.dir/Testing_the_each_element_of_input_prop.cu.o: test_files/CMakeFiles/testing_input_prop.dir/flags.make
test_files/CMakeFiles/testing_input_prop.dir/Testing_the_each_element_of_input_prop.cu.o: ../../test_files/Testing_the_each_element_of_input_prop.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/luca/Documents/CLDL-CUDA/build/test_files/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object test_files/CMakeFiles/testing_input_prop.dir/Testing_the_each_element_of_input_prop.cu.o"
	cd /home/luca/Documents/CLDL-CUDA/build/test_files/test_files && /usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/luca/Documents/CLDL-CUDA/test_files/Testing_the_each_element_of_input_prop.cu -o CMakeFiles/testing_input_prop.dir/Testing_the_each_element_of_input_prop.cu.o

test_files/CMakeFiles/testing_input_prop.dir/Testing_the_each_element_of_input_prop.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/testing_input_prop.dir/Testing_the_each_element_of_input_prop.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

test_files/CMakeFiles/testing_input_prop.dir/Testing_the_each_element_of_input_prop.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/testing_input_prop.dir/Testing_the_each_element_of_input_prop.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

test_files/CMakeFiles/testing_input_prop.dir/Testing_the_each_element_of_input_prop.cu.o.requires:

.PHONY : test_files/CMakeFiles/testing_input_prop.dir/Testing_the_each_element_of_input_prop.cu.o.requires

test_files/CMakeFiles/testing_input_prop.dir/Testing_the_each_element_of_input_prop.cu.o.provides: test_files/CMakeFiles/testing_input_prop.dir/Testing_the_each_element_of_input_prop.cu.o.requires
	$(MAKE) -f test_files/CMakeFiles/testing_input_prop.dir/build.make test_files/CMakeFiles/testing_input_prop.dir/Testing_the_each_element_of_input_prop.cu.o.provides.build
.PHONY : test_files/CMakeFiles/testing_input_prop.dir/Testing_the_each_element_of_input_prop.cu.o.provides

test_files/CMakeFiles/testing_input_prop.dir/Testing_the_each_element_of_input_prop.cu.o.provides.build: test_files/CMakeFiles/testing_input_prop.dir/Testing_the_each_element_of_input_prop.cu.o


# Object files for target testing_input_prop
testing_input_prop_OBJECTS = \
"CMakeFiles/testing_input_prop.dir/Testing_the_each_element_of_input_prop.cu.o"

# External object files for target testing_input_prop
testing_input_prop_EXTERNAL_OBJECTS =

test_files/CMakeFiles/testing_input_prop.dir/cmake_device_link.o: test_files/CMakeFiles/testing_input_prop.dir/Testing_the_each_element_of_input_prop.cu.o
test_files/CMakeFiles/testing_input_prop.dir/cmake_device_link.o: test_files/CMakeFiles/testing_input_prop.dir/build.make
test_files/CMakeFiles/testing_input_prop.dir/cmake_device_link.o: libCLDL.a
test_files/CMakeFiles/testing_input_prop.dir/cmake_device_link.o: test_files/CMakeFiles/testing_input_prop.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/luca/Documents/CLDL-CUDA/build/test_files/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/testing_input_prop.dir/cmake_device_link.o"
	cd /home/luca/Documents/CLDL-CUDA/build/test_files/test_files && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testing_input_prop.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test_files/CMakeFiles/testing_input_prop.dir/build: test_files/CMakeFiles/testing_input_prop.dir/cmake_device_link.o

.PHONY : test_files/CMakeFiles/testing_input_prop.dir/build

# Object files for target testing_input_prop
testing_input_prop_OBJECTS = \
"CMakeFiles/testing_input_prop.dir/Testing_the_each_element_of_input_prop.cu.o"

# External object files for target testing_input_prop
testing_input_prop_EXTERNAL_OBJECTS =

test_files/testing_input_prop: test_files/CMakeFiles/testing_input_prop.dir/Testing_the_each_element_of_input_prop.cu.o
test_files/testing_input_prop: test_files/CMakeFiles/testing_input_prop.dir/build.make
test_files/testing_input_prop: libCLDL.a
test_files/testing_input_prop: test_files/CMakeFiles/testing_input_prop.dir/cmake_device_link.o
test_files/testing_input_prop: test_files/CMakeFiles/testing_input_prop.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/luca/Documents/CLDL-CUDA/build/test_files/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable testing_input_prop"
	cd /home/luca/Documents/CLDL-CUDA/build/test_files/test_files && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testing_input_prop.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test_files/CMakeFiles/testing_input_prop.dir/build: test_files/testing_input_prop

.PHONY : test_files/CMakeFiles/testing_input_prop.dir/build

test_files/CMakeFiles/testing_input_prop.dir/requires: test_files/CMakeFiles/testing_input_prop.dir/Testing_the_each_element_of_input_prop.cu.o.requires

.PHONY : test_files/CMakeFiles/testing_input_prop.dir/requires

test_files/CMakeFiles/testing_input_prop.dir/clean:
	cd /home/luca/Documents/CLDL-CUDA/build/test_files/test_files && $(CMAKE_COMMAND) -P CMakeFiles/testing_input_prop.dir/cmake_clean.cmake
.PHONY : test_files/CMakeFiles/testing_input_prop.dir/clean

test_files/CMakeFiles/testing_input_prop.dir/depend:
	cd /home/luca/Documents/CLDL-CUDA/build/test_files && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/luca/Documents/CLDL-CUDA /home/luca/Documents/CLDL-CUDA/test_files /home/luca/Documents/CLDL-CUDA/build/test_files /home/luca/Documents/CLDL-CUDA/build/test_files/test_files /home/luca/Documents/CLDL-CUDA/build/test_files/test_files/CMakeFiles/testing_input_prop.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test_files/CMakeFiles/testing_input_prop.dir/depend

