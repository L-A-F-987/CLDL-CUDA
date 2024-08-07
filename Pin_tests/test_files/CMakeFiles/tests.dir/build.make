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
CMAKE_BINARY_DIR = /home/luca/Documents/CLDL-CUDA/Pin_tests

# Include any dependencies generated for this target.
include test_files/CMakeFiles/tests.dir/depend.make

# Include the progress variables for this target.
include test_files/CMakeFiles/tests.dir/progress.make

# Include the compile flags for this target's objects.
include test_files/CMakeFiles/tests.dir/flags.make

test_files/CMakeFiles/tests.dir/t.cu.o: test_files/CMakeFiles/tests.dir/flags.make
test_files/CMakeFiles/tests.dir/t.cu.o: ../test_files/t.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/luca/Documents/CLDL-CUDA/Pin_tests/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object test_files/CMakeFiles/tests.dir/t.cu.o"
	cd /home/luca/Documents/CLDL-CUDA/Pin_tests/test_files && /usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/luca/Documents/CLDL-CUDA/test_files/t.cu -o CMakeFiles/tests.dir/t.cu.o

test_files/CMakeFiles/tests.dir/t.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/tests.dir/t.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

test_files/CMakeFiles/tests.dir/t.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/tests.dir/t.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

test_files/CMakeFiles/tests.dir/t.cu.o.requires:

.PHONY : test_files/CMakeFiles/tests.dir/t.cu.o.requires

test_files/CMakeFiles/tests.dir/t.cu.o.provides: test_files/CMakeFiles/tests.dir/t.cu.o.requires
	$(MAKE) -f test_files/CMakeFiles/tests.dir/build.make test_files/CMakeFiles/tests.dir/t.cu.o.provides.build
.PHONY : test_files/CMakeFiles/tests.dir/t.cu.o.provides

test_files/CMakeFiles/tests.dir/t.cu.o.provides.build: test_files/CMakeFiles/tests.dir/t.cu.o


# Object files for target tests
tests_OBJECTS = \
"CMakeFiles/tests.dir/t.cu.o"

# External object files for target tests
tests_EXTERNAL_OBJECTS =

test_files/CMakeFiles/tests.dir/cmake_device_link.o: test_files/CMakeFiles/tests.dir/t.cu.o
test_files/CMakeFiles/tests.dir/cmake_device_link.o: test_files/CMakeFiles/tests.dir/build.make
test_files/CMakeFiles/tests.dir/cmake_device_link.o: libCLDL.a
test_files/CMakeFiles/tests.dir/cmake_device_link.o: test_files/CMakeFiles/tests.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/luca/Documents/CLDL-CUDA/Pin_tests/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/tests.dir/cmake_device_link.o"
	cd /home/luca/Documents/CLDL-CUDA/Pin_tests/test_files && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tests.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test_files/CMakeFiles/tests.dir/build: test_files/CMakeFiles/tests.dir/cmake_device_link.o

.PHONY : test_files/CMakeFiles/tests.dir/build

# Object files for target tests
tests_OBJECTS = \
"CMakeFiles/tests.dir/t.cu.o"

# External object files for target tests
tests_EXTERNAL_OBJECTS =

test_files/tests: test_files/CMakeFiles/tests.dir/t.cu.o
test_files/tests: test_files/CMakeFiles/tests.dir/build.make
test_files/tests: libCLDL.a
test_files/tests: test_files/CMakeFiles/tests.dir/cmake_device_link.o
test_files/tests: test_files/CMakeFiles/tests.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/luca/Documents/CLDL-CUDA/Pin_tests/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable tests"
	cd /home/luca/Documents/CLDL-CUDA/Pin_tests/test_files && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tests.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test_files/CMakeFiles/tests.dir/build: test_files/tests

.PHONY : test_files/CMakeFiles/tests.dir/build

test_files/CMakeFiles/tests.dir/requires: test_files/CMakeFiles/tests.dir/t.cu.o.requires

.PHONY : test_files/CMakeFiles/tests.dir/requires

test_files/CMakeFiles/tests.dir/clean:
	cd /home/luca/Documents/CLDL-CUDA/Pin_tests/test_files && $(CMAKE_COMMAND) -P CMakeFiles/tests.dir/cmake_clean.cmake
.PHONY : test_files/CMakeFiles/tests.dir/clean

test_files/CMakeFiles/tests.dir/depend:
	cd /home/luca/Documents/CLDL-CUDA/Pin_tests && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/luca/Documents/CLDL-CUDA /home/luca/Documents/CLDL-CUDA/test_files /home/luca/Documents/CLDL-CUDA/Pin_tests /home/luca/Documents/CLDL-CUDA/Pin_tests/test_files /home/luca/Documents/CLDL-CUDA/Pin_tests/test_files/CMakeFiles/tests.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test_files/CMakeFiles/tests.dir/depend

