# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/src/ros2_control/hardware_interface

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/build/hardware_interface

# Include any dependencies generated for this target.
include CMakeFiles/test_macros.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/test_macros.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/test_macros.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_macros.dir/flags.make

CMakeFiles/test_macros.dir/test/test_macros.cpp.o: CMakeFiles/test_macros.dir/flags.make
CMakeFiles/test_macros.dir/test/test_macros.cpp.o: /home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/src/ros2_control/hardware_interface/test/test_macros.cpp
CMakeFiles/test_macros.dir/test/test_macros.cpp.o: CMakeFiles/test_macros.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/build/hardware_interface/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_macros.dir/test/test_macros.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/test_macros.dir/test/test_macros.cpp.o -MF CMakeFiles/test_macros.dir/test/test_macros.cpp.o.d -o CMakeFiles/test_macros.dir/test/test_macros.cpp.o -c /home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/src/ros2_control/hardware_interface/test/test_macros.cpp

CMakeFiles/test_macros.dir/test/test_macros.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_macros.dir/test/test_macros.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/src/ros2_control/hardware_interface/test/test_macros.cpp > CMakeFiles/test_macros.dir/test/test_macros.cpp.i

CMakeFiles/test_macros.dir/test/test_macros.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_macros.dir/test/test_macros.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/src/ros2_control/hardware_interface/test/test_macros.cpp -o CMakeFiles/test_macros.dir/test/test_macros.cpp.s

# Object files for target test_macros
test_macros_OBJECTS = \
"CMakeFiles/test_macros.dir/test/test_macros.cpp.o"

# External object files for target test_macros
test_macros_EXTERNAL_OBJECTS =

test_macros: CMakeFiles/test_macros.dir/test/test_macros.cpp.o
test_macros: CMakeFiles/test_macros.dir/build.make
test_macros: gmock/libgmock_main.a
test_macros: gmock/libgmock.a
test_macros: /opt/ros/humble/lib/librcpputils.so
test_macros: /opt/ros/humble/lib/librcutils.so
test_macros: CMakeFiles/test_macros.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/build/hardware_interface/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_macros"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_macros.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_macros.dir/build: test_macros
.PHONY : CMakeFiles/test_macros.dir/build

CMakeFiles/test_macros.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_macros.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_macros.dir/clean

CMakeFiles/test_macros.dir/depend:
	cd /home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/build/hardware_interface && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/src/ros2_control/hardware_interface /home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/src/ros2_control/hardware_interface /home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/build/hardware_interface /home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/build/hardware_interface /home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/build/hardware_interface/CMakeFiles/test_macros.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_macros.dir/depend

