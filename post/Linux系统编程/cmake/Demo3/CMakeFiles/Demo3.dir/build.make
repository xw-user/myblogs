# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

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
CMAKE_SOURCE_DIR = /root/Note/Linux/cmake/Demo3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/Note/Linux/cmake/Demo3

# Include any dependencies generated for this target.
include CMakeFiles/Demo3.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Demo3.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Demo3.dir/flags.make

CMakeFiles/Demo3.dir/test/main.c.o: CMakeFiles/Demo3.dir/flags.make
CMakeFiles/Demo3.dir/test/main.c.o: test/main.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/Note/Linux/cmake/Demo3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/Demo3.dir/test/main.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/Demo3.dir/test/main.c.o   -c /root/Note/Linux/cmake/Demo3/test/main.c

CMakeFiles/Demo3.dir/test/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/Demo3.dir/test/main.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /root/Note/Linux/cmake/Demo3/test/main.c > CMakeFiles/Demo3.dir/test/main.c.i

CMakeFiles/Demo3.dir/test/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/Demo3.dir/test/main.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /root/Note/Linux/cmake/Demo3/test/main.c -o CMakeFiles/Demo3.dir/test/main.c.s

# Object files for target Demo3
Demo3_OBJECTS = \
"CMakeFiles/Demo3.dir/test/main.c.o"

# External object files for target Demo3
Demo3_EXTERNAL_OBJECTS =

Demo3: CMakeFiles/Demo3.dir/test/main.c.o
Demo3: CMakeFiles/Demo3.dir/build.make
Demo3: src/libpow.a
Demo3: CMakeFiles/Demo3.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/Note/Linux/cmake/Demo3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable Demo3"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Demo3.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Demo3.dir/build: Demo3

.PHONY : CMakeFiles/Demo3.dir/build

CMakeFiles/Demo3.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Demo3.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Demo3.dir/clean

CMakeFiles/Demo3.dir/depend:
	cd /root/Note/Linux/cmake/Demo3 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/Note/Linux/cmake/Demo3 /root/Note/Linux/cmake/Demo3 /root/Note/Linux/cmake/Demo3 /root/Note/Linux/cmake/Demo3 /root/Note/Linux/cmake/Demo3/CMakeFiles/Demo3.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Demo3.dir/depend

