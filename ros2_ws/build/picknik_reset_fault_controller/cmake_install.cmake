# Install script for directory: /home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/src/picknik_controllers/picknik_reset_fault_controller

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/install/picknik_reset_fault_controller")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/picknik_reset_fault_controller" TYPE FILE FILES "/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/src/picknik_controllers/picknik_reset_fault_controller/controller_plugins.xml")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libpicknik_reset_fault_controller.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libpicknik_reset_fault_controller.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libpicknik_reset_fault_controller.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/build/picknik_reset_fault_controller/libpicknik_reset_fault_controller.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libpicknik_reset_fault_controller.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libpicknik_reset_fault_controller.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libpicknik_reset_fault_controller.so"
         OLD_RPATH "/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/install/realtime_tools/lib:/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/install/hardware_interface/lib:/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/install/control_msgs/lib:/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/install/controller_interface/lib:/opt/ros/humble/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libpicknik_reset_fault_controller.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/src/picknik_controllers/picknik_reset_fault_controller/include/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/picknik_reset_fault_controller" TYPE FILE FILES "/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/src/picknik_controllers/picknik_reset_fault_controller/controller_plugins.xml")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/picknik_reset_fault_controller/environment" TYPE FILE FILES "/opt/ros/humble/lib/python3.10/site-packages/ament_package/template/environment_hook/library_path.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/picknik_reset_fault_controller/environment" TYPE FILE FILES "/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/build/picknik_reset_fault_controller/ament_cmake_environment_hooks/library_path.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/package_run_dependencies" TYPE FILE FILES "/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/build/picknik_reset_fault_controller/ament_cmake_index/share/ament_index/resource_index/package_run_dependencies/picknik_reset_fault_controller")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/parent_prefix_path" TYPE FILE FILES "/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/build/picknik_reset_fault_controller/ament_cmake_index/share/ament_index/resource_index/parent_prefix_path/picknik_reset_fault_controller")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/picknik_reset_fault_controller/environment" TYPE FILE FILES "/opt/ros/humble/share/ament_cmake_core/cmake/environment_hooks/environment/ament_prefix_path.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/picknik_reset_fault_controller/environment" TYPE FILE FILES "/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/build/picknik_reset_fault_controller/ament_cmake_environment_hooks/ament_prefix_path.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/picknik_reset_fault_controller/environment" TYPE FILE FILES "/opt/ros/humble/share/ament_cmake_core/cmake/environment_hooks/environment/path.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/picknik_reset_fault_controller/environment" TYPE FILE FILES "/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/build/picknik_reset_fault_controller/ament_cmake_environment_hooks/path.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/picknik_reset_fault_controller" TYPE FILE FILES "/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/build/picknik_reset_fault_controller/ament_cmake_environment_hooks/local_setup.bash")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/picknik_reset_fault_controller" TYPE FILE FILES "/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/build/picknik_reset_fault_controller/ament_cmake_environment_hooks/local_setup.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/picknik_reset_fault_controller" TYPE FILE FILES "/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/build/picknik_reset_fault_controller/ament_cmake_environment_hooks/local_setup.zsh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/picknik_reset_fault_controller" TYPE FILE FILES "/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/build/picknik_reset_fault_controller/ament_cmake_environment_hooks/local_setup.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/picknik_reset_fault_controller" TYPE FILE FILES "/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/build/picknik_reset_fault_controller/ament_cmake_environment_hooks/package.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/packages" TYPE FILE FILES "/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/build/picknik_reset_fault_controller/ament_cmake_index/share/ament_index/resource_index/packages/picknik_reset_fault_controller")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/controller_interface__pluginlib__plugin" TYPE FILE FILES "/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/build/picknik_reset_fault_controller/ament_cmake_index/share/ament_index/resource_index/controller_interface__pluginlib__plugin/picknik_reset_fault_controller")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/picknik_reset_fault_controller/cmake" TYPE FILE FILES "/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/build/picknik_reset_fault_controller/ament_cmake_export_dependencies/ament_cmake_export_dependencies-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/picknik_reset_fault_controller/cmake" TYPE FILE FILES "/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/build/picknik_reset_fault_controller/ament_cmake_export_include_directories/ament_cmake_export_include_directories-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/picknik_reset_fault_controller/cmake" TYPE FILE FILES "/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/build/picknik_reset_fault_controller/ament_cmake_export_libraries/ament_cmake_export_libraries-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/picknik_reset_fault_controller/cmake" TYPE FILE FILES
    "/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/build/picknik_reset_fault_controller/ament_cmake_core/picknik_reset_fault_controllerConfig.cmake"
    "/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/build/picknik_reset_fault_controller/ament_cmake_core/picknik_reset_fault_controllerConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/picknik_reset_fault_controller" TYPE FILE FILES "/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/src/picknik_controllers/picknik_reset_fault_controller/package.xml")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/leo/Documents/Tactile_Sensing_Raptor_Lab/ros2_ws/build/picknik_reset_fault_controller/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
