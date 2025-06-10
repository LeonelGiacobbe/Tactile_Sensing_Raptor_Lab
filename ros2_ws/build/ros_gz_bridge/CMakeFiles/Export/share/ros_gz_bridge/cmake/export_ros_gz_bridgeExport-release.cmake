#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ros_gz_bridge::ros_gz_bridge" for configuration "Release"
set_property(TARGET ros_gz_bridge::ros_gz_bridge APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ros_gz_bridge::ros_gz_bridge PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libros_gz_bridge.so"
  IMPORTED_SONAME_RELEASE "libros_gz_bridge.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS ros_gz_bridge::ros_gz_bridge )
list(APPEND _IMPORT_CHECK_FILES_FOR_ros_gz_bridge::ros_gz_bridge "${_IMPORT_PREFIX}/lib/libros_gz_bridge.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
