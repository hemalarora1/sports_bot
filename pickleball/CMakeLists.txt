set(PICKLEBALL_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}")
add_definitions(-DPICKLEBALL_FOLDER="${PICKLEBALL_FOLDER}")

# Targets: simviz_pickleball + controller_pickleball, output to
# cs225a/bin/pickleball/. Naming chosen to make it obvious this is our
# pickleball implementation, not one of the project_starter examples — and so
# they don't collide with cs225a/project_starter/mmp_panda's target names if
# both subtrees are added by the same parent CMakeLists.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CS225A_BINARY_DIR}/pickleball)
add_executable(simviz_pickleball     simviz.cpp     ${CS225A_COMMON_SOURCE})
add_executable(controller_pickleball controller.cpp ${CS225A_COMMON_SOURCE})

target_link_libraries(simviz_pickleball     ${CS225A_COMMON_LIBRARIES})
target_link_libraries(controller_pickleball ${CS225A_COMMON_LIBRARIES})
