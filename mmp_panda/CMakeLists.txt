set(MMP_PANDA_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}")
add_definitions(-DMMP_PANDA_FOLDER="${MMP_PANDA_FOLDER}")

# Targets are renamed (vs cs225a/project_starter/mmp_panda) so both trees can
# be built side-by-side without target-name collisions.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CS225A_BINARY_DIR}/sports_bot_mmp_panda)
add_executable(controller_sports_bot_mmp_panda controller.cpp ${CS225A_COMMON_SOURCE})
add_executable(simviz_sports_bot_mmp_panda simviz.cpp ${CS225A_COMMON_SOURCE})

target_link_libraries(controller_sports_bot_mmp_panda ${CS225A_COMMON_LIBRARIES})
target_link_libraries(simviz_sports_bot_mmp_panda ${CS225A_COMMON_LIBRARIES})
