set(SPORTS_BOT_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}")
add_definitions(-DSPORTS_BOT_FOLDER="${SPORTS_BOT_FOLDER}")

# Only mmp_panda is wired into the cs225a build right now; the panda and
# my_project subdirs share executable target names with
# cs225a/project_starter/{panda,my_project} and would clash if both trees are
# included by the same parent CMakeLists. Re-enable the others by renaming
# their executable targets first.
add_subdirectory(mmp_panda)
# add_subdirectory(panda)
# add_subdirectory(my_project)
