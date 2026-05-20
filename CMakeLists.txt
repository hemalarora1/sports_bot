set(SPORTS_BOT_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}")
add_definitions(-DSPORTS_BOT_FOLDER="${SPORTS_BOT_FOLDER}")

# The pickleball/ subdir holds our actual implementation (sim, controller,
# world URDF, run.sh, launcher). The panda/ and my_project/ folders at this
# level are unmodified copies of cs225a/project_starter/{panda,my_project}
# kept here for reference only — they're not built (and would collide with
# the project_starter targets if added).
add_subdirectory(pickleball)
