CMAKE_MINIMUM_REQUIRED(VERSION 3.15)

get_filename_component(APP_FOLDER_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)

set(THIS_APPLICATION ${APP_FOLDER_NAME})

PROJECT("${APP_FOLDER_NAME_PARENT}_${THIS_APPLICATION}")


SET(MAIN_FILE "${THIS_APPLICATION}") # the name of the main file with no extension
SET(EXEC_FILE "${APP_FOLDER_NAME_PARENT}_${MAIN_FILE}") # the name of the executable file

femusMacroBuildApplication(${MAIN_FILE} ${EXEC_FILE})

# ---------------------------------------------------------------------------
# femusMacroBuildApplication builds exactly ONE source file (ex20.cpp), so the
# remaining translation units are appended with target_sources() rather than by
# replacing the macro with a hand-written ADD_EXECUTABLE: that keeps every
# library, include directory, output-directory property and file-copy rule the
# macro supplies, by construction instead of by duplication.
#
#   Config, Petsc  ->  DirichletControl  ->  Discretization  ->  SpaceTimeOperators
#     ->  Preconditioners  ->  {BoundaryReducedSchur, DirichletReducedSchur}
#     ->  Solver  ->  Output
#     ->  {Verification, Benchmarks, CommandLine}  (leaves)
# ---------------------------------------------------------------------------

target_include_directories(${EXEC_FILE} PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_sources(${EXEC_FILE} PRIVATE
  ${PROJECT_SOURCE_DIR}/include/DirichletControl.cpp
  ${PROJECT_SOURCE_DIR}/include/Discretization.cpp
  ${PROJECT_SOURCE_DIR}/include/SpaceTimeOperators.cpp
  ${PROJECT_SOURCE_DIR}/include/Preconditioners.cpp
  ${PROJECT_SOURCE_DIR}/include/BoundaryReducedSchur.cpp
  ${PROJECT_SOURCE_DIR}/include/DirichletReducedSchur.cpp
  ${PROJECT_SOURCE_DIR}/include/Solver.cpp
  ${PROJECT_SOURCE_DIR}/include/Output.cpp
  ${PROJECT_SOURCE_DIR}/include/Verification.cpp
  ${PROJECT_SOURCE_DIR}/include/Benchmarks.cpp
  ${PROJECT_SOURCE_DIR}/include/CommandLine.cpp
)
