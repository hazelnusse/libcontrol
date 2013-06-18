add_library(control care.cc
                    dare.cc
                    continuous_to_discrete.cc
                    control_internal_ctrb.cc
                    controllability.cc
                    observability.cc)
target_link_libraries(control reflapacke)

if (LIBCONTROL_BUILD_TESTS)
  add_subdirectory(tests)
endif()

