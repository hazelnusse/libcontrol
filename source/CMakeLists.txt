add_library(control care.cc
                    dare.cc
                    continuous_to_discrete.cc
                    control_internal_ctrb.cc
                    controllability.cc
                    observability.cc)

# see is library reflapacke is installed
# if not, link against lapacke
set(USE_REFLAPACKE_LIB "" CACHE INTERNAL "")
find_library(USE_REFLAPACKE_LIB reflapacke)
if (USE_REFLAPACKE_LIB)
    target_link_libraries(control reflapacke)
else()
    target_link_libraries(control lapacke)
endif()

if (LIBCONTROL_BUILD_TESTS)
  add_subdirectory(tests)
endif()

install(TARGETS control
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

