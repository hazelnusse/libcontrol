add_executable(test_care test_care.cc)
target_link_libraries(test_care gtest_main control)
add_test(NAME test_care
         COMMAND test_care)

add_executable(test_continuous_to_discrete test_continuous_to_discrete.cc)
target_link_libraries(test_continuous_to_discrete gtest_main control)
add_test(NAME test_continuous_to_discrete
         COMMAND test_continuous_to_discrete)

add_executable(test_controllability test_controllability.cc)
target_link_libraries(test_controllability gtest_main control)
add_test(NAME test_controllability
         COMMAND test_controllability)

add_executable(test_dare test_dare.cc)
target_link_libraries(test_dare gtest_main control)
add_test(NAME test_dare
         COMMAND test_dare)

add_executable(test_observability test_observability.cc)
target_link_libraries(test_observability gtest_main control)
add_test(NAME test_observability
         COMMAND test_observability)
