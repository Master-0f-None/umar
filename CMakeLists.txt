project(ArrayFire_ZCash)
cmake_minimum_required(VERSION 3.5)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(OpenCL REQUIRED)
find_library(Blake2b b2)
message(STATUS ${Blake2b})

set(sources main.cpp)

message(STATUS ${Boost_LIBRARIES})
message(STATUS ${Boost_program_options_FOUND})
include_directories(${Boost_INCLUDE_DIR})
add_definitions(-std=c++1z)
add_executable(af_cash ${sources})
target_link_libraries(af_cash ${Blake2b} ${Boost_LIBRARIES} ${OpenCL_LIBRARIES})

#set_property(TARGET af_cash PROPERTY CXX_STANDARD 14)
set_property(TARGET af_cash PROPERTY CXX_STANDARD_REQUIRED ON)
