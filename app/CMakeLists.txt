add_executable(MamutExample main.cpp)

target_link_directories(MamutExample PUBLIC Mamut)
target_include_directories(MamutExample PUBLIC ${PROJECT_SOURCE_DIR})