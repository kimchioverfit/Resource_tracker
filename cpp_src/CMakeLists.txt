cmake_minimum_required(VERSION 3.16)

# 프로젝트 이름과 언어
project(PC365Logger
    VERSION 1.0
    LANGUAGES CXX)

# C++ 표준 설정 (17 이상이면 충분)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 소스 파일들 목록
# main.cpp 이름이 다르면 여기만 바꿔줘 (예: SystemMonitorConsole.cpp)
set(SOURCES
    main.cpp
    util_format.cpp
    csv_writer.cpp
    wmi_helper.cpp
    disk_helper.cpp
    net_helper.cpp
    nvml_helper.cpp
    pdh_helper.cpp
)

# 실행 파일 생성
add_executable(${PROJECT_NAME} ${SOURCES})

# Windows 전용 설정
if (MSVC)
    # 경고 레벨 & 멀티바이트/유니코드 관련은 필요시 추가
    target_compile_options(${PROJECT_NAME} PRIVATE /W4)
endif()

# Windows 헤더에서 min/max 매크로 안 쓰게 (std::max/std::min 깨지는 것 방지)
target_compile_definitions(${PROJECT_NAME} PRIVATE NOMINMAX)

# 필요 라이브러리 링크
# - wbemuuid : WMI
# - iphlpapi : 네트워크 어댑터 정보
# - pdh      : Performance Data Helper
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        wbemuuid
        iphlpapi
        pdh
)

# 인클루드 디렉토리 (현재 폴더)
target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
