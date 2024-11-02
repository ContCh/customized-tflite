@echo off

@REM Build parameter
set BUILD_DIRNAME=build
set COMPILE_MODE=Release
set WITH_TEST=OFF

if not exist %BUILD_DIRNAME% mkdir %BUILD_DIRNAME%
set CMAKE_COMMAND=cmake ^
                  -G "MinGW Makefiles" ^
                  -DCMAKE_BUILD_TYPE=%COMPILE_MODE% ^
                  -DENABLE_UNIT_TEST=%WITH_TEST% ^
                  -B%BUILD_DIRNAME%

set BUILD_COMMAND=cmake --build %BUILD_DIRNAME% ^
                  -j 40 ^
                  --clean-first

set INSTALL_COMMAND=cmake --install %BUILD_DIRNAME%

%CMAKE_COMMAND%
if %errorlevel%==0 %BUILD_COMMAND%
if %errorlevel%==0 %INSTALL_COMMAND%
