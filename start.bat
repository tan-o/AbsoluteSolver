@echo off
:: 只在当前窗口临时增加 OpenCV 和 LLVM 路径
set PATH=<your opencv path>\x64\vc16\bin;<your llvm path>\bin;%PATH%
:: 强制指定 OpenCV 配置路径解决之前的 CMake 报错
set OpenCV_DIR=<your opencv path>\x64\vc16\lib
set OPENCV_LINK_LIBS=opencv_world4120

cargo run --release
