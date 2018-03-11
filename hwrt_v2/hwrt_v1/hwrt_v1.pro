TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    ecs.cpp \
    io.cpp \
    predict.cpp \
    _matrix.cpp \
    allocate.cpp

HEADERS += \
    lib_time.h \
    lib_io.h \
    predict.h \
    _matrix.h \
    allocate.h
