#-------------------------------------------------
#
# Project created by QtCreator 2016-03-15T19:57:14
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = tipl_cnn
TEMPLATE = app

INCLUDEPATH += ../include

SOURCES += main.cpp \
    load_mnist.cpp \
    load_cifar.cpp \
    prog_interface.cpp \
    ../include/tipl/ml/svm.cpp

SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h \
    ../include/tipl/filter/anisotropic_diffusion.hpp \
    ../include/tipl/filter/canny_edge.hpp \
    ../include/tipl/filter/filter_model.hpp \
    ../include/tipl/filter/gaussian.hpp \
    ../include/tipl/filter/gradient_magnitude.hpp \
    ../include/tipl/filter/laplacian.hpp \
    ../include/tipl/filter/mean.hpp \
    ../include/tipl/filter/sobel.hpp \
    ../include/tipl/io/2dseq.hpp \
    ../include/tipl/io/avi.hpp \
    ../include/tipl/io/bitmap.hpp \
    ../include/tipl/io/dicom.hpp \
    ../include/tipl/io/interface.hpp \
    ../include/tipl/io/io.hpp \
    ../include/tipl/io/mat.hpp \
    ../include/tipl/io/nifti.hpp \
    ../include/tipl/io/tiff.hpp \
    ../include/tipl/io/tiff_tag.hpp \
    ../include/tipl/ml/ada_boost.hpp \
    ../include/tipl/ml/cnn.hpp \
    ../include/tipl/ml/decision_tree.hpp \
    ../include/tipl/ml/em.hpp \
    ../include/tipl/ml/hmc.hpp \
    ../include/tipl/ml/k_means.hpp \
    ../include/tipl/ml/lg.hpp \
    ../include/tipl/ml/nb.hpp \
    ../include/tipl/ml/non_parametric.hpp \
    ../include/tipl/ml/utility.hpp \
    ../include/tipl/morphology/morphology.hpp \
    ../include/tipl/numerical/basic_op.hpp \
    ../include/tipl/numerical/dif.hpp \
    ../include/tipl/numerical/fft.hpp \
    ../include/tipl/numerical/index_algorithm.hpp \
    ../include/tipl/numerical/interpolation.hpp \
    ../include/tipl/numerical/matrix.hpp \
    ../include/tipl/numerical/numerical.hpp \
    ../include/tipl/numerical/optimization.hpp \
    ../include/tipl/numerical/resampling.hpp \
    ../include/tipl/numerical/slice.hpp \
    ../include/tipl/numerical/statistics.hpp \
    ../include/tipl/numerical/transformation.hpp \
    ../include/tipl/numerical/window.hpp \
    ../include/tipl/reg/bfnorm.hpp \
    ../include/tipl/reg/dmdm.hpp \
    ../include/tipl/reg/lddmm.hpp \
    ../include/tipl/reg/linear.hpp \
    ../include/tipl/segmentation/disjoint_set.hpp \
    ../include/tipl/segmentation/fast_marching.hpp \
    ../include/tipl/segmentation/graph_cut.hpp \
    ../include/tipl/segmentation/otsu.hpp \
    ../include/tipl/segmentation/segmentation.hpp \
    ../include/tipl/segmentation/stochastic_competition.hpp \
    ../include/tipl/segmentation/watershed.hpp \
    ../include/tipl/utility/basic_image.hpp \
    ../include/tipl/utility/geometry.hpp \
    ../include/tipl/utility/multi_thread.hpp \
    ../include/tipl/utility/pixel_index.hpp \
    ../include/tipl/utility/pixel_value.hpp \
    ../include/tipl/vis/color_map.hpp \
    ../include/tipl/vis/march_cube.hpp \
    ../include/tipl/reg/reg.hpp \
    gzip_interface.hpp \
    ../include/tipl/reg/cdm.hpp \
    ../include/tipl/tipl.hpp \
    ../include/tipl/ml/svm.hpp

FORMS    += mainwindow.ui
