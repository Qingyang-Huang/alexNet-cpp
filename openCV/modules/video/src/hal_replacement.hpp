/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_VIDEO_HAL_REPLACEMENT_HPP
#define OPENCV_VIDEO_HAL_REPLACEMENT_HPP

#include "opencv2/core/hal/interface.h"

#if defined(__clang__)  // clang or MSVC clang
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

inline int hal_ni_optflowpyrLK(const uchar* src1_data, const uchar* src2_data, size_t src1_step, size_t src2_step, int width, int height, int cn, int depth, float* pre_pt, float* next_pt,
                                int num_pt, const uchar* status, float *err,int winx, int winy, int maxl, int flags, double min_threshold, int criteria_type, int max_count, double epsilon) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

//! @cond IGNORED
#define cv_hal_optflowpyrLK hal_ni_optflowpyrLK
//! @endcond

//! @}


#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include "custom_hal.hpp"

//! @cond IGNORED
#define CALL_HAL_RET(name, fun, retval, ...) \
    int res = __CV_EXPAND(fun(__VA_ARGS__, &retval)); \
    if (res == CV_HAL_ERROR_OK) \
        return retval; \
    else if (res != CV_HAL_ERROR_NOT_IMPLEMENTED) \
        CV_Error_(cv::Error::StsInternal, \
            ("HAL implementation " CVAUX_STR(name) " ==> " CVAUX_STR(fun) " returned %d (0x%08x)", res, res));


#define CALL_HAL(name, fun, ...) \
{                                           \
    int res = __CV_EXPAND(fun(__VA_ARGS__)); \
    if (res == CV_HAL_ERROR_OK) \
        return; \
    else if (res != CV_HAL_ERROR_NOT_IMPLEMENTED) \
        CV_Error_(cv::Error::StsInternal, \
            ("HAL implementation " CVAUX_STR(name) " ==> " CVAUX_STR(fun) " returned %d (0x%08x)", res, res)); \
}
//! @endcond

#endif
