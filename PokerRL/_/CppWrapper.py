# Copyright (c) 2019 Eric Steinberger


import ctypes
import os

import numpy as np


class CppWrapper:
    """
    C++库的Python包装器基类，使用ctypes实现。
    
    这个类提供了将Python代码与C++库进行交互的基础功能：
    1. 加载C++动态链接库
    2. 处理NumPy数组与C++数组之间的转换
    3. 提供通用的数据类型定义
    
    主要用途：
    - 包装高性能的C++扑克牌评估库
    - 提供Python友好的接口来调用C++函数
    - 处理Python和C++之间的数据类型转换
    """
    # 定义2D数组参数类型，用于C++函数调用
    ARR_2D_ARG_TYPE = np.ctypeslib.ndpointer(dtype=np.intp, ndim=1, flags='C')
    # 根据操作系统确定动态链接库的文件扩展名
    CPP_LIB_FILE_ENDING = "dll" if os.name == 'nt' else "so"

    def __init__(self, path_to_dll):
        """
        初始化C++包装器
        
        参数:
            path_to_dll (str): C++动态链接库的路径
        """
        # 加载C++动态链接库
        self._clib = ctypes.cdll.LoadLibrary(path_to_dll)

    @staticmethod
    def np_1d_arr_to_c(np_arr):
        """
        将NumPy一维数组转换为C++可用的指针
        
        参数:
            np_arr (np.ndarray): 要转换的NumPy数组
            
        返回:
            ctypes.c_void_p: C++可用的指针
        """
        return ctypes.c_void_p(np_arr.ctypes.data)

    @staticmethod
    def np_2d_arr_to_c(np_2d_arr):
        """
        将NumPy二维数组转换为C++可用的指针数组
        
        参数:
            np_2d_arr (np.ndarray): 要转换的NumPy二维数组
            
        返回:
            np.ndarray: 包含每行数据指针的数组
        """
        return (np_2d_arr.__array_interface__['data'][0]
                + np.arange(np_2d_arr.shape[0]) * np_2d_arr.strides[0]).astype(np.intp)
