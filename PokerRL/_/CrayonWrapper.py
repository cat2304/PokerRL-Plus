# Copyright (c) 2019 Eric Steinberger

from os.path import join as ospj

from pycrayon import CrayonClient

from PokerRL.rl.MaybeRay import MaybeRay
from PokerRL.util.file_util import create_dir_if_not_exist, write_dict_to_file_json


class CrayonWrapper:
    """
    PyCrayon的包装器类，用于与TensorBoard进行交互。
    
    PyCrayon (https://github.com/torrvision/crayon) 是一个语言无关的TensorBoard接口。
    这个包装器提供了以下功能：
    1. 实验日志的记录和管理
    2. 分布式环境下的日志同步
    3. 日志数据的导出和存储
    
    主要用途：
    - 记录强化学习训练过程中的各种指标
    - 支持分布式训练环境下的日志收集
    - 提供TensorBoard可视化的数据接口
    """

    def __init__(self, name, runs_distributed, runs_cluster, chief_handle, path_log_storage=None,
                 crayon_server_address="localhost"):
        """
        初始化Crayon包装器
        
        参数:
            name (str): 实验名称
            runs_distributed (bool): 是否在分布式环境中运行
            runs_cluster (bool): 是否在集群环境中运行
            chief_handle: 主节点的句柄
            path_log_storage (str, optional): 日志存储路径
            crayon_server_address (str): Crayon服务器地址
        """
        self._name = name
        self._path_log_storage = path_log_storage
        if path_log_storage is not None:
            create_dir_if_not_exist(path_log_storage)

        self._chief_handle = chief_handle
        # 连接到Crayon服务器
        self._crayon = CrayonClient(hostname=crayon_server_address)
        self._experiments = {}
        self.clear()
        # 自定义日志存储
        self._custom_logs = {}  # dict of exps containing dict of graph names containing lists of {step: val, } dicts

        # 初始化Ray支持
        self._ray = MaybeRay(runs_distributed=runs_distributed, runs_cluster=runs_cluster)

    @property
    def name(self):
        """获取实验名称"""
        return self._name

    @property
    def path_log_storage(self):
        """获取日志存储路径"""
        return self._path_log_storage

    def clear(self):
        """
        清除当前的实验记录。
        注意：不会清除Crayon内部的实验日志和文件。
        """
        self._experiments = {}

    def export_all(self, iter_nr):
        """
        导出当前运行的所有日志，包括TensorBoard格式和JSON格式。
        
        参数:
            iter_nr (int): 迭代次数
        """
        if self._path_log_storage is not None:
            # 创建导出目录
            path_crayon = ospj(self._path_log_storage, str(self._name), str(iter_nr), "crayon")
            path_json = ospj(self._path_log_storage, str(self._name), str(iter_nr), "as_json")
            create_dir_if_not_exist(path=path_crayon)
            create_dir_if_not_exist(path=path_json)
            # 导出每个实验的数据
            for e in self._experiments.values():
                e.to_zip(filename=ospj(path_crayon, e.xp_name + ".zip"))
                write_dict_to_file_json(dictionary=self._custom_logs, _dir=path_json, file_name="logs")

    def update_from_log_buffer(self):
        """
        从主节点获取新添加的日志，并将其添加到TensorBoard中。
        这个函数在分布式环境中特别有用，用于同步各个工作节点的日志。
        """
        # 获取新的日志值
        new_v, exp_names = self._get_new_vals()

        # 为新的实验创建记录
        for e in exp_names:
            if e not in self._experiments.keys():
                self._custom_logs[e] = {}
                try:
                    self._experiments[e] = self._crayon.create_experiment(xp_name=e)
                except ValueError:
                    self._crayon.remove_experiment(xp_name=e)
                    self._experiments[e] = self._crayon.create_experiment(xp_name=e)

        # 添加新的数据点
        for name, vals_dict in new_v.items():
            for graph_name, data_points in vals_dict.items():
                for data_point in data_points:
                    step = int(data_point[0])
                    val = data_point[1]

                    # 添加到TensorBoard
                    self._experiments[name].add_scalar_value(name=graph_name, step=step, value=val)
                    if graph_name not in self._custom_logs[name].keys():
                        self._custom_logs[name][graph_name] = []

                    self._custom_logs[name][graph_name].append({step: val})

    def _get_new_vals(self):
        """
        从主节点获取新添加的日志值。
        
        返回:
            dict: 包含新添加的日志数据的字典
        """
        return self._ray.get(self._ray.remote(self._chief_handle.get_new_values))
