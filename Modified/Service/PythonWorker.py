import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Callable
import logging


class PythonWorkerBase(ABC):
    """Python Worker基类 - 使用HTTP服务器与C#通信"""

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        CORS(self.app)  # 允许跨域请求

        self.functions: Dict[str, Callable] = {}
        self.globals: Dict[str, Any] = {}  # 存储全局变量
        self.server_thread = None
        self.actual_port = None
        self.initialized = False  # 初始化状态标志

        # 配置日志
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.WARNING)

        # 自动注册所有以 'exposed_' 开头的方法
        self._register_exposed_functions()

        # 设置路由
        self._setup_routes()

    def _register_exposed_functions(self):
        """自动注册所有暴露的函数"""
        for attr_name in dir(self):
            if attr_name.startswith('exposed_'):
                func = getattr(self, attr_name)
                if callable(func):
                    func_name = attr_name[8:]  # 去掉 'exposed_' 前缀
                    self.functions[func_name] = func
                    print(f"[OK] Registered function: {func_name}")

    def register_function(self, name: str, func: Callable):
        """手动注册函数"""
        self.functions[name] = func
        print(f"[OK] Manually registered function: {name}")

    def get_global(self, name: str, default: Any = None) -> Any:
        """获取全局变量值"""
        return self.globals.get(name, default)

    def _setup_routes(self):
        """设置HTTP路由"""

        @self.app.route('/health', methods=['GET'])
        def health_check():
            """健康检查端点"""
            return jsonify({
                "status": "healthy",
                "initialized": self.initialized,
                "functions": list(self.functions.keys()),
                "globals_count": len(self.globals)
            })

        @self.app.route('/initialize', methods=['POST'])
        def initialize_worker():
            """延迟初始化端点 - 在设置全局变量后调用"""
            try:
                if self.initialized:
                    return jsonify({
                        "success": True,
                        "message": "Worker已经初始化过了"
                    })

                print("\n[INIT] 开始初始化 Worker...")
                print(f"[INIT] 当前全局变量: {list(self.globals.keys())}")
                
                self.initialize()
                self.initialized = True
                
                print("[OK] Worker 初始化完成!")
                return jsonify({
                    "success": True,
                    "message": "Worker初始化成功",
                    "globals_set": list(self.globals.keys())
                })

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"[ERROR] 初始化失败: {error_msg}")
                traceback.print_exc()
                return jsonify({
                    "success": False,
                    "error": error_msg,
                    "traceback": traceback.format_exc()
                }), 500

        @self.app.route('/call', methods=['POST'])
        def call_function():
            """调用Python函数的主要端点"""
            try:
                if not self.initialized:
                    return jsonify({
                        "success": False,
                        "error": "Worker还未初始化，请先调用 /initialize 端点"
                    }), 503

                data = request.get_json()
                function_name = data.get('function')
                arguments = data.get('arguments', {})

                if function_name not in self.functions:
                    return jsonify({
                        "success": False,
                        "error": f"Function '{function_name}' not found"
                    }), 404

                func = self.functions[function_name]

                # 调用函数
                if isinstance(arguments, dict):
                    result = func(**arguments)
                elif isinstance(arguments, list):
                    result = func(*arguments)
                else:
                    result = func(arguments)

                return jsonify({
                    "success": True,
                    "result": result
                })

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"[ERROR] {error_msg}")
                traceback.print_exc()

                return jsonify({
                    "success": False,
                    "error": error_msg,
                    "traceback": traceback.format_exc()
                }), 500

        @self.app.route('/functions', methods=['GET'])
        def list_functions():
            """列出所有可用的函数"""
            return jsonify({
                "functions": list(self.functions.keys())
            })

        @self.app.route('/set_global', methods=['POST'])
        def set_global():
            """设置单个全局变量"""
            try:
                data = request.get_json()
                name = data.get('name')
                value = data.get('value')

                if not name:
                    return jsonify({
                        "success": False,
                        "error": "变量名不能为空"
                    }), 400

                # 存储到全局变量字典（用于追踪）
                self.globals[name] = value
                print(f"[GLOBAL] Set '{name}' = {value} (in global scope)")

                return jsonify({
                    "success": True,
                    "message": f"全局变量 '{name}' 设置成功"
                })

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"[ERROR] Failed to set global: {error_msg}")
                return jsonify({
                    "success": False,
                    "error": error_msg,
                    "traceback": traceback.format_exc()
                }), 500

        @self.app.route('/set_globals', methods=['POST'])
        def set_globals():
            """批量设置全局变量"""
            try:
                data = request.get_json()

                if not isinstance(data, dict):
                    return jsonify({
                        "success": False,
                        "error": "请求数据必须是字典格式"
                    }), 400

                # 批量存储全局变量
                for name, value in data.items():
                    self.globals[name] = value
                    print(f"[GLOBAL] Set '{name}' = {value} (in global scope)")

                return jsonify({
                    "success": True,
                    "message": f"成功设置 {len(data)} 个全局变量",
                    "variables": list(data.keys())
                })

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"[ERROR] Failed to set globals: {error_msg}")
                return jsonify({
                    "success": False,
                    "error": error_msg,
                    "traceback": traceback.format_exc()
                }), 500

        @self.app.route('/get_globals', methods=['GET'])
        def get_globals():
            """获取所有已设置的全局变量"""
            return jsonify({
                "success": True,
                "globals": self.globals,
                "count": len(self.globals)
            })

    @abstractmethod
    def initialize(self):
        """子类实现:初始化模型等资源"""
        pass

    def start(self, blocking: bool = True):
        """启动HTTP服务器（延迟初始化模式）"""
        print("=" * 60)
        print("[START] Starting Python Worker...")
        print("=" * 60)

        # 注意：不在这里调用 initialize()！
        # 初始化将在C#端设置全局变量后，通过 /initialize 端点触发
        print("\n[INFO] Worker启动中（延迟初始化模式）...")
        print("[INFO] 将在接收全局变量后通过 /initialize 端点初始化")

        # 启动Flask服务器
        if blocking:
            self._start_server_blocking()
        else:
            self._start_server_threaded()

    def _start_server_blocking(self):
        """阻塞方式启动服务器"""
        # 如果port为0,让系统自动分配端口
        if self.port == 0:
            import socket
            sock = socket.socket()
            sock.bind(('', 0))
            self.actual_port = sock.getsockname()[1]
            sock.close()
        else:
            self.actual_port = self.port

        print(f"\n[OK] Worker ready on http://{self.host}:{self.actual_port}")
        print(f"[OK] Available functions: {list(self.functions.keys())}")
        print(f"\n[INFO] Use Ctrl+C to stop\n")
        print("=" * 60)

        # 输出端口号供C#程序读取
        sys.stdout.flush()
        print(f"WORKER_PORT:{self.actual_port}")
        sys.stdout.flush()

        self.app.run(host=self.host, port=self.actual_port, debug=False)

    def _start_server_threaded(self):
        """线程方式启动服务器(非阻塞)"""
        self.server_thread = threading.Thread(
            target=self._start_server_blocking,
            daemon=True
        )
        self.server_thread.start()
