import sys
import os
import subprocess
import json
import glob
import importlib.util
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QGroupBox, QFormLayout, QLabel, QLineEdit, QComboBox,
                               QSpinBox, QPushButton, QPlainTextEdit, QRadioButton, QTabWidget,
                               QCheckBox, QProgressBar, QScrollArea, QSpacerItem,
                               QSizePolicy)
from PySide6.QtCore import QThread, Slot
from PySide6.QtGui import QFont

from simulation_runner import SimulationWorker
from config import PATHS, load_gui_settings


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("配电网智能调度仿真与评估平台 v2.1")
        self.setGeometry(100, 100, 1200, 800)

        main_layout = QHBoxLayout()
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs, 1)
        self.create_output_panel(main_layout)

        self.create_config_page()
        self.create_train_page()
        self.create_evaluate_page()

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.connect_signals()
        self.load_settings()

        self.thread = None
        self.worker = None

    # (create_config_page, create_train_page, create_evaluate_page, create_output_panel 这几个页面创建函数无需任何修改)
    def create_config_page(self):
        config_widget = QWidget()
        layout = QVBoxLayout(config_widget)
        grid_group = QGroupBox("配电网模型")
        form_layout = QFormLayout()
        self.grid_model_combo = QComboBox()
        self.grid_model_combo.addItems(["ieee33", "ieee69", "ieee123"])
        self.solver_combo = QComboBox()
        self.solver_combo.addItems(["gurobi", "glpk", "cbc", "scip"])
        form_layout.addRow("电网模型:", self.grid_model_combo)
        form_layout.addRow("求解器:", self.solver_combo)
        grid_group.setLayout(form_layout)
        comp_group = QGroupBox("配电网组件开关")
        comp_layout = QVBoxLayout()
        self.chk_pv = QCheckBox("包含光伏 (PV)")
        self.chk_wind = QCheckBox("包含风电 (Wind)")
        self.chk_ess = QCheckBox("包含储能 (ESS)")
        self.chk_sop = QCheckBox("包含软开关 (SOP)")
        self.chk_nop = QCheckBox("包含常开点 (NOP)")
        comp_layout.addWidget(self.chk_pv)
        comp_layout.addWidget(self.chk_wind)
        comp_layout.addWidget(self.chk_ess)
        comp_layout.addWidget(self.chk_sop)
        comp_layout.addWidget(self.chk_nop)
        comp_group.setLayout(comp_layout)
        time_group = QGroupBox("时间设置")
        time_layout = QFormLayout()
        self.start_hour_spin = QSpinBox()
        self.start_hour_spin.setRange(0, 23)
        self.end_hour_spin = QSpinBox()
        self.end_hour_spin.setRange(1, 24)
        self.step_minutes_combo = QComboBox()
        self.step_minutes_combo.addItems(["60", "30", "15"])
        time_layout.addRow("仿真开始时间 (h):", self.start_hour_spin)
        time_layout.addRow("仿真结束时间 (h):", self.end_hour_spin)
        time_layout.addRow("仿真步长 (min):", self.step_minutes_combo)
        time_group.setLayout(time_layout)
        ev_group = QGroupBox("EV 充电场景数据源")
        ev_layout = QVBoxLayout()

        self.rb_ev_random = QRadioButton("随机生成 (推荐用于训练)")
        self.rb_ev_external = QRadioButton("使用外部文件 (data/my_ev_scenarios.csv)")

        # 创建一个水平布局来放置单选按钮和编辑按钮
        external_file_layout = QHBoxLayout()
        external_file_layout.addWidget(self.rb_ev_external)
        external_file_layout.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.edit_ev_scenarios_button = QPushButton("编辑 EV 场景文件")
        external_file_layout.addWidget(self.edit_ev_scenarios_button)

        ev_layout.addWidget(self.rb_ev_random)
        ev_layout.addLayout(external_file_layout)  # 添加包含按钮的水平布局

        ev_group.setLayout(ev_layout)

        # 默认选择
        self.rb_ev_random.setChecked(True)
        self.edit_ev_scenarios_button.setEnabled(False)  # 默认禁用，因为 "随机生成" 被选中
        actions_group = QGroupBox("操作")
        actions_layout = QVBoxLayout()
        self.edit_params_button = QPushButton("编辑 grid_parameters.xlsx 文件")
        self.save_settings_button = QPushButton("保存当前配置")
        self.save_settings_button.setFont(QFont("Arial", 10, QFont.Bold))
        actions_layout.addWidget(self.edit_params_button)
        actions_layout.addWidget(self.save_settings_button)
        actions_group.setLayout(actions_layout)
        layout.addWidget(grid_group)
        layout.addWidget(comp_group)
        layout.addWidget(time_group)
        layout.addWidget(ev_group)
        layout.addWidget(actions_group)
        layout.addStretch()
        self.tabs.addTab(config_widget, "参数配置")

    def create_train_page(self):
        train_widget = QWidget()
        layout = QVBoxLayout(train_widget)
        algo_group = QGroupBox("算法选择")
        algo_layout = QVBoxLayout()
        self.rb_run_baseline = QRadioButton("仅运行 Baseline (基于求解器)")
        self.rb_train_rl = QRadioButton("训练/运行强化学习模型")
        self.rl_model_combo = QComboBox()
        self.available_rl_algos = self.discover_rl_algorithms()
        self.rl_model_combo.addItems(self.available_rl_algos.keys())
        self.rb_run_baseline.setChecked(True)
        algo_layout.addWidget(self.rb_run_baseline)
        algo_layout.addWidget(self.rb_train_rl)
        algo_layout.addWidget(self.rl_model_combo)
        algo_group.setLayout(algo_layout)
        mode_group = QGroupBox("运行模式")
        mode_layout = QHBoxLayout()
        self.rb_train_single_stage = QRadioButton("一阶段潮流模式")
        self.rb_train_two_stage = QRadioButton("两阶段潮流模式")
        self.rb_train_two_stage.setChecked(True)
        mode_layout.addWidget(self.rb_train_single_stage)
        mode_layout.addWidget(self.rb_train_two_stage)
        mode_group.setLayout(mode_layout)
        run_group = QGroupBox("运行控制")
        run_layout = QVBoxLayout()
        self.train_button = QPushButton("开始运行 / 训练")
        self.train_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.stop_button = QPushButton("终止任务")
        self.stop_button.setEnabled(False)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        run_layout.addWidget(self.train_button)
        run_layout.addWidget(self.stop_button)
        run_layout.addWidget(QLabel("任务进度:"))
        run_layout.addWidget(self.progress_bar)
        run_group.setLayout(run_layout)
        layout.addWidget(algo_group)
        layout.addWidget(mode_group)
        layout.addWidget(run_group)
        layout.addStretch()
        self.tabs.addTab(train_widget, "训练与运行")

    def create_evaluate_page(self):
        eval_widget = QWidget()
        layout = QVBoxLayout(eval_widget)
        mode_group = QGroupBox("评估模式选择")
        mode_layout = QHBoxLayout()
        self.rb_eval_single_stage = QRadioButton("一阶段潮流模式")
        self.rb_eval_two_stage = QRadioButton("两阶段潮流模式")
        self.rb_eval_two_stage.setChecked(True)
        mode_layout.addWidget(self.rb_eval_single_stage)
        mode_layout.addWidget(self.rb_eval_two_stage)
        mode_group.setLayout(mode_layout)
        self.eval_algo_group = QGroupBox("选择要对比的算法")
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.eval_algo_layout = QVBoxLayout()
        scroll_content_widget = QWidget()
        scroll_content_widget.setLayout(self.eval_algo_layout)
        scroll_area.setWidget(scroll_content_widget)
        self.eval_algo_group.setLayout(QVBoxLayout())
        self.eval_algo_group.layout().addWidget(scroll_area)
        self.eval_checkboxes = {}
        self.eval_button = QPushButton("开始评估")
        self.eval_button.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(mode_group)
        layout.addWidget(self.eval_algo_group)
        layout.addWidget(self.eval_button)
        layout.addStretch()
        self.tabs.addTab(eval_widget, "评估与对比")
        self.update_eval_models_list()

    def create_output_panel(self, main_layout):
        output_widget = QWidget()
        output_layout = QVBoxLayout(output_widget)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Courier New", 10))
        self.status_label = QLabel("状态: 空闲")
        output_layout.addWidget(QLabel("运行日志:"))
        output_layout.addWidget(self.log_box)
        output_layout.addWidget(self.status_label)
        main_layout.addWidget(output_widget, 2)

    def connect_signals(self):
        self.save_settings_button.clicked.connect(self.save_settings)
        self.edit_params_button.clicked.connect(self.open_parameters_file)
        self.train_button.clicked.connect(self.start_training_or_baseline)

        # (stop_button 的连接已移至 start_worker_thread 中)

        self.rb_eval_single_stage.toggled.connect(self.update_eval_models_list)
        self.rb_eval_two_stage.toggled.connect(self.update_eval_models_list)
        self.eval_button.clicked.connect(self.start_evaluation)

        # --- ▼▼▼ 【新增】连接 EV 场景按钮信号 ▼▼▼ ---
        # 1. 将 "使用外部文件" 单选按钮的切换信号，连接到 "编辑" 按钮的 setEnabled 槽
        #    这样，只有选中 "使用外部文件" 时，"编辑" 按钮才可用
        self.rb_ev_external.toggled.connect(self.edit_ev_scenarios_button.setEnabled)
        # 2. 连接 "编辑 EV 场景文件" 按钮的点击信号到新创建的槽函数
        self.edit_ev_scenarios_button.clicked.connect(self.open_ev_scenarios_file)
        # --- ▲▲▲ 【新增结束】 ▲▲▲ ---

    # (save_settings, load_settings, start_training_or_baseline, start_evaluation 无需修改)
    @Slot()
    def save_settings(self):
        settings = {
            "grid_model": self.grid_model_combo.currentText(),
            "solver": self.solver_combo.currentText(),
            "start_hour": self.start_hour_spin.value(),
            "end_hour": self.end_hour_spin.value(),
            "step_minutes": int(self.step_minutes_combo.currentText()),
            "use_pv": self.chk_pv.isChecked(),
            "use_wind": self.chk_wind.isChecked(),
            "use_ess": self.chk_ess.isChecked(),
            "use_sop": self.chk_sop.isChecked(),
            "use_nop": self.chk_nop.isChecked(),
            # --- ▼▼▼ 【新增】保存 EV 数据源选择 ▼▼▼ ---
            "ev_data_source": "external" if self.rb_ev_external.isChecked() else "random"
            # --- ▲▲▲ 【新增结束】 ▲▲▲ ---
        }
        try:
            with open(PATHS["gui_settings"], 'w') as f:
                json.dump(settings, f, indent=4)
            self.update_log("配置已成功保存至 gui_settings.json")
        except Exception as e:
            self.update_log(f"错误: 保存配置失败: {e}")

    # ▼▼▼ 4. 修改 load_settings 方法 ▼▼▼
    def load_settings(self):
        settings = load_gui_settings()
        self.grid_model_combo.setCurrentText(settings.get("grid_model", "ieee33"))
        self.solver_combo.setCurrentText(settings.get("solver", "gurobi"))
        self.start_hour_spin.setValue(settings.get("start_hour", 1))
        self.end_hour_spin.setValue(settings.get("end_hour", 24))
        self.step_minutes_combo.setCurrentText(str(settings.get("step_minutes", 60)))
        self.chk_pv.setChecked(settings.get("use_pv", True))
        self.chk_wind.setChecked(settings.get("use_wind", True))
        self.chk_ess.setChecked(settings.get("use_ess", True))
        self.chk_sop.setChecked(settings.get("use_sop", True))
        self.chk_nop.setChecked(settings.get("use_nop", True))

        # --- ▼▼▼ 【新增】加载 EV 数据源设置 ▼▼▼ ---
        ev_source = settings.get("ev_data_source", "random")
        if ev_source == "external":
            self.rb_ev_external.setChecked(True)
        else:
            self.rb_ev_random.setChecked(True)
        # --- ▲▲▲ 【新增结束】 ▲▲▲ ---

        self.update_log("已加载配置。")

    @Slot()
    def start_training_or_baseline(self):
        if self.thread and self.thread.isRunning():
            self.update_log("警告：一个任务正在进行中，请勿重复点击。")
            return
        self.log_box.clear()
        self.save_settings()
        self.update_log("开始新任务，当前配置已自动保存。")
        self.set_ui_for_task(is_running=True)
        task_params = {}
        if self.rb_run_baseline.isChecked():
            task_type = "run_baseline"
        else:
            task_type = "train_rl"
            task_params['rl_algo_name'] = self.rl_model_combo.currentText()
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
        task_params['mode'] = 'two_stage' if self.rb_train_two_stage.isChecked() else 'single_stage'
        self.start_worker_thread(task_type, task_params)

    @Slot()
    def start_evaluation(self):
        if self.thread and self.thread.isRunning():
            self.update_log("警告：一个任务正在进行中，请勿重复点击。")
            return
        self.log_box.clear()
        self.save_settings()
        self.update_log("开始评估任务，当前配置已自动保存。")
        selected_algos = [name for name, chk in self.eval_checkboxes.items() if chk.isChecked()]
        if not selected_algos:
            self.update_log("错误：请至少选择一个算法进行评估。")
            return
        self.set_ui_for_task(is_running=True)
        task_params = {
            'selected_algos': selected_algos,
            'mode': 'two_stage' if self.rb_eval_two_stage.isChecked() else 'single_stage'
        }
        self.start_worker_thread("evaluate", task_params)

    def start_worker_thread(self, task_type, task_params):
        self.thread = QThread()
        self.worker = SimulationWorker(task_type, task_params)
        self.worker.moveToThread(self.thread)

        # ▼▼▼ 【核心修正 3】: 在启动线程前，直接连接按钮信号到worker的槽 ▼▼▼
        self.stop_button.clicked.connect(self.worker.request_stop)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.task_finished)
        self.worker.finished.connect(self.thread.quit)

        # 当任务结束后，断开连接，避免旧worker的槽被误触发
        self.worker.finished.connect(self.stop_button.clicked.disconnect)

        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.cleanup_references)
        self.worker.progress.connect(self.update_log)
        self.worker.error.connect(self.show_error)
        self.worker.progress_update.connect(self.update_progress_bar)

        self.thread.start()

    def open_ev_scenarios_file(self):
        """
        打开 EV 场景 CSV 文件 (my_ev_scenarios.csv)
        """
        try:
            # 从 config.py 获取 EV 场景文件的路径
            filepath = PATHS["ev_scenarios_csv"]
            if not os.path.exists(filepath):
                self.update_log(f"错误: 找不到文件 {filepath}")
                self.update_log(f"提示: 请先在 'data' 文件夹中创建 'my_ev_scenarios.csv' 文件。")
                return
            # 使用与 open_parameters_file 相同的逻辑打开文件
            if sys.platform == "win32":
                os.startfile(filepath)
            elif sys.platform == "darwin":
                subprocess.run(["open", filepath])
            else:
                subprocess.run(["xdg-open", filepath])
        except Exception as e:
            self.update_log(f"打开 EV 场景文件失败: {e}")



    @Slot(int, int)
    def update_progress_bar(self, value, max_value):
        self.progress_bar.setMaximum(max_value)
        self.progress_bar.setValue(value)
    def set_ui_for_task(self, is_running):
        self.tabs.setEnabled(not is_running)
        self.train_button.setEnabled(not is_running)
        self.eval_button.setEnabled(not is_running)
        self.stop_button.setEnabled(is_running)
        if not is_running:
            self.progress_bar.setVisible(False)
            self.status_label.setText(f"状态: 任务完成/已终止")
        else:
            self.status_label.setText("状态: 正在运行...")

    def update_log(self, text):
        self.log_box.appendPlainText(text.strip())

    def task_finished(self):
        self.set_ui_for_task(is_running=False)

    def show_error(self, error_text):
        # 移除字符串两端的空白符，方便判断
        clean_text = error_text.strip()

        # 检查这是否只是一个警告
        if "Warning:" in clean_text or "UserWarning:" in clean_text:
            # 如果是警告，我们将其作为普通日志打印，而不是当作错误
            self.log_box.appendPlainText(f"[警告] {clean_text}")
            # 关键：不要停止任务，因为它只是一个警告
        else:
            # 否则，这才是我们应该处理的真正错误
            self.log_box.appendPlainText(f"\n!!!!!! 发生错误 !!!!!!\n{clean_text}")
            self.status_label.setText("状态: 发生错误，已停止")
            self.set_ui_for_task(is_running=False)

    @Slot()
    def cleanup_references(self):
        self.update_log("...后台线程已清理，可以开始新任务。")
        self.thread = None
        self.worker = None

    def open_parameters_file(self):
        try:
            filepath = PATHS["grid_params_excel"]
            if not os.path.exists(filepath):
                self.update_log(f"错误: 找不到文件 {filepath}")
                return
            if sys.platform == "win32":
                os.startfile(filepath)
            elif sys.platform == "darwin":
                subprocess.run(["open", filepath])
            else:
                subprocess.run(["xdg-open", filepath])
        except Exception as e:
            self.update_log(f"打开文件失败: {e}")

    def discover_rl_algorithms(self):
        model_class_registry = {}
        from stable_baselines3 import PPO, DDPG, SAC, TD3
        model_class_registry.update({"PPO": PPO, "DDPG": DDPG, "SAC": SAC, "TD3": TD3})
        plugin_dir = "custom_algorithms"
        if os.path.isdir(plugin_dir):
            plugin_files = glob.glob(os.path.join(plugin_dir, "*.py"))
            for plugin_file in plugin_files:
                module_name = os.path.basename(plugin_file)[:-3]
                try:
                    spec = importlib.util.spec_from_file_location(module_name, plugin_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, 'register_algorithm'):
                        info = module.register_algorithm()
                        model_class_registry[info['name']] = info['class']
                except Exception as e:
                    print(f"加载自定义算法插件 {plugin_file} 失败: {e}")
        return model_class_registry

    @Slot()
    def update_eval_models_list(self):
        """
        根据当前评估模式（单阶段 / 两阶段），自动扫描 models 目录，
        兼容以下几种命名后缀：
        - 两阶段:   _two_stage, _2stage
        - 单阶段:   _single_stage, _one_stage, _1stage
        只要目录下存在 best_model.zip，就认为是一个可评估模型。
        """
        # 1. 清空旧的勾选框
        for chk in self.eval_checkboxes.values():
            self.eval_algo_layout.removeWidget(chk)
            chk.deleteLater()
        self.eval_checkboxes.clear()

        # 2. 默认添加 Baseline
        chk_baseline = QCheckBox("Baseline (全局优化)")
        chk_baseline.setChecked(True)
        self.eval_algo_layout.addWidget(chk_baseline)
        self.eval_checkboxes["Baseline"] = chk_baseline

        # 3. 根据当前选择的评估模式，确定允许的后缀
        if self.rb_eval_two_stage.isChecked():
            suffix_candidates = ["_two_stage", "_2stage"]
        else:
            suffix_candidates = ["_single_stage", "_one_stage", "_1stage"]

        models_root = PATHS["models_dir"]
        model_folders = [
            f for f in os.listdir(models_root)
            if os.path.isdir(os.path.join(models_root, f))
        ]

        # 4. 扫描 models 目录
        for folder_name in model_folders:
            # 后缀不匹配，就跳过
            if not any(folder_name.endswith(suf) for suf in suffix_candidates):
                continue

            model_path = os.path.join(models_root, folder_name, "best_model.zip")
            if not os.path.exists(model_path):
                continue

            # 生成在 GUI 上展示的名字，比如：
            #  - best_sac_2stage    -> Sac_2Stage
            #  - best_sac_two_stage -> Sac_Two_Stage
            display_name = (
                folder_name
                .replace("best_", "")
                .replace("_", " ")
                .title()
                .replace(" ", "_")
            )

            chk = QCheckBox(display_name)
            chk.setChecked(True)
            self.eval_algo_layout.addWidget(chk)
            self.eval_checkboxes[display_name] = chk

        self.eval_algo_layout.addStretch()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())