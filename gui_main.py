import sys
import os
import subprocess
import json
import glob
import importlib.util

from PySide6.QtWidgets import (
    QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout, QGridLayout,QGroupBox,QFormLayout,QLabel,
    QComboBox,QSpinBox,QPushButton,QPlainTextEdit,QRadioButton, QTabWidget,QCheckBox,QProgressBar,QScrollArea,
    QSpacerItem,QSizePolicy,QDoubleSpinBox,QStackedWidget,QFrame,QSplitter,
)
from PySide6.QtCore import QThread, Slot, Qt
from PySide6.QtGui import QFont

from simulation_runner import SimulationWorker
from config import PATHS, load_gui_settings


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("配电网智能调度仿真与评估平台 v2.1")
        self.setGeometry(80, 60, 1480, 920)
        self.setMinimumSize(1280, 820)

        self.thread = None
        self.worker = None
        self.eval_checkboxes = {}
        self.summary_value_labels = {}

        self.apply_theme()
        self.build_ui()
        self.connect_signals()
        self.load_settings()
        self.refresh_overview()
        self.refresh_train_controls()

    def build_ui(self):
        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(14)

        root_layout.addWidget(self.create_header_panel())

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(False)
        left_layout.addWidget(self.tabs)

        splitter.addWidget(left_widget)
        splitter.addWidget(self.create_output_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1080, 360])

        self.create_config_page()
        self.create_rl_tuning_page()
        self.create_station_operator_page()
        ops_index = self.tabs.indexOf(self.ops_widget)
        self.tabs.setTabVisible(ops_index, True)
        self.create_train_page()
        self.create_evaluate_page()

        root_layout.addWidget(splitter, 1)
        self.setCentralWidget(central_widget)

    def apply_theme(self):
        self.setStyleSheet(
            """
            QWidget#centralWidget {
                background: #f4f7fb;
            }
            QLabel#pageTitle {
                font-size: 24px;
                font-weight: 700;
                color: #1b2a41;
            }
            QLabel#pageSubtitle {
                color: #60708a;
                font-size: 13px;
            }
            QFrame#heroCard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #1e63ff, stop:1 #2f8cff);
                border-radius: 18px;
            }
            QLabel#heroTitle {
                color: white;
                font-size: 20px;
                font-weight: 700;
            }
            QLabel#heroDesc {
                color: rgba(255,255,255,0.9);
                font-size: 13px;
            }
            QFrame#summaryCard,
            QFrame#sideCard,
            QScrollArea,
            QWidget#tabPage {
                background: transparent;
            }
            QFrame#summaryCard,
            QFrame#sidePanel,
            QGroupBox {
                background: white;
                border: 1px solid #dbe3ef;
                border-radius: 14px;
            }
            QFrame#summaryCard {
                min-height: 90px;
            }
            QLabel#summaryTitle {
                color: #6b7b93;
                font-size: 12px;
            }
            QLabel#summaryValue {
                color: #1b2a41;
                font-size: 18px;
                font-weight: 700;
            }
            QLabel#summaryHint {
                color: #8a97ab;
                font-size: 11px;
            }
            QTabWidget::pane {
                border: none;
                background: transparent;
                top: -1px;
            }
            QTabBar::tab {
                background: #e9eef7;
                color: #51627a;
                border: none;
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
                padding: 10px 18px;
                margin-right: 6px;
                min-width: 118px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background: white;
                color: #1e63ff;
            }
            QTabBar::tab:hover:!selected {
                background: #dde6f5;
            }
            QGroupBox {
                margin-top: 14px;
                padding: 14px 16px 16px 16px;
                font-size: 13px;
                font-weight: 700;
                color: #22324d;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px;
            }
            QLabel {
                color: #24364f;
            }
            QLabel#mutedText,
            QLabel#panelCaption,
            QLabel#statusCaption,
            QLabel#trainHint {
                color: #72839b;
            }
            QLabel#panelTitle {
                font-size: 15px;
                font-weight: 700;
                color: #1b2a41;
            }
            QLabel#statusValue {
                font-size: 20px;
                font-weight: 700;
                color: #1b2a41;
            }
            QLabel#taskValue {
                font-size: 13px;
                color: #63748d;
            }
            QLineEdit,
            QComboBox,
            QSpinBox,
            QDoubleSpinBox,
            QPlainTextEdit,
            QScrollArea {
                border: 1px solid #ccd7e6;
                border-radius: 10px;
                background: white;
            }
            QComboBox,
            QSpinBox,
            QDoubleSpinBox {
                min-height: 38px;
                padding: 0 12px;
            }
            QComboBox::drop-down {
                border: none;
                width: 28px;
            }
            QPlainTextEdit {
                padding: 10px;
                background: #0f1724;
                color: #d9e1ef;
                border-radius: 12px;
                border: 1px solid #1f2d42;
            }
            QPushButton {
                min-height: 38px;
                border-radius: 10px;
                border: 1px solid #cbd7e8;
                background: white;
                color: #1e2d42;
                padding: 0 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #f4f8ff;
                border-color: #adc4ef;
            }
            QPushButton:disabled {
                background: #eef2f7;
                color: #9aa8bb;
                border-color: #d7dee8;
            }
            QPushButton#primaryButton {
                background: #1e63ff;
                color: white;
                border: none;
            }
            QPushButton#primaryButton:hover {
                background: #1755de;
            }
            QPushButton#secondaryButton {
                background: #edf4ff;
                color: #1452d1;
                border: 1px solid #c3d8ff;
            }
            QPushButton#secondaryButton:hover {
                background: #dfeeff;
            }
            QPushButton#dangerButton {
                background: #fff1f0;
                color: #d0473b;
                border: 1px solid #f1c0ba;
            }
            QPushButton#dangerButton:hover {
                background: #ffe4e0;
            }
            QRadioButton,
            QCheckBox {
                spacing: 10px;
                min-height: 26px;
                color: #23344d;
            }
            QRadioButton::indicator,
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QProgressBar {
                min-height: 12px;
                border-radius: 6px;
                background: #edf2fb;
                border: none;
                text-align: center;
                color: #45566f;
            }
            QProgressBar::chunk {
                background: #1e63ff;
                border-radius: 6px;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                width: 10px;
                background: transparent;
                margin: 6px 0 6px 0;
            }
            QScrollBar::handle:vertical {
                background: #c9d4e4;
                border-radius: 5px;
                min-height: 28px;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: transparent;
                border: none;
            }
            QSplitter::handle {
                background: #e2e8f2;
            }
            """
        )

    def create_header_panel(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        hero = QFrame()
        hero.setObjectName("heroCard")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(22, 18, 22, 18)
        hero_layout.setSpacing(18)

        title_col = QVBoxLayout()
        title_col.setSpacing(4)
        title = QLabel("VGridSim")
        title.setObjectName("heroTitle")
        subtitle = QLabel("面向车网互动的人工智能调控算法开源测试平台")
        subtitle.setObjectName("heroDesc")
        subtitle.setWordWrap(True)
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        hero_layout.addLayout(title_col, 1)

        hero_hint = QLabel("左侧完成核心操作，右侧统一查看状态、进度与日志。")
        hero_hint.setObjectName("heroDesc")
        hero_hint.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hero_hint.setWordWrap(True)
        hero_layout.addWidget(hero_hint, 1)

        layout.addWidget(hero)

        summary_grid = QGridLayout()
        summary_grid.setHorizontalSpacing(12)
        summary_grid.setVerticalSpacing(12)

        summary_items = [
            ("grid_model", "电网模型", "IEEE 测试系统"),
            ("solver", "求解器", "优化后端"),
            ("mode", "运行模式", "单阶段 / 两阶段"),
            ("ev_source", "EV 数据", "随机 / 外部文件"),

        ]

        for index, (key, title_text, hint_text) in enumerate(summary_items):
            card = QFrame()
            card.setObjectName("summaryCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(16, 14, 16, 14)
            card_layout.setSpacing(4)

            title_label = QLabel(title_text)
            title_label.setObjectName("summaryTitle")
            value_label = QLabel("--")
            value_label.setObjectName("summaryValue")
            hint_label = QLabel(hint_text)
            hint_label.setObjectName("summaryHint")

            card_layout.addWidget(title_label)
            card_layout.addWidget(value_label)
            card_layout.addWidget(hint_label)

            self.summary_value_labels[key] = value_label
            summary_grid.addWidget(card, 0, index)

        layout.addLayout(summary_grid)
        return container

    def wrap_tab(self, content_widget, title):
        content_widget.setObjectName("tabPage")
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setWidget(content_widget)
        self.tabs.addTab(scroll_area, title)

    def create_intro_label(self, text):
        label = QLabel(text)
        label.setObjectName("mutedText")
        label.setWordWrap(True)
        return label

    def create_config_page(self):
        config_widget = QWidget()
        layout = QVBoxLayout(config_widget)
        layout.setContentsMargins(8, 8, 8, 12)
        layout.setSpacing(14)
        layout.addWidget(self.create_intro_label("先确定电网场景与物理参数，再开始训练或评估。建议先保存配置，避免不同实验之间相互覆盖"))

        grid_group = QGroupBox("配电网模型")
        form_layout = QFormLayout()
        form_layout.setSpacing(12)
        self.grid_model_combo = QComboBox()
        self.grid_model_combo.addItems(["ieee33", "ieee69", "ieee123"])
        self.solver_combo = QComboBox()
        self.solver_combo.addItems(["gurobi", "glpk", "cbc", "scip"])
        form_layout.addRow("电网模型:", self.grid_model_combo)
        form_layout.addRow("求解器:", self.solver_combo)
        grid_group.setLayout(form_layout)

        comp_group = QGroupBox("配电网组件开关")
        comp_layout = QVBoxLayout()
        comp_layout.setSpacing(10)
        self.chk_pv = QCheckBox("包含光伏 (PV)")
        self.chk_wind = QCheckBox("包含风电 (Wind)")
        self.chk_ess = QCheckBox("包含储能 (ESS)")
        self.chk_sop = QCheckBox("包含软开关 (SOP)")
        self.chk_nop = QCheckBox("包含常开点 (NOP)")
        for checkbox in [self.chk_pv, self.chk_wind, self.chk_ess, self.chk_sop, self.chk_nop]:
            comp_layout.addWidget(checkbox)
        comp_group.setLayout(comp_layout)

        time_group = QGroupBox("时间设置")
        time_layout = QFormLayout()
        time_layout.setSpacing(12)
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
        ev_layout.setSpacing(12)
        self.rb_ev_random = QRadioButton("随机生成（推荐用于训练）")
        self.rb_ev_external = QRadioButton("使用外部文件（data/my_ev_scenarios.csv）")

        external_file_layout = QHBoxLayout()
        external_file_layout.setSpacing(12)
        external_file_layout.addWidget(self.rb_ev_external)
        external_file_layout.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.edit_ev_scenarios_button = QPushButton("编辑 EV 场景文件")
        self.edit_ev_scenarios_button.setObjectName("secondaryButton")
        external_file_layout.addWidget(self.edit_ev_scenarios_button)

        ev_layout.addWidget(self.rb_ev_random)
        ev_layout.addLayout(external_file_layout)
        ev_group.setLayout(ev_layout)

        self.rb_ev_random.setChecked(True)
        self.edit_ev_scenarios_button.setEnabled(False)

        ev_param_group = QGroupBox("EV 物理参数设置")
        ev_param_layout = QFormLayout()
        ev_param_layout.setSpacing(12)

        self.ev_cap_spin = QDoubleSpinBox()
        self.ev_cap_spin.setRange(10.0, 200.0)
        self.ev_cap_spin.setValue(70.0)
        self.ev_cap_spin.setSuffix(" kWh")

        self.ev_charge_spin = QDoubleSpinBox()
        self.ev_charge_spin.setRange(1.0, 300.0)
        self.ev_charge_spin.setValue(60.0)
        self.ev_charge_spin.setSuffix(" kW")

        self.ev_discharge_spin = QDoubleSpinBox()
        self.ev_discharge_spin.setRange(0.0, 300.0)
        self.ev_discharge_spin.setValue(25.0)
        self.ev_discharge_spin.setSuffix(" kW")

        self.ev_eff_c_spin = QDoubleSpinBox()
        self.ev_eff_c_spin.setRange(0.1, 1.0)
        self.ev_eff_c_spin.setValue(0.95)
        self.ev_eff_c_spin.setSingleStep(0.01)

        self.ev_eff_d_spin = QDoubleSpinBox()
        self.ev_eff_d_spin.setRange(0.1, 1.0)
        self.ev_eff_d_spin.setValue(0.90)
        self.ev_eff_d_spin.setSingleStep(0.01)

        ev_param_layout.addRow("电池容量 (capacity):", self.ev_cap_spin)
        ev_param_layout.addRow("最大充电功率 (max_charge):", self.ev_charge_spin)
        ev_param_layout.addRow("最大放电功率 (max_discharge):", self.ev_discharge_spin)
        ev_param_layout.addRow("充电效率 (charge_eff):", self.ev_eff_c_spin)
        ev_param_layout.addRow("放电效率 (discharge_eff):", self.ev_eff_d_spin)
        ev_param_group.setLayout(ev_param_layout)

        actions_group = QGroupBox("操作")
        actions_layout = QVBoxLayout()
        actions_layout.setSpacing(10)
        self.edit_params_button = QPushButton("编辑 grid_parameters.xlsx 文件")
        self.edit_params_button.setObjectName("secondaryButton")
        self.save_settings_button = QPushButton("保存当前配置")
        self.save_settings_button.setObjectName("primaryButton")
        self.save_settings_button.setFont(QFont("Arial", 10, QFont.Bold))
        actions_layout.addWidget(self.edit_params_button)
        actions_layout.addWidget(self.save_settings_button)
        actions_layout.addWidget(self.create_intro_label("建议每次调整参数后先保存，再进入训练或评估页面。"))
        actions_group.setLayout(actions_layout)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(14)
        grid.addWidget(grid_group, 0, 0)
        grid.addWidget(comp_group, 0, 1)
        grid.addWidget(time_group, 1, 0)
        grid.addWidget(ev_group, 1, 1)
        grid.addWidget(ev_param_group, 2, 0)
        grid.addWidget(actions_group, 2, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        layout.addLayout(grid)
        layout.addStretch()
        self.wrap_tab(config_widget, "参数配置")

    def create_rl_tuning_page(self):
        tuning_widget = QWidget()
        layout = QVBoxLayout(tuning_widget)
        layout.setContentsMargins(8, 8, 8, 12)
        layout.setSpacing(14)
        layout.addWidget(self.create_intro_label("把通用超参数、算法专属参数和奖励权重分别展示，方便调试"))

        hyper_group = QGroupBox("RL 算法超参数")
        hyper_layout = QVBoxLayout()
        hyper_layout.setSpacing(12)

        common_group = QGroupBox("通用超参数")
        common_form = QFormLayout()
        common_form.setSpacing(12)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0001)
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.1, 1.0)
        self.gamma_spin.setDecimals(3)
        self.gamma_spin.setSingleStep(0.01)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(16, 4096)
        self.batch_spin.setSingleStep(32)
        common_form.addRow("学习率 (Learning Rate):", self.lr_spin)
        common_form.addRow("折扣因子 (Gamma):", self.gamma_spin)
        common_form.addRow("批次大小 (Batch Size):", self.batch_spin)
        common_group.setLayout(common_form)

        specific_group = QGroupBox("算法专属超参数")
        specific_layout = QVBoxLayout()
        specific_layout.setSpacing(12)

        algo_select_layout = QHBoxLayout()
        algo_select_layout.addWidget(QLabel("选择算法预览参数:"))
        self.tuning_algo_combo = QComboBox()
        self.tuning_algo_combo.addItems(["PPO", "SAC", "DDPG", "TD3"])
        algo_select_layout.addWidget(self.tuning_algo_combo)
        algo_select_layout.addStretch()
        specific_layout.addLayout(algo_select_layout)

        self.algo_stacked_widget = QStackedWidget()

        ppo_widget = QWidget()
        ppo_form = QFormLayout(ppo_widget)
        ppo_form.setSpacing(12)
        self.ppo_clip_spin = QDoubleSpinBox()
        self.ppo_clip_spin.setRange(0.0, 1.0)
        self.ppo_clip_spin.setSingleStep(0.1)
        self.ppo_ent_spin = QDoubleSpinBox()
        self.ppo_ent_spin.setRange(0.0, 1.0)
        self.ppo_ent_spin.setSingleStep(0.01)
        ppo_form.addRow("裁剪范围 (clip_range):", self.ppo_clip_spin)
        ppo_form.addRow("熵系数 (ent_coef):", self.ppo_ent_spin)
        self.algo_stacked_widget.addWidget(ppo_widget)

        sac_widget = QWidget()
        sac_form = QFormLayout(sac_widget)
        sac_form.setSpacing(12)
        self.sac_tau_spin = QDoubleSpinBox()
        self.sac_tau_spin.setRange(0.0, 1.0)
        self.sac_tau_spin.setDecimals(3)
        self.sac_ent_spin = QDoubleSpinBox()
        self.sac_ent_spin.setRange(0.0, 1.0)
        self.sac_ent_spin.setSingleStep(0.01)
        sac_form.addRow("软更新系数 (tau):", self.sac_tau_spin)
        sac_form.addRow("熵系数 (ent_coef):", self.sac_ent_spin)
        self.algo_stacked_widget.addWidget(sac_widget)

        ddpg_widget = QWidget()
        ddpg_form = QFormLayout(ddpg_widget)
        ddpg_form.setSpacing(12)
        self.ddpg_tau_spin = QDoubleSpinBox()
        self.ddpg_tau_spin.setRange(0.0, 1.0)
        self.ddpg_tau_spin.setDecimals(3)
        self.ddpg_noise_spin = QDoubleSpinBox()
        self.ddpg_noise_spin.setRange(0.0, 1.0)
        self.ddpg_noise_spin.setSingleStep(0.1)
        ddpg_form.addRow("软更新系数 (tau):", self.ddpg_tau_spin)
        ddpg_form.addRow("动作噪声 (action_noise):", self.ddpg_noise_spin)
        self.algo_stacked_widget.addWidget(ddpg_widget)

        td3_widget = QWidget()
        td3_form = QFormLayout(td3_widget)
        td3_form.setSpacing(12)
        self.td3_delay_spin = QSpinBox()
        self.td3_delay_spin.setRange(1, 10)
        self.td3_noise_spin = QDoubleSpinBox()
        self.td3_noise_spin.setRange(0.0, 1.0)
        self.td3_noise_spin.setSingleStep(0.1)
        td3_form.addRow("策略延迟更新 (policy_delay):", self.td3_delay_spin)
        td3_form.addRow("目标动作噪声 (target_noise):", self.td3_noise_spin)
        self.algo_stacked_widget.addWidget(td3_widget)

        self.tuning_algo_combo.currentIndexChanged.connect(self.algo_stacked_widget.setCurrentIndex)

        specific_layout.addWidget(self.algo_stacked_widget)
        specific_group.setLayout(specific_layout)

        hyper_layout.addWidget(common_group)
        hyper_layout.addWidget(specific_group)
        hyper_group.setLayout(hyper_layout)

        reward_group = QGroupBox("奖励函数惩罚项权重")
        reward_form = QFormLayout()
        reward_form.setSpacing(12)

        self.rew_ev_spin = QDoubleSpinBox()
        self.rew_ev_spin.setRange(-10000, 0)
        self.rew_ev_spin.setValue(-100)
        self.rew_volt_spin = QDoubleSpinBox()
        self.rew_volt_spin.setRange(-10000, 0)
        self.rew_volt_spin.setValue(-100)
        self.rew_cost_spin = QDoubleSpinBox()
        self.rew_cost_spin.setRange(0, 100)
        self.rew_cost_spin.setValue(1)
        self.rew_fail_spin = QDoubleSpinBox()
        self.rew_fail_spin.setRange(-50000, 0)
        self.rew_fail_spin.setValue(-5000)

        reward_form.addRow("EV未充满惩罚 (ev_shortage):", self.rew_ev_spin)
        reward_form.addRow("电压越限惩罚 (voltage_violation):", self.rew_volt_spin)
        reward_form.addRow("成本惩罚乘子 (cost_penalty):", self.rew_cost_spin)
        reward_form.addRow("潮流不收敛惩罚 (opendss_failure):", self.rew_fail_spin)
        reward_group.setLayout(reward_form)

        tuning_grid = QGridLayout()
        tuning_grid.setHorizontalSpacing(14)
        tuning_grid.setVerticalSpacing(14)
        tuning_grid.addWidget(hyper_group, 0, 0)
        tuning_grid.addWidget(reward_group, 0, 1)
        tuning_grid.setColumnStretch(0, 3)
        tuning_grid.setColumnStretch(1, 2)

        layout.addLayout(tuning_grid)
        layout.addStretch()
        self.wrap_tab(tuning_widget, "RL参数调试")

    def create_station_operator_page(self):
        self.ops_widget = QWidget()
        layout = QVBoxLayout(self.ops_widget)
        layout.setContentsMargins(8, 8, 8, 12)
        layout.setSpacing(14)
        layout.addWidget(self.create_intro_label("这一页把奖励主体从系统运营视角切换为充电站运营视角，便于论文中做不同目标函数的对比。"))

        mode_group = QGroupBox("奖励主体选择")
        mode_form = QFormLayout()
        mode_form.setSpacing(12)
        self.reward_mode_combo = QComboBox()
        self.reward_mode_combo.addItem("电网运营商模式（原始奖励）", userData="grid_operator")
        self.reward_mode_combo.addItem("充电站运营商模式（4.5净收益）", userData="station_operator")
        mode_form.addRow("奖励模式:", self.reward_mode_combo)
        mode_group.setLayout(mode_form)

        pricing_group = QGroupBox("运营商结算参数")
        pricing_form = QFormLayout()
        pricing_form.setSpacing(12)

        self.ops_charge_price_spin = QDoubleSpinBox()
        self.ops_charge_price_spin.setRange(0.0, 10.0)
        self.ops_charge_price_spin.setDecimals(2)
        self.ops_charge_price_spin.setSingleStep(0.05)
        self.ops_charge_price_spin.setValue(1.20)
        self.ops_charge_price_spin.setSuffix(" 元/kWh")

        self.ops_discharge_price_spin = QDoubleSpinBox()
        self.ops_discharge_price_spin.setRange(0.0, 10.0)
        self.ops_discharge_price_spin.setDecimals(2)
        self.ops_discharge_price_spin.setSingleStep(0.05)
        self.ops_discharge_price_spin.setValue(0.80)
        self.ops_discharge_price_spin.setSuffix(" 元/kWh")

        pricing_form.addRow("充电服务电价 π^(u,c):", self.ops_charge_price_spin)
        pricing_form.addRow("V2G补贴电价 π^(u,d):", self.ops_discharge_price_spin)
        pricing_group.setLayout(pricing_form)

        cost_group = QGroupBox("附加成本项设置")
        cost_layout = QVBoxLayout()
        cost_layout.setSpacing(10)
        self.chk_ops_include_grid_cost = QCheckBox("计入购电成本（谨慎使用，可能与价差重复计价）")
        self.chk_ops_include_generation_cost = QCheckBox("计入发电成本")
        self.chk_ops_include_ess_cost = QCheckBox("计入 ESS 放电成本")
        self.chk_ops_include_sop_loss_cost = QCheckBox("计入 SOP 损耗成本")
        self.chk_ops_include_penalty_cost = QCheckBox("计入惩罚项成本 C_p")

        self.chk_ops_include_grid_cost.setChecked(False)
        self.chk_ops_include_generation_cost.setChecked(True)
        self.chk_ops_include_ess_cost.setChecked(True)
        self.chk_ops_include_sop_loss_cost.setChecked(True)
        self.chk_ops_include_penalty_cost.setChecked(True)

        for checkbox in [
            self.chk_ops_include_grid_cost,
            self.chk_ops_include_generation_cost,
            self.chk_ops_include_ess_cost,
            self.chk_ops_include_sop_loss_cost,
            self.chk_ops_include_penalty_cost,
        ]:
            cost_layout.addWidget(checkbox)
        cost_group.setLayout(cost_layout)

        tip_group = QGroupBox("说明")
        tip_layout = QVBoxLayout()
        tip = QLabel(
            "1) grid_operator：沿用当前“系统总成本最小化”奖励。\n"
            "2) station_operator：按论文 4.5，用‘充电服务价差 + V2G价差 - 附加成本’计算奖励。\n"
            "3) 做论文图表时，建议保持其他设置不变，只切换奖励主体和附加成本项。"
        )
        tip.setWordWrap(True)
        tip_layout.addWidget(tip)
        tip_group.setLayout(tip_layout)

        ops_grid = QGridLayout()
        ops_grid.setHorizontalSpacing(14)
        ops_grid.setVerticalSpacing(14)
        ops_grid.addWidget(mode_group, 0, 0)
        ops_grid.addWidget(pricing_group, 0, 1)
        ops_grid.addWidget(cost_group, 1, 0, 1, 2)
        ops_grid.addWidget(tip_group, 2, 0, 1, 2)
        ops_grid.setColumnStretch(0, 1)
        ops_grid.setColumnStretch(1, 1)

        layout.addLayout(ops_grid)
        layout.addStretch()
        self.tabs.addTab(self.ops_widget, "运营商模块")


    @Slot()
    def on_reward_mode_changed(self):
        mode = self.reward_mode_combo.currentData()
        enabled = mode == "station_operator"

        self.ops_charge_price_spin.setEnabled(enabled)
        self.ops_discharge_price_spin.setEnabled(enabled)
        self.chk_ops_include_grid_cost.setEnabled(enabled)
        self.chk_ops_include_generation_cost.setEnabled(enabled)
        self.chk_ops_include_ess_cost.setEnabled(enabled)
        self.chk_ops_include_sop_loss_cost.setEnabled(enabled)
        self.chk_ops_include_penalty_cost.setEnabled(enabled)
        self.refresh_overview()

    def create_train_page(self):
        train_widget = QWidget()
        layout = QVBoxLayout(train_widget)
        layout.setContentsMargins(8, 8, 8, 12)
        layout.setSpacing(14)
        layout.addWidget(self.create_intro_label("训练页只保留算法选择、运行模式和执行按钮，实时状态在右侧显示"))

        algo_group = QGroupBox("算法选择")
        algo_layout = QVBoxLayout()
        algo_layout.setSpacing(10)
        self.rb_run_baseline = QRadioButton("仅运行 Baseline（基于求解器）")
        self.rb_train_rl = QRadioButton("训练 / 运行强化学习模型")
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
        mode_layout.setSpacing(18)
        self.rb_train_single_stage = QRadioButton("单阶段潮流模式")
        self.rb_train_two_stage = QRadioButton("两阶段潮流模式")
        self.rb_train_two_stage.setChecked(True)
        mode_layout.addWidget(self.rb_train_single_stage)
        mode_layout.addWidget(self.rb_train_two_stage)
        mode_layout.addStretch()
        mode_group.setLayout(mode_layout)

        run_group = QGroupBox("运行控制")
        run_layout = QVBoxLayout()
        run_layout.setSpacing(12)
        self.train_button = QPushButton("开始运行 / 训练")
        self.train_button.setObjectName("primaryButton")
        self.train_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.stop_button = QPushButton("终止任务")
        self.stop_button.setObjectName("dangerButton")
        self.stop_button.setEnabled(False)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)

        self.train_hint_label = QLabel("实时进度与运行日志显示在右侧状态面板。")
        self.train_hint_label.setObjectName("trainHint")
        self.train_hint_label.setWordWrap(True)

        run_layout.addWidget(self.train_button)
        run_layout.addWidget(self.stop_button)
        run_layout.addWidget(QLabel("任务进度："))
        run_layout.addWidget(self.progress_bar)
        run_layout.addWidget(self.train_hint_label)
        run_group.setLayout(run_layout)

        train_grid = QGridLayout()
        train_grid.setHorizontalSpacing(14)
        train_grid.setVerticalSpacing(14)
        train_grid.addWidget(algo_group, 0, 0)
        train_grid.addWidget(mode_group, 0, 1)
        train_grid.addWidget(run_group, 1, 0, 1, 2)
        train_grid.setColumnStretch(0, 1)
        train_grid.setColumnStretch(1, 1)

        layout.addLayout(train_grid)
        layout.addStretch()
        self.wrap_tab(train_widget, "训练与运行")

    def create_evaluate_page(self):
        eval_widget = QWidget()
        layout = QVBoxLayout(eval_widget)
        layout.setContentsMargins(8, 8, 8, 12)
        layout.setSpacing(14)
        layout.addWidget(self.create_intro_label(""))

        mode_group = QGroupBox("评估模式选择")
        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(18)
        self.rb_eval_single_stage = QRadioButton("一阶段潮流模式")
        self.rb_eval_two_stage = QRadioButton("两阶段潮流模式")
        self.rb_eval_two_stage.setChecked(True)
        mode_layout.addWidget(self.rb_eval_single_stage)
        mode_layout.addWidget(self.rb_eval_two_stage)
        mode_layout.addStretch()
        mode_group.setLayout(mode_layout)

        self.eval_algo_group = QGroupBox("选择要对比的算法")
        eval_group_layout = QVBoxLayout()
        eval_group_layout.setSpacing(10)
        self.eval_selection_label = QLabel("已选择 0 个算法")
        self.eval_selection_label.setObjectName("panelCaption")
        eval_group_layout.addWidget(self.eval_selection_label)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content_widget = QWidget()
        self.eval_algo_layout = QVBoxLayout(scroll_content_widget)
        self.eval_algo_layout.setContentsMargins(6, 6, 6, 6)
        self.eval_algo_layout.setSpacing(10)
        scroll_area.setWidget(scroll_content_widget)
        eval_group_layout.addWidget(scroll_area)
        self.eval_algo_group.setLayout(eval_group_layout)

        self.eval_button = QPushButton("开始评估")
        self.eval_button.setObjectName("primaryButton")
        self.eval_button.setFont(QFont("Arial", 12, QFont.Bold))

        layout.addWidget(mode_group)
        layout.addWidget(self.eval_algo_group, 1)
        layout.addWidget(self.eval_button)
        self.wrap_tab(eval_widget, "评估与对比")
        self.update_eval_models_list()

    def create_output_panel(self):
        output_widget = QFrame()
        output_widget.setObjectName("sidePanel")
        output_layout = QVBoxLayout(output_widget)
        output_layout.setContentsMargins(16, 16, 16, 16)
        output_layout.setSpacing(14)

        status_card = QGroupBox("状态总览")
        status_layout = QVBoxLayout()
        status_layout.setSpacing(8)
        status_caption = QLabel("当前任务状态")
        status_caption.setObjectName("statusCaption")
        self.status_label = QLabel("空闲")
        self.status_label.setObjectName("statusValue")
        self.task_label = QLabel("当前任务：无")
        self.task_label.setObjectName("taskValue")
        self.side_progress_hint = QLabel("执行训练任务时会显示进度条。")
        self.side_progress_hint.setObjectName("panelCaption")
        self.side_progress_hint.setWordWrap(True)
        status_layout.addWidget(status_caption)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.task_label)
        status_layout.addWidget(self.side_progress_hint)
        status_card.setLayout(status_layout)

        quick_card = QGroupBox("快速提示")
        quick_layout = QVBoxLayout()
        quick_layout.setSpacing(8)
        quick_layout.addWidget(self.create_intro_label("训练前先保存参数；评估前确认 models 目录下存在 best_model.zip。"))
        self.clear_log_button = QPushButton("清空日志")
        self.clear_log_button.setObjectName("secondaryButton")
        quick_layout.addWidget(self.clear_log_button)
        quick_card.setLayout(quick_layout)

        log_card = QGroupBox("运行日志")
        log_layout = QVBoxLayout()
        log_layout.setSpacing(10)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Consolas", 10))
        log_layout.addWidget(self.log_box, 1)
        log_card.setLayout(log_layout)

        output_layout.addWidget(status_card)
        output_layout.addWidget(quick_card)
        output_layout.addWidget(log_card, 1)
        return output_widget

    def connect_signals(self):
        self.save_settings_button.clicked.connect(self.save_settings)
        self.edit_params_button.clicked.connect(self.open_parameters_file)
        self.train_button.clicked.connect(self.start_training_or_baseline)
        self.eval_button.clicked.connect(self.start_evaluation)
        self.clear_log_button.clicked.connect(self.log_box.clear)

        self.rb_eval_single_stage.toggled.connect(self.update_eval_models_list)
        self.rb_eval_two_stage.toggled.connect(self.update_eval_models_list)

        self.rb_ev_external.toggled.connect(self.edit_ev_scenarios_button.setEnabled)
        self.edit_ev_scenarios_button.clicked.connect(self.open_ev_scenarios_file)

        self.reward_mode_combo.currentIndexChanged.connect(self.on_reward_mode_changed)
        self.rb_run_baseline.toggled.connect(self.refresh_train_controls)
        self.rb_train_rl.toggled.connect(self.refresh_train_controls)

        for signal in [
            self.grid_model_combo.currentTextChanged,
            self.solver_combo.currentTextChanged,
            self.step_minutes_combo.currentTextChanged,
            self.tabs.currentChanged,
            self.rb_ev_random.toggled,
            self.rb_ev_external.toggled,
            self.rb_train_single_stage.toggled,
            self.rb_train_two_stage.toggled,
            self.rb_run_baseline.toggled,
            self.rb_train_rl.toggled,
        ]:
            signal.connect(self.refresh_overview)

    @Slot()
    def refresh_train_controls(self):
        enable_rl = self.rb_train_rl.isChecked()
        self.rl_model_combo.setEnabled(enable_rl)
        if enable_rl:
            self.train_hint_label.setText("当前为强化学习模式，训练进度条将在任务开始后激活。")
        else:
            self.train_hint_label.setText("当前为 Baseline 模式，右侧日志会输出优化器求解过程。")
        self.refresh_overview()

    @Slot()
    def refresh_overview(self, *args):
        if not self.summary_value_labels:
            return

        self.summary_value_labels["grid_model"].setText(self.grid_model_combo.currentText().upper())
        self.summary_value_labels["solver"].setText(self.solver_combo.currentText().upper())

        mode_prefix = "训练" if self.tabs.currentIndex() == 3 else "评估"
        flow_mode = "两阶段" if self.rb_train_two_stage.isChecked() else "一阶段"
        if self.tabs.currentIndex() == 4:
            flow_mode = "两阶段" if self.rb_eval_two_stage.isChecked() else "一阶段"
        self.summary_value_labels["mode"].setText(f"{mode_prefix} / {flow_mode}")

        ev_source = "外部文件" if self.rb_ev_external.isChecked() else "随机生成"
        self.summary_value_labels["ev_source"].setText(ev_source)

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
            "ev_data_source": "external" if self.rb_ev_external.isChecked() else "random",
            "ev_params": {
                "capacity_kwh": self.ev_cap_spin.value(),
                "max_charge_kw": self.ev_charge_spin.value(),
                "max_discharge_kw": self.ev_discharge_spin.value(),
                "charge_efficiency": self.ev_eff_c_spin.value(),
                "discharge_efficiency": self.ev_eff_d_spin.value(),
            },
            "rl_common": {
                "learning_rate": self.lr_spin.value(),
                "gamma": self.gamma_spin.value(),
                "batch_size": self.batch_spin.value(),
            },
            "rl_specific": {
                "PPO": {"clip_range": self.ppo_clip_spin.value(), "ent_coef": self.ppo_ent_spin.value()},
                "SAC": {"tau": self.sac_tau_spin.value(), "ent_coef": self.sac_ent_spin.value()},
                "DDPG": {"tau": self.ddpg_tau_spin.value(), "action_noise": self.ddpg_noise_spin.value()},
                "TD3": {"policy_delay": self.td3_delay_spin.value(), "target_policy_noise": self.td3_noise_spin.value()},
            },
            "reward_weights": {
                "ev_kwh_shortage_penalty": self.rew_ev_spin.value(),
                "voltage_violation_penalty": self.rew_volt_spin.value(),
                "cost_penalty_factor": self.rew_cost_spin.value(),
                "opendss_failure_penalty": self.rew_fail_spin.value(),
            },
            "reward_mode": self.reward_mode_combo.currentData(),
            "station_operator": {
                "charge_service_price": self.ops_charge_price_spin.value(),
                "v2g_subsidy_price": self.ops_discharge_price_spin.value(),
                "include_grid_cost": self.chk_ops_include_grid_cost.isChecked(),
                "include_generation_cost": self.chk_ops_include_generation_cost.isChecked(),
                "include_ess_cost": self.chk_ops_include_ess_cost.isChecked(),
                "include_sop_loss_cost": self.chk_ops_include_sop_loss_cost.isChecked(),
                "include_penalty_cost": self.chk_ops_include_penalty_cost.isChecked(),
            },
        }
        try:
            with open(PATHS["gui_settings"], "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=4, ensure_ascii=False)
            self.update_log("配置已成功保存至 gui_settings.json")
        except Exception as e:
            self.update_log(f"错误: 保存配置失败: {e}")

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

        ev_source = settings.get("ev_data_source", "random")
        if ev_source == "external":
            self.rb_ev_external.setChecked(True)
        else:
            self.rb_ev_random.setChecked(True)

        ev_params = settings.get("ev_params", {})
        self.ev_cap_spin.setValue(ev_params.get("capacity_kwh", 70.0))
        self.ev_charge_spin.setValue(ev_params.get("max_charge_kw", 60.0))
        self.ev_discharge_spin.setValue(ev_params.get("max_discharge_kw", 25.0))
        self.ev_eff_c_spin.setValue(ev_params.get("charge_efficiency", 0.95))
        self.ev_eff_d_spin.setValue(ev_params.get("discharge_efficiency", 0.90))

        from config import RL_HYPERPARAMS, RL_ENV_CONFIG

        rl_common = settings.get("rl_common", RL_HYPERPARAMS["common"])
        self.lr_spin.setValue(rl_common.get("learning_rate", 0.0003))
        self.gamma_spin.setValue(rl_common.get("gamma", 0.99))
        self.batch_spin.setValue(rl_common.get("batch_size", 256))

        rl_spec = settings.get("rl_specific", RL_HYPERPARAMS["algo_specific"])
        self.ppo_clip_spin.setValue(rl_spec.get("PPO", {}).get("clip_range", 0.2))
        self.ppo_ent_spin.setValue(rl_spec.get("PPO", {}).get("ent_coef", 0.0))
        self.sac_tau_spin.setValue(rl_spec.get("SAC", {}).get("tau", 0.005))
        self.sac_ent_spin.setValue(rl_spec.get("SAC", {}).get("ent_coef", 0.1))
        self.ddpg_tau_spin.setValue(rl_spec.get("DDPG", {}).get("tau", 0.005))
        self.ddpg_noise_spin.setValue(rl_spec.get("DDPG", {}).get("action_noise", 0.1))
        self.td3_delay_spin.setValue(rl_spec.get("TD3", {}).get("policy_delay", 2))
        self.td3_noise_spin.setValue(rl_spec.get("TD3", {}).get("target_policy_noise", 0.2))

        rew_weights = settings.get("reward_weights", RL_ENV_CONFIG["reward_weights"])
        self.rew_ev_spin.setValue(rew_weights.get("ev_kwh_shortage_penalty", -100))
        self.rew_volt_spin.setValue(rew_weights.get("voltage_violation_penalty", -100))
        self.rew_cost_spin.setValue(rew_weights.get("cost_penalty_factor", 1))
        self.rew_fail_spin.setValue(
            settings.get("reward_weights", {}).get(
                "opendss_failure_penalty",
                RL_ENV_CONFIG["penalties"]["opendss_failure_penalty"],
            )
        )

        reward_mode = settings.get("reward_mode", "grid_operator")
        idx = self.reward_mode_combo.findData(reward_mode)
        if idx >= 0:
            self.reward_mode_combo.setCurrentIndex(idx)

        ops_cfg = settings.get("station_operator", {})
        self.ops_charge_price_spin.setValue(ops_cfg.get("charge_service_price", 1.20))
        self.ops_discharge_price_spin.setValue(ops_cfg.get("v2g_subsidy_price", 0.80))
        self.chk_ops_include_grid_cost.setChecked(ops_cfg.get("include_grid_cost", False))
        self.chk_ops_include_generation_cost.setChecked(ops_cfg.get("include_generation_cost", True))
        self.chk_ops_include_ess_cost.setChecked(ops_cfg.get("include_ess_cost", True))
        self.chk_ops_include_sop_loss_cost.setChecked(ops_cfg.get("include_sop_loss_cost", True))
        self.chk_ops_include_penalty_cost.setChecked(ops_cfg.get("include_penalty_cost", True))

        self.on_reward_mode_changed()
        self.update_log("已加载配置。")
        self.refresh_overview()

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
            self.task_label.setText("当前任务：Baseline 求解")
        else:
            task_type = "train_rl"
            task_params["rl_algo_name"] = self.rl_model_combo.currentText()
            self.task_label.setText(f"当前任务：训练 / 运行 {self.rl_model_combo.currentText()}")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

        task_params["mode"] = "two_stage" if self.rb_train_two_stage.isChecked() else "single_stage"
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
        self.task_label.setText(f"当前任务：评估 {len(selected_algos)} 个算法")
        task_params = {
            "selected_algos": selected_algos,
            "mode": "two_stage" if self.rb_eval_two_stage.isChecked() else "single_stage",
        }
        self.start_worker_thread("evaluate", task_params)

    def disconnect_stop_button(self):
        try:
            self.stop_button.clicked.disconnect()
        except Exception:
            pass

    def start_worker_thread(self, task_type, task_params):
        self.thread = QThread()
        self.worker = SimulationWorker(task_type, task_params)
        self.worker.moveToThread(self.thread)

        self.disconnect_stop_button()
        self.stop_button.clicked.connect(self.worker.request_stop)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.task_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.disconnect_stop_button)

        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.cleanup_references)
        self.worker.progress.connect(self.update_log)
        self.worker.error.connect(self.show_error)
        self.worker.progress_update.connect(self.update_progress_bar)

        self.thread.start()

    def open_ev_scenarios_file(self):
        try:
            filepath = PATHS["ev_scenarios_csv"]
            if not os.path.exists(filepath):
                self.update_log(f"错误: 找不到文件 {filepath}")
                self.update_log("提示: 请先在 data 文件夹中创建 my_ev_scenarios.csv 文件。")
                return
            if sys.platform == "win32":
                os.startfile(filepath)
            elif sys.platform == "darwin":
                subprocess.run(["open", filepath], check=False)
            else:
                subprocess.run(["xdg-open", filepath], check=False)
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
            self.status_label.setText("任务完成 / 已终止")
        else:
            self.status_label.setText("正在运行")

    def update_log(self, text):
        self.log_box.appendPlainText(text.strip())
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def task_finished(self):
        self.set_ui_for_task(is_running=False)

    def show_error(self, error_text):
        clean_text = error_text.strip()
        if "Warning:" in clean_text or "UserWarning:" in clean_text:
            self.log_box.appendPlainText(f"[警告] {clean_text}")
        else:
            self.log_box.appendPlainText(f"\n!!!!!! 发生错误 !!!!!!\n{clean_text}")
            self.status_label.setText("发生错误，已停止")
            self.set_ui_for_task(is_running=False)

    @Slot()
    def cleanup_references(self):
        self.update_log("...后台线程已清理，可以开始新任务。")
        self.thread = None
        self.worker = None
        self.task_label.setText("当前任务：无")

    def open_parameters_file(self):
        try:
            filepath = PATHS["grid_params_excel"]
            if not os.path.exists(filepath):
                self.update_log(f"错误: 找不到文件 {filepath}")
                return
            if sys.platform == "win32":
                os.startfile(filepath)
            elif sys.platform == "darwin":
                subprocess.run(["open", filepath], check=False)
            else:
                subprocess.run(["xdg-open", filepath], check=False)
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
                    if hasattr(module, "register_algorithm"):
                        info = module.register_algorithm()
                        model_class_registry[info["name"]] = info["class"]
                except Exception as e:
                    print(f"加载自定义算法插件 {plugin_file} 失败: {e}")
        return model_class_registry

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self.clear_layout(child_layout)

    @Slot()
    def update_eval_models_list(self):
        self.clear_layout(self.eval_algo_layout)
        self.eval_checkboxes.clear()

        chk_baseline = QCheckBox("Baseline（全局优化）")
        chk_baseline.setChecked(True)
        chk_baseline.toggled.connect(self.update_eval_selection_summary)
        self.eval_algo_layout.addWidget(chk_baseline)
        self.eval_checkboxes["Baseline"] = chk_baseline

        if self.rb_eval_two_stage.isChecked():
            suffix_candidates = ["_two_stage", "_2stage"]
        else:
            suffix_candidates = ["_single_stage", "_one_stage", "_1stage"]

        models_root = PATHS["models_dir"]
        if os.path.isdir(models_root):
            model_folders = [
                folder_name
                for folder_name in os.listdir(models_root)
                if os.path.isdir(os.path.join(models_root, folder_name))
            ]
        else:
            model_folders = []
            self.update_log(f"提示: models 目录不存在或不可访问：{models_root}")

        for folder_name in sorted(model_folders):
            if not any(folder_name.endswith(suf) for suf in suffix_candidates):
                continue

            model_path = os.path.join(models_root, folder_name, "best_model.zip")
            if not os.path.exists(model_path):
                continue

            display_name = folder_name.replace("best_", "").replace("_", " ").title().replace(" ", "_")
            chk = QCheckBox(f"{display_name}  ·  已检测到 best_model.zip")
            chk.setChecked(True)
            chk.toggled.connect(self.update_eval_selection_summary)
            self.eval_algo_layout.addWidget(chk)
            self.eval_checkboxes[display_name] = chk

        self.eval_algo_layout.addStretch()
        self.update_eval_selection_summary()
        self.refresh_overview()

    @Slot()
    def update_eval_selection_summary(self):
        selected_count = sum(1 for chk in self.eval_checkboxes.values() if chk.isChecked())
        total_count = len(self.eval_checkboxes)
        self.eval_selection_label.setText(f"已选择 {selected_count} / {total_count} 个算法")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
