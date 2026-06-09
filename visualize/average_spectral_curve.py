import spectral as sp
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import matplotlib
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 强制使用 TkAgg 后端（适配Tkinter）
matplotlib.use('TkAgg')

# -------------------------- 全局配置 --------------------------
# 可视化配置（中文显示+负号正常显示）
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.autolayout"] = True

# 定义核心配置
VALID_FRUIT_TYPES = ["jjk", "bd", "ys", "jk"]  # 支持的水果种类
VALID_TIME_TAGS = [0, 1, 6, 12, 18, 24, 30, 36]  # 有效时间标签
TIME_COLOR_MAP = {  # 时间-颜色映射
    0: "#FF6B6B", 1: "#4ECDC4", 6: "#45B7D1", 12: "#96CEB4",
    18: "#FECA57", 24: "#FF9FF3", 30: "#54A0FF", 36: "#5F27CD"
}
INDIVIDUAL_COLORS = plt.cm.tab20(np.linspace(0, 1, 40))  # 个体-颜色映射（40种）
DOWNSAMPLE_RATE = 2  # 波段降采样率（提升绘制效率）

# 全局数据存储（批量读取后缓存）
global_data_dict = {}  # 格式：{(水果种类, 时间标签, 个体ID): (band_means, wavelengths, filename)}


# -------------------------- 工具函数 --------------------------
def parse_filename_tags(filename):
    """从文件名解析水果种类、时间标签、个体ID（格式：水果种类-时间-个体-f.hdr）"""
    try:
        core_name = Path(filename).stem  # 例："bd-0-1-f"
        parts = core_name.split("-")  # 例：["bd", "0", "1", "f"]
        fruit_type = parts[0]
        time_tag = int(parts[1])
        individual_id = parts[2]

        # 验证有效性
        if fruit_type not in VALID_FRUIT_TYPES:
            raise ValueError(f"水果种类 {fruit_type} 不在支持列表 {VALID_FRUIT_TYPES} 中")
        if time_tag not in VALID_TIME_TAGS:
            raise ValueError(f"时间标签 {time_tag} 不在有效列表 {VALID_TIME_TAGS} 中")
        return fruit_type, time_tag, individual_id
    except Exception as e:
        raise ValueError(f"文件名 {filename} 格式错误：{str(e)}（正确格式：水果种类-时间-个体-f.hdr）")

# 在这里修改正反的识别
def batch_read_hyperspectral_data(folder_path):
    """批量读取文件夹下所有高光谱数据，按（水果种类, 时间, 个体）缓存"""
    global global_data_dict
    global_data_dict.clear()
    folder = Path(folder_path)

    if not folder.exists():
        messagebox.showerror("错误", f"文件夹 {folder_path} 不存在！")
        return False

    print(f"\n📂 开始批量读取文件夹：{folder_path}")
    for hdr_file in folder.glob("*-*-*-z.hdr"):  # 匹配格式：水果种类-时间-个体-f.hdr
        filename = hdr_file.name
        data_file = folder / (hdr_file.stem + ".float")
        if not data_file.exists():
            print(f"⚠️  跳过 {filename}：缺少对应的 .float 文件")
            continue

        try:
            # 解析标签
            fruit_type, time_tag, individual_id = parse_filename_tags(filename)
            print(f"📌 读取成功：{filename} → 种类：{fruit_type}，时间：{time_tag}，个体：{individual_id}")

            # 读取数据并计算波段均值
            img = sp.envi.open(str(hdr_file), image=str(data_file))
            data = img.load()
            if data.shape[0] == img.nbands:
                data = np.transpose(data, (1, 2, 0))
            band_means = np.nanmean(data, axis=(0, 1))

            # 波段降采样（减少绘制点数）
            wavelengths = None
            if "wavelength" in img.metadata:
                wavelengths = np.array(img.metadata["wavelength"], dtype=float)
                wavelengths = wavelengths[~np.isnan(wavelengths)][::DOWNSAMPLE_RATE]
            band_means = band_means[::DOWNSAMPLE_RATE]

            # 缓存数据（键：(水果种类, 时间标签, 个体ID)）
            global_data_dict[(fruit_type, time_tag, individual_id)] = (band_means, wavelengths, filename)
        except Exception as e:
            print(f"⚠️  读取 {filename} 失败：{str(e)}")
            continue

    if len(global_data_dict) == 0:
        messagebox.showwarning("警告", "未读取到有效高光谱数据！")
        return False
    else:
        messagebox.showinfo("成功", f"批量读取完成，共加载 {len(global_data_dict)} 个样本")
        return True


def add_hover_interaction(ax, line, info_dict):
    """保留原鼠标悬停交互功能"""
    annot = ax.annotate(
        "", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", alpha=0.6)
    )
    annot.set_visible(False)

    def update_annot(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = line.contains(event)
            if cont:
                idx = ind["ind"][0]
                x = line.get_xdata()[idx]
                y = line.get_ydata()[idx]
                text = (
                    f"文件名：{info_dict['filename']}\n"
                    f"{info_dict['x_label']}：{x:.1f}\n"
                    f"反射率/亮度值：{y:.4f}\n"
                    f"水果种类：{info_dict['fruit_type']}\n"
                    f"时间标签：{info_dict['time_tag']}\n"
                    f"个体ID：{info_dict['individual_id']}"
                )
                annot.xy = (x, y)
                annot.set_text(text)
                annot.set_visible(True)
                ax.figure.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    ax.figure.canvas.draw_idle()

    ax.figure.canvas.mpl_connect("motion_notify_event", update_annot)
    return annot


def save_current_figure(fig):
    """保存当前展示的图表"""
    file_types = [
        ("PNG 图片", "*.png"),
        ("JPG 图片", "*.jpg"),
        ("SVG 矢量图", "*.svg"),
        ("所有文件", "*.*")
    ]
    file_path = filedialog.asksaveasfilename(
        title="保存图片",
        defaultextension=".png",
        filetypes=file_types,
        initialfile="hyperspectral_plot"
    )
    if file_path:
        try:
            fig.savefig(file_path, dpi=300, bbox_inches="tight")
            messagebox.showinfo("成功", f"图片已保存至：{file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败：{str(e)}")


# -------------------------- UI界面类 --------------------------
class HyperspectralPlotUI:
    def __init__(self, root):
        self.root = root
        self.root.title("高光谱数据可视化工具")
        self.root.geometry("1500x900")  # 窗口大小
        self.current_fig = None  # 当前展示的图表
        self.folder_path = ""  # 数据文件夹路径

        # 初始化UI组件
        self.init_ui()

    def init_ui(self):
        # 1. 顶部控制区（文件夹选择+批量读取）
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X, side=tk.TOP)

        # 文件夹选择
        ttk.Label(control_frame, text="数据文件夹：").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.folder_var = tk.StringVar()
        self.folder_entry = ttk.Entry(control_frame, textvariable=self.folder_var, width=60)
        self.folder_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(control_frame, text="浏览", command=self.select_folder).grid(row=0, column=2, padx=5, pady=5)

        # 批量读取按钮
        self.read_btn = ttk.Button(control_frame, text="批量读取数据", command=self.batch_read_data, state=tk.DISABLED)
        self.read_btn.grid(row=0, column=3, padx=10, pady=5)

        # 2. 模式选择区
        mode_frame = ttk.LabelFrame(self.root, text="展示模式", padding="10")
        mode_frame.pack(fill=tk.X, side=tk.TOP, padx=10, pady=10)

        self.mode_var = tk.StringVar(value="mode1")  # 默认模式1
        ttk.Radiobutton(mode_frame, text="模式1：同时间不同个体", variable=self.mode_var, value="mode1",
                        command=self.switch_mode).grid(row=0, column=0, padx=20, pady=5)
        ttk.Radiobutton(mode_frame, text="模式2：同个体不同时间", variable=self.mode_var, value="mode2",
                        command=self.switch_mode).grid(row=0, column=1, padx=20, pady=5)

        # 3. 筛选条件区（根据模式动态切换）
        self.filter_frame = ttk.LabelFrame(self.root, text="筛选条件", padding="10")
        self.filter_frame.pack(fill=tk.X, side=tk.TOP, padx=10, pady=5)
        self.init_filter_widgets()

        # 4. 绘图按钮
        self.plot_btn = ttk.Button(self.root, text="绘制图表", command=self.plot_data, state=tk.DISABLED)
        self.plot_btn.pack(side=tk.TOP, padx=10, pady=10)

        # 5. 图片保存按钮
        self.save_btn = ttk.Button(self.root, text="保存当前图表", command=self.save_plot, state=tk.DISABLED)
        self.save_btn.pack(side=tk.TOP, padx=10, pady=5)

        # 6. 图表展示区
        self.plot_frame = ttk.Frame(self.root, padding="10")
        self.plot_frame.pack(fill=tk.BOTH, expand=True, side=tk.TOP)

    def init_filter_widgets(self):
        """初始化筛选条件组件（根据当前模式）"""
        # 清空现有组件
        for widget in self.filter_frame.winfo_children():
            widget.destroy()

        mode = self.mode_var.get()
        if mode == "mode1":
            # 模式1：同时间不同个体 → 筛选：水果种类（可选）+ 时间标签
            ttk.Label(self.filter_frame, text="水果种类（可选）：").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
            self.fruit_var_mode1 = tk.StringVar(value="all")
            fruit_options = ["all"] + VALID_FRUIT_TYPES  # all表示全部种类
            self.fruit_combobox_mode1 = ttk.Combobox(self.filter_frame, textvariable=self.fruit_var_mode1,
                                                     values=fruit_options, state="readonly")
            self.fruit_combobox_mode1.grid(row=0, column=1, padx=5, pady=5)

            ttk.Label(self.filter_frame, text="选择时间标签：").grid(row=0, column=2, padx=20, pady=5, sticky=tk.W)
            self.time_var_mode1 = tk.IntVar(value=VALID_TIME_TAGS[0])
            self.time_combobox_mode1 = ttk.Combobox(self.filter_frame, textvariable=self.time_var_mode1,
                                                    values=VALID_TIME_TAGS, state="readonly")
            self.time_combobox_mode1.grid(row=0, column=3, padx=5, pady=5)

        else:
            # 模式2：同个体不同时间 → 筛选：水果种类 + 个体ID
            ttk.Label(self.filter_frame, text="选择水果种类：").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
            self.fruit_var_mode2 = tk.StringVar(value=VALID_FRUIT_TYPES[0])
            self.fruit_combobox_mode2 = ttk.Combobox(self.filter_frame, textvariable=self.fruit_var_mode2,
                                                     values=VALID_FRUIT_TYPES, state="readonly")
            self.fruit_combobox_mode2.grid(row=0, column=1, padx=5, pady=5)
            self.fruit_combobox_mode2.bind("<<ComboboxSelected>>", self.update_individual_combobox)

            ttk.Label(self.filter_frame, text="选择个体ID：").grid(row=0, column=2, padx=20, pady=5, sticky=tk.W)
            self.individual_var_mode2 = tk.StringVar()
            self.individual_combobox_mode2 = ttk.Combobox(self.filter_frame, textvariable=self.individual_var_mode2,
                                                          state="readonly")
            self.individual_combobox_mode2.grid(row=0, column=3, padx=5, pady=5)

            # 初始化个体下拉框（默认第一个水果种类）
            self.update_individual_combobox()

    def switch_mode(self):
        """切换展示模式时更新筛选条件组件"""
        self.init_filter_widgets()

    def select_folder(self):
        """选择数据文件夹"""
        folder = filedialog.askdirectory(title="选择高光谱数据文件夹")
        if folder:
            self.folder_var.set(folder)
            self.folder_path = folder
            self.read_btn.config(state=tk.NORMAL)  # 启用批量读取按钮

    def batch_read_data(self):
        """批量读取数据"""
        success = batch_read_hyperspectral_data(self.folder_path)
        if success:
            self.plot_btn.config(state=tk.NORMAL)  # 启用绘图按钮
            self.init_filter_widgets()  # 重新初始化筛选组件（加载数据后）

    def update_individual_combobox(self, event=None):
        """根据选择的水果种类，更新个体ID下拉框"""
        fruit_type = self.fruit_var_mode2.get()
        # 提取该水果种类下的所有个体ID
        individuals = list({
            indiv_id for (ft, t, indiv_id) in global_data_dict.keys()
            if ft == fruit_type
        })
        individuals.sort()  # 排序
        if individuals:
            self.individual_combobox_mode2.config(values=individuals)
            self.individual_var_mode2.set(individuals[0])
        else:
            self.individual_combobox_mode2.config(values=[])
            self.individual_var_mode2.set("")
            messagebox.showwarning("警告", f"该水果种类 {fruit_type} 无有效个体数据！")

    def plot_data(self):
        """根据选择的模式和筛选条件绘制图表"""
        # 清空现有图表
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        mode = self.mode_var.get()
        if mode == "mode1":
            self.plot_mode1()
        else:
            self.plot_mode2()

    def plot_mode1(self):
        """模式1：同时间不同个体（选择时间标签，展示该时间下所有个体）"""
        fruit_type = self.fruit_var_mode1.get()
        time_tag = self.time_var_mode1.get()

        # 筛选数据：指定时间，可选水果种类
        plot_data = []
        for (ft, t, indiv_id), (band_means, wavelengths, filename) in global_data_dict.items():
            if t == time_tag and (fruit_type == "all" or ft == fruit_type):
                plot_data.append((ft, indiv_id, band_means, wavelengths, filename))

        if not plot_data:
            messagebox.showwarning("警告", f"未找到 时间标签={time_tag} 的有效数据！")
            return

        # 排序（按水果种类→个体ID）
        plot_data.sort(key=lambda x: (x[0], x[1]))

        # 创建图表
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        self.current_fig = fig  # 缓存当前图表

        # 图表标题
        fruit_title = "所有种类" if fruit_type == "all" else f"{fruit_type}类"
        ax.set_title(
            f"模式1：{fruit_title} - 时间标签 {time_tag} 不同个体对比（共 {len(plot_data)} 个个体）",
            fontsize=13, fontweight="bold"
        )
        ax.set_xlabel("波长 (nm)" if plot_data[0][3] is not None else "波段索引", fontsize=11)
        ax.set_ylabel("反射率/亮度值", fontsize=11)
        ax.grid(True, alpha=0.3, linestyle="--", color="gray")

        # 绘制曲线（个体颜色固定）
        all_individuals = list({indiv_id for (ft, t, indiv_id) in global_data_dict.keys()})
        individual_color_map = {indiv: INDIVIDUAL_COLORS[i % len(INDIVIDUAL_COLORS)]
                                for i, indiv in enumerate(all_individuals)}

        for ft, indiv_id, band_means, wavelengths, filename in plot_data:
            x_data = wavelengths if wavelengths is not None else np.arange(1, len(band_means) + 1)
            color = individual_color_map[indiv_id]
            label = f"{ft}-个体{indiv_id}"  # 图例显示：种类-个体ID

            line = ax.plot(
                x_data, band_means, color=color, linewidth=2.0, alpha=0.9,
                label=label, antialiased=False
            )[0]

            # 添加鼠标交互
            add_hover_interaction(ax, line, {
                "filename": filename,
                "x_label": "波长 (nm)" if wavelengths is not None else "波段索引",
                "fruit_type": ft,
                "time_tag": time_tag,
                "individual_id": indiv_id
            })

        # 图例优化（多列显示在下方）
        n_cols = min(5, (len(plot_data) + 1) // 2)
        ax.legend(
            title="个体信息（种类-个体ID）", bbox_to_anchor=(0.5, -0.15), loc="upper center",
            fontsize=9, ncol=n_cols, frameon=True, fancybox=True, shadow=True
        )

        # 调整布局
        plt.subplots_adjust(bottom=0.2)

        # 嵌入Tkinter窗口
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 启用保存按钮
        self.save_btn.config(state=tk.NORMAL)

    def plot_mode2(self):
        """模式2：同个体不同时间（选择水果种类+个体，展示该个体所有时间）"""
        fruit_type = self.fruit_var_mode2.get()
        individual_id = self.individual_var_mode2.get()

        if not individual_id:
            messagebox.showwarning("警告", "请选择有效个体ID！")
            return

        # 筛选数据：指定水果种类+个体
        plot_data = []
        for (ft, t, indiv_id), (band_means, wavelengths, filename) in global_data_dict.items():
            if ft == fruit_type and indiv_id == individual_id:
                plot_data.append((t, band_means, wavelengths, filename))

        if not plot_data:
            messagebox.showwarning("警告", f"未找到 {fruit_type}类-个体{individual_id} 的有效数据！")
            return

        # 按时间排序
        plot_data.sort(key=lambda x: x[0])

        # 创建图表
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        self.current_fig = fig  # 缓存当前图表

        # 图表标题
        ax.set_title(
            f"模式2：{fruit_type}类-个体{individual_id} 不同时间对比（共 {len(plot_data)} 个时间点）",
            fontsize=13, fontweight="bold"
        )
        ax.set_xlabel("波长 (nm)" if plot_data[0][2] is not None else "波段索引", fontsize=11)
        ax.set_ylabel("反射率/亮度值", fontsize=11)
        ax.grid(True, alpha=0.3, linestyle="--", color="gray")

        # 绘制曲线（时间颜色固定）
        for time_tag, band_means, wavelengths, filename in plot_data:
            x_data = wavelengths if wavelengths is not None else np.arange(1, len(band_means) + 1)
            color = TIME_COLOR_MAP[time_tag]

            line = ax.plot(
                x_data, band_means, color=color, linewidth=2.0, alpha=0.9,
                label=f"时间 {time_tag}", antialiased=False
            )[0]

            # 添加鼠标交互
            add_hover_interaction(ax, line, {
                "filename": filename,
                "x_label": "波长 (nm)" if wavelengths is not None else "波段索引",
                "fruit_type": fruit_type,
                "time_tag": time_tag,
                "individual_id": individual_id
            })

        # 图例优化（多列显示在下方）
        n_cols = min(4, (len(plot_data) + 1) // 2)
        ax.legend(
            title="时间标签", bbox_to_anchor=(0.5, -0.15), loc="upper center",
            fontsize=9, ncol=n_cols, frameon=True, fancybox=True, shadow=True
        )

        # 调整布局
        plt.subplots_adjust(bottom=0.2)

        # 嵌入Tkinter窗口
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 启用保存按钮
        self.save_btn.config(state=tk.NORMAL)

    def save_plot(self):
        """保存当前图表"""
        if self.current_fig is None:
            messagebox.showwarning("警告", "暂无图表可保存！")
            return
        save_current_figure(self.current_fig)


# -------------------------- 程序入口 --------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = HyperspectralPlotUI(root)
    root.mainloop()