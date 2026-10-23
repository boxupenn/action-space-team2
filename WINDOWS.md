# Action Space Kit on Windows 11 (native, no WSL)

本机安装记录与使用说明。上游只在 Ubuntu 上验证过；这里是在 Windows 11 原生环境下完成的安装，
所有东西都自包含在 `Z:\isaac`（ISSACLAB 盘），不改驱动、不改系统设置、不改 PATH。

安装日期 2026-10-16。GPU: RTX 5080 Laptop 16 GB，驱动 617.42。

> 训练 / 评估 / 导出 / 录像的完整命令、本机实测每轮耗时和跑基准时发现的问题，见 [`WINDOWS_TRAINING.md`](WINDOWS_TRAINING.md)（2026-10-17）。

## 目录布局

| 路径 | 内容 |
|---|---|
| `Z:\isaac\env_isaaclab\` | Python 3.11.17 venv（uv 创建）：isaacsim 5.1.0、torch 2.7.0+cu128、Isaac Lab v2.3.2 六个扩展、skrl、stable-baselines3、rsl-rl、本 kit（editable）、djitellopy、imageio-ffmpeg、opencv<5、pupil-apriltags、tello_calibration（editable，无依赖） |
| `Z:\isaac\IsaacLab\` | Isaac Lab v2.3.2 checkout（detached HEAD, `core.longpaths=true`） |
| `Z:\isaac\assets\tello.usd` + `configuration\` | 由 `simulation_infra/assets/tello/tello.urdf` 转换的无人机模型，两者必须放在一起 |
| `Z:\isaac\.python\` | uv 托管的 CPython 3.11 / 3.12 |
| `Z:\isaac\tools\uv.exe` | uv 本体（没加入 PATH） |
| `Z:\isaac\.cache\`, `Z:\isaac\tmp\` | 安装期间的 pip/uv 缓存与临时目录 |
| `Z:\isaac\logs\` | 每个安装阶段的日志 `stage1_isaacsim.log` … `stage5_verify.log` |
| `Z:\isaac\stage*.cmd` | 各安装阶段脚本，可重复运行（已完成的步骤会跳过或快速 no-op） |
| `Z:\isaac\activate_isaac.cmd` / `.ps1` | 日常使用的环境激活脚本（等价于 README 的 Step 2） |
| `Z:\isaac\kill_isaac.cmd` | 一键结束所有 Isaac Sim 进程（窗口关不掉时用） |
| `Z:\Action_Space\.venv\` | **实机侧**独立的 Python 3.12 venv（numpy 2.5.3 / opencv 5 / av 19 / pupil-apriltags / djitellopy / tello_calibration） |

Isaac Sim 自己的着色器缓存与日志在 `%LOCALAPPDATA%\ov\` 和 `%USERPROFILE%\.nvidia-omniverse\`（C 盘，几 GB，Kit 的硬编码位置）。

## 每次打开新终端（对应 README Step 2）

cmd:

```bat
call Z:\isaac\activate_isaac.cmd
```

PowerShell:

```powershell
. Z:\isaac\activate_isaac.ps1
```

脚本会清掉 conda 变量、**把所有 Anaconda 目录从 PATH 里剔除**（`D:\Anaconda\Library\bin` 里的 hdf5/zlib/ssl 等 DLL 会被 Kit 按名字误加载）、把 venv 放到 PATH 最前、设置 `OMNI_KIT_ACCEPT_EULA=YES`、
`TELLO_USD_PATH=Z:/isaac/assets/tello.usd`、`PYTHONUTF8=1`，并 cd 到 `Z:\Action_Space\simulation_infra`。

不要在 conda 环境里运行；`isaaclab.bat` 会优先用 `CONDA_PREFIX` 的 Python。

## 常用命令（在 simulation_infra 目录、已激活）

```bat
python -u scripts\check_approach_env.py --headless --test all > outputs\check_approach.log 2>&1
```

```bat
python -u scripts\sb3\play.py --task AS-Tello-Approach-v0 --headless --num_envs 64 --episodes 256 --checkpoint policies\approach_it5_room_fixed\model.zip env.spawn_azimuth_range=0.0 env.rew_scale_separation=-20.0 env.rew_violation=-100.0 > outputs\play_policy.log 2>&1
```

```bat
python -u scripts\record_approach.py --pilot policy --checkpoint policies\approach_it5_room_fixed\model.zip --spawn_azimuth_range 0.0 --set rew_scale_separation=-20.0 rew_violation=-100.0 --episodes 2 --out outputs\videos\approach_policy.mp4 --frame_dir outputs\frames > outputs\record.log 2>&1
```

```bat
python -m action_space_kit.bridge.tello_bridge --dry-run --duration 3
```

带窗口观看：去掉 `--headless`、加 `--num_envs 4`，**并且必须加 `--kit_args=--/app/vulkan=false`**（原因见下一节）：

```bat
python -u scripts\sb3\play.py --task AS-Tello-Approach-v0 --num_envs 4 --real-time --kit_args=--/app/vulkan=false --checkpoint policies\approach_it5_room_fixed\model.zip env.spawn_azimuth_range=0.0 env.rew_scale_separation=-20.0 env.rew_violation=-100.0 env.viewer.origin_type=env env.viewer.env_index=0 env.viewer.eye=[4.5,-14.0,7.0] env.viewer.lookat=[4.5,0.0,0.8]
```

停止用 Ctrl+Break（Windows 上 Ctrl+C 对 Kit 不可靠）。点窗口的关闭按钮有时没反应（Kit 在策略循环里不处理关闭事件），这时运行 `Z:\isaac\kill_isaac.cmd`：它只结束从 `Z:\isaac` 启动的 python 进程，不影响其它 Python 任务。

后面四个 `env.viewer.*` 不是可选项：这个任务的 `env_spacing` 是 28 m，4 个环境摆在 (±14, ±14)，而默认相机从 (7.5, 7.5, 7.5) 盯着世界原点，那里是四个环境之间的空地，窗口里只会看到地面网格，像是“什么都没渲染”。把相机绑到 0 号环境（`origin_type=env`）、从 -y 侧看 +x 方向的进场走廊（`spawn_azimuth_range=0.0` 时无人机固定在 +x 侧 9 m 处出生），就能看到四架无人机、闸环、禁入球和着陆盘。只开 `--num_envs 1` 时 0 号环境就在原点，默认相机也能看到。

## 窗口模式崩溃与解决办法（驱动分支问题）

现象：任何带窗口的脚本（包括 Isaac Lab 自带的 `scripts\tutorials\00_sim\create_empty.py`）在窗口弹出几秒后崩溃，
回溯在 `rtx.scenedb.plugin.dll!carbOnPluginStartup+0x252db`，无头模式、录像、训练全部正常。

根因：Isaac Sim 5.1.0 在 Windows 上验证过的最新驱动是 580.88；595 及更新的分支在 Blackwell（RTX 50 系）上的 **Vulkan** 渲染路径会崩，
无头模式走 D3D12 所以不受影响。本机驱动 617.42。NVIDIA 版主在论坛确认为已知问题，官方建议是降级驱动：
- https://forums.developer.nvidia.com/t/isaac-sim-5-1-gui-crash-access-violation-on-rtx-5070-ti-blackwell-fixed-by-driver-downgrade-to-591-74/365335
- https://forums.developer.nvidia.com/t/isaac-sim-5-1-crashes-on-startup-with-rtx-5060-ti-blackwell-sm-120-rtx-scenedb-plugin-crash/366252

本机采用的不降驱动的办法：Isaac Lab 的 `apps\isaaclab.python.kit` 写死了 `vulkan = true`，用 `--kit_args=--/app/vulkan=false`
把窗口模式切回 D3D12 后窗口稳定（2026-10-17 验证，`create_empty.py` 与 `sb3/play.py` 均可）。
写法注意：要用 `--kit_args=值` 这种带等号的形式，PowerShell 会把不含空格的引号参数的引号去掉。
`--/renderer/multiGpu/enabled=false --/renderer/activeGpu=0` 等多 GPU 参数对这个崩溃无效（已测）。
如果以后要用 Vulkan 专属功能，就只能把驱动降到 580.88（NVIDIA 验证版本）或 591.74（论坛在 Blackwell 笔记本上验证可用）。

PowerShell 下不要用 `>` 重定向 Python 输出（5.1 会写成 UTF-16），用上面 cmd 的写法或 `cmd /c "python ... > log 2>&1"`。

## 显存与隔壁任务

本机 16 GB 显存里另一个 Anaconda 任务常驻约 9.2 GB。`stage5_verify.cmd` 在启动 Isaac Sim 前会检查
空闲显存 ≥ 4 GB，不够就退出。无头 + 64 个 env 的评分大约占 3 GB 左右；训练（512 env）在
显存被占一半时不要跑。所有安装/验证进程都以“低于正常”优先级启动。

## 对上游代码的 Windows 补丁（git diff 可见）

| 文件 | 改动 | 原因 |
|---|---|---|
| `simulation_infra/scripts/record_approach.py` | `--out`/`--frame_dir` 默认改为 `outputs/...`；DejaVu 字体加回退（Arial Bold → Pillow 内置）；复制 PNG 时对 `PermissionError` 重试 | `/mnt/data` 与 `/usr/share/fonts` 在 Windows 不存在；Kit 的 PNG 写入是异步的 |
| `simulation_infra/scripts/record_clip.py` | `--out`/`--frame_dir` 默认改为 `outputs/...` | 同上 |
| `source/action_space_kit/action_space_kit/tags/spawn.py` | 纹理路径用 `as_posix()` | USD 资产路径用正斜杠 |
| `source/action_space_kit/action_space_kit/__init__.py` | 任务/UI 注册包在 `try` 里，只放过 `pxr/omni/carb/usdrt/isaacsim` 缺失 | 不启动 Kit 时（例如 `python -m action_space_kit.bridge.tello_bridge`）`pxr` 不可导入，桥接等纯 Python 部分必须仍可导入 |
| `scripts/record_approach.py`, `record_clip.py`, `record_room.py`, `check_room_render.py`, `render_room_overview.py`, `stream_room.py`, `sb3/play.py`, `skrl/play.py` | 在 `from isaaclab.app import AppLauncher` 之前加 `import h5py` | Kit 的 `isaacsim.sensors.rtx` 扩展自带 HDF5 1.14.6 的 `hdf5.dll`，与 venv 里 h5py 的 `hdf5.dll` 同名；谁先加载谁赢。h5py 3.15+ 自带 HDF5 2.0.0（删掉了 `H5Tdecode` 等旧接口），两边版本不同时必有一方加载失败：h5py 失败则 `isaaclab_tasks` 导入报 `DLL load failed while importing _errors`；Kit 失败则弹出 `python.exe - 无法找到入口 H5Tdecode` 对话框。最终做法：venv 里把 h5py 固定为 **3.14.0**（HDF5 1.14.6，与 Kit 一致，stage3 脚本已固定），预导入保留作保险 |

另有一个**环境级**垫片，不在仓库里：`Z:/isaac/env_isaaclab/Lib/site-packages/numpy2_pickle_compat.py`（由同目录的 `.pth` 在解释器启动时加载）。
出厂权重 `model_vecnormalize.pkl` / `model.zip` 是用 numpy 2 pickle 的，而 Isaac Lab 锁定 numpy 1.26；垫片把 `numpy._core.*` 映射回 `numpy.core.*`，
并让随机数生成器接受 numpy 2 的 pickle 格式。不装它 `sb3/play.py` 会报 `No module named 'numpy._core'`。
numpy 1.26 自己的 pickle 往返经测试不受影响。

`setup.sh`、`convert.sh`、`run_iter.sh`、`dashboard/*.sh` 是 bash 脚本，Windows 下由 `Z:\isaac\stage*.cmd` 替代；
它们在工作区里是 CRLF（`core.autocrlf`），若要在 Git Bash 里跑需先 `sed -i 's/\r$//'`。

## 实机侧（drone_deployment_infra）在 Windows 上的限制

依赖已经装进 `Z:\Action_Space\.venv`（Python 3.12），纯计算部分可用：
`core/positional.py`、`positional_kalman_filter.py`（`python core\positional_kalman_filter.py` 自测）、
`extract_track.py`、`build_flight_viz.py`、标定与建图的数学部分。

但连接真机的代码是 Linux 网络栈专用，Windows 上会直接报错，需要改写后才能飞：

- `nmcli` / `ip -json addr`（`cal_and_map/tello_calibration/utils/net.py`、`tello_map.py`、`core/tello_dongle_setup.py`、`core/tello_link.py`）→ 需换成 `netsh wlan` + `psutil.net_if_addrs()`
- `SO_REUSEPORT` / `SO_BINDTODEVICE` / `fcntl.ioctl`（`core/tello_dual_video.py`，多 dongle 绑定）→ Windows 没有对应能力，单机只能 `bind((adapter_ip, port))`
- `os.killpg` / `setsid` / `pkill`（`core/tello_link.py`、`drone_deployment_tests/tello_range_test.py`）
- UDP 收包在 Windows 上会抛 `ConnectionResetError`（`comms/tello_link.py`），需要在 `except` 里加上

实机侧代码在 WSL2 的 Ubuntu 里跑也不行（WSL2 看不到宿主机 Wi-Fi 适配器）。真要在这台机器上飞，
要么改写上述几处，要么用一台 Linux 机器做地面站。

## 验证结果（2026-10-16，无头，隔壁另有一个 GPU 任务占 7-9 GB 显存）

| 检查 | 结果 | 参考值（CLAUDE.md，Linux） |
|---|---|---|
| `python -c "import isaaclab, isaacsim, torch"` | `isaaclab ok, cuda: True`，torch 2.7.0+cu128，算力 12.0 | 同 |
| `scripts/list_envs.py` | 9 个 `AS-*` 任务全部注册 | 含 `AS-Tello-Approach-v0` |
| `check_approach_env.py --headless --test all` | 22/22 `[ok]`（12 geometry + 6 controller + 4 baseline） | 同 |
| 基线 zero / scripted | 0% / 100% 成功，team return -803.7 / +1542.7 | ~-800 / ~+1540 |
| `sb3/play.py` 出厂策略，64 env x 256 episodes | near_base_frac=0.897 success_rate=0.864 collision_rate=0.136 gates_passed=4.000 illegal_entry_rate=0.000 crash_rate=0.000 | 完全相同 |
| `tello_bridge --dry-run --duration 3` | 打印 `[dry-run] connect` 与 `rc 0 30 0 0` 序列 | 同 |
| `core/positional_kalman_filter.py`（两个 venv） | `SELFTEST PASSED` | 同 |
| `scripts/fit_tello_motion.py --selftest` | `SELFTEST PASSED` | 同 |
| `skrl/train.py --task AS-Tello-Hover-v0 --num_envs 64 --max_iterations 3`（skrl 2.1.0） | 正常训练 18 s，写出 `logs/skrl/tello_hover/<run>/checkpoints/agent_14.pt` | — |
| `record_approach.py --pilot scripted --episodes 1` | `episode 1: SUCCESS -- gates 4/4, landed 4/4`，248 帧，`outputs/videos/approach_scripted.mp4`（538 KB），画面含网格地面、红色禁入球、四色闸环与着陆盘 | 同 |
| Isaac Sim 无头启动到任务可用 | 约 25-50 s（含首次着色器编译） | — |
| 显存占用（无头，64 env） | Isaac Sim 进程约 2.4 GB | — |

日志：`Z:\isaac\logs\stage5_verify.log`、`stage5b_play_bridge.log`、仓库内 `simulation_infra/outputs/*.log`。
