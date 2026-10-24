# Windows 训练 / 评估 / 导出 / 录像 命令手册

本机（RTX 5080 Laptop 16 GB，驱动 617.42，Isaac Sim 5.1.0 + Isaac Lab v2.3.2，Windows 原生）上，
每个任务家族从训练到出视频的完整命令，全部实际跑通过（2026-10-17）。安装与环境细节见 `WINDOWS.md`。
所有命令都在 **已激活的 shell、`Z:\Action_Space\simulation_infra` 目录** 下运行。

## 0. 每个新终端

cmd：

```bat
call Z:\isaac\activate_isaac.cmd
```

PowerShell：

```powershell
. Z:\isaac\activate_isaac.ps1
```

## 1. 通用规则（Windows 特有的都在这里）

| 规则 | 原因 |
|---|---|
| Python 一律 `python -u ...`，并把输出重定向到 `outputs\xxx.log` | Isaac Sim 退出时会丢掉缓冲区里的输出，不加 `-u` 看不到最后几行 |
| 训练、play、evaluate 加 `--headless` | 窗口模式在本机需要额外参数（见 `WINDOWS.md`） |
| **凡是 `--video`（skrl/sb3 的 play、train，`evaluate.py --video`）必须再加 `--kit_args=--/app/vulkan=false`** | 这些脚本会加载 `isaaclab.python.headless.rendering.kit`，它写死 `app.vulkan = true`；本机 Blackwell + 617 驱动下 Vulkan 启动 9 秒即崩溃（access violation）。改用 D3D12 就正常 |
| `record_approach.py` / `record_room.py` / `record_clip.py` 不需要上面的参数 | 它们走 livestream 的 viewport 截帧，本来就是 D3D12 |
| skrl 的 `play.py` 无头时加 `--steps N`，否则永不退出 | 上游脚本只在 `--video` 时才跳出循环（本机补丁：结束时打印平均回报） |
| 并行跑别的 GPU 任务时，用低优先级启动 | `bench.ps1` 自动做了；手动跑可以用 `start /belownormal` |
| 停止：Ctrl+Break；窗口关不掉就 `Z:\isaac\kill_isaac.cmd` | Kit 在策略循环里不处理 Ctrl+C |
| 看曲线：`tensorboard --logdir logs` | 激活后 venv 里自带 |

续训：两个训练脚本都支持 `--checkpoint <文件>` 从已有权重继续训练（这是"提前预训练、拿到题后微调"的关键）。
SB3 的 checkpoint 是 `model.zip`（旁边的 `model_vecnormalize.pkl` 会自动加载），skrl 的是 `agent_<N>.pt`。

## 2. Hover / Waypoint（skrl PPO，单机）

任务名：`AS-Tello-Hover-v0`、`AS-Tello-Waypoint-v0`。产物在 `logs\skrl\tello_hover\<run>\`（waypoint 同理）。

```bat
python -u scripts\skrl\train.py --task AS-Tello-Hover-v0 --headless --num_envs 1024 --max_iterations 200 > outputs\train_hover.log 2>&1
```

```bat
python -u scripts\skrl\train.py --task AS-Tello-Hover-v0 --headless --num_envs 1024 --max_iterations 200 --checkpoint logs\skrl\tello_hover\<run>\checkpoints\agent_4800.pt > outputs\train_hover_resume.log 2>&1
```

play（400 步，结束时打印 `[play] N episodes finished: mean return ...`）：

```bat
python -u scripts\skrl\play.py --task AS-Tello-Hover-v0 --headless --num_envs 16 --checkpoint logs\skrl\tello_hover\<run>\checkpoints\agent_4800.pt --steps 400 > outputs\play_hover.log 2>&1
```

导出 TorchScript（观测标准化 + 确定性策略 + 裁剪到 [-1,1]；不需要 Kit）：

```bat
python -u scripts\export_skrl_policy.py --checkpoint tello_hover --out outputs\exports\hover.pt
```

录像（写到 `logs\skrl\tello_hover\<run>\videos\play\rl-video-step-0.mp4`，1280x720）：

```bat
python -u scripts\skrl\play.py --task AS-Tello-Hover-v0 --headless --num_envs 4 --checkpoint logs\skrl\tello_hover\<run>\checkpoints\agent_4800.pt --video --video_length 200 --kit_args=--/app/vulkan=false > outputs\video_hover.log 2>&1
```

skrl 的 `--max_iterations` 乘以 `rollouts`（hover/waypoint 24，defend 16）才是 timesteps；checkpoint 文件名里的数字是 timesteps。

## 3. Defend（skrl IPPO / MAPPO，hackathon 场景）

任务名：`AS-Defend-v0`。产物在 `logs\skrl\tello_defend\<run>\`。

```bat
python -u scripts\skrl\train.py --task AS-Defend-v0 --algorithm IPPO --headless --num_envs 512 --max_iterations 500 > outputs\train_defend_ippo.log 2>&1
```

README 建议的课程第一阶段（更大的捕获半径、更近更慢的攻击者），之后用 `--checkpoint` 从它续训并把参数逐步调回默认：

```bat
python -u scripts\skrl\train.py --task AS-Defend-v0 --algorithm IPPO --headless --num_envs 512 --max_iterations 300 env.capture_radius=1.0 env.spawn_radius=4.0 env.attacker_speed=[0.2,0.4] > outputs\train_defend_stage0.log 2>&1
```

第 10 节加的三个 env 开关（默认值 = 原始行为，都能用 `env.<字段>=<值>` 覆盖，训练和 `evaluate.py` 都认）：

| 开关 | 默认 | 作用 |
|---|---|---|
| `env.closing_mode` | `distance` | `rew_scale_closing` 付给什么：`distance` = 与目标攻击者距离每步的减少量（攻击者自己飞过来，不动的防守机也拿钱）；`own_motion` = 防守机自身速度在"指向目标"方向上的投影 x step_dt，只有自己朝目标飞才计分 |
| `env.target_mode` | `nearest` | closing / proximity 项对准哪架攻击者：`nearest` = 最近的存活攻击者（两机可能追同一架）；`assigned` = 一对一指派（总距离最小的排列），指派的那架被击落后退回 nearest |
| `env.yaw_action` | `True` | `False` 时第四个动作通道（偏航角速度）被置零，动作空间仍是 4 维。它是第 10 节的绕行开关：当时力是在机体系施加的，偏航后水平速度环跟着机头转、180 度时反相（chase / zero 基线永远不偏航所以不受影响）。**第 11 节已把这个坐标系问题修掉，默认 `True` 即可，`False` 只用于复现第 10 节。** |

第 10 节的结论先写在这里：三个奖励开关都没用；唯一出过高于 zero 的策略的是 `env.yaw_action=False` **加上**把策略初始 std 从 1.0 降到 0.37
（skrl 的 Hydra 覆盖 `agent.models.policy.initial_log_std=-1.0`，评估时 `evaluate.py` 也接受同样的 `agent.*` 参数），但它看种子：seed 42 得 41.8% 拦截 / 78.5% 突破（易设置、波次关、200 回合；zero 16-23% / 92-98%，chase 100% / 0%），`--seed 1` / `--seed 2` 只有 10-11%，3 个种子 1 个成。没有找到可靠的改法，细节见第 10 节：

```bat
python -u scripts\skrl\train.py --task AS-Defend-v0 --algorithm IPPO --headless --num_envs 512 --max_iterations 300 env.capture_radius=1.0 env.spawn_radius=4.0 env.attacker_speed=[0.2,0.4] env.yaw_action=False agent.models.policy.initial_log_std=-1.0 > outputs\train_defend_s10.log 2>&1
```

**第 11 节（2026-10-17）修掉了控制器坐标系 bug，并在修好的环境上不锁偏航、不改奖励重跑了同一协议：3 个种子里 1 个到 79.2% 拦截 / 39.0% 突破（seed 1，目前最好），另两个在 zero 以下；初始 std 0.37 则 3 个里 2 个高于 zero（43.0%、31.2%）。所以上面这条 `env.yaw_action=False` 的命令只用于复现第 10 节，新实验不需要锁偏航。**

第 10 节的实验协议一键重跑：`bench.ps1 -Family defend_s10 -Tag <前缀> -Extra "<空格分隔的覆盖>" [-Iters 300]`
（易设置 + 覆盖训练 IPPO，再在同一易设置、`env.respawn_waves=False` 下各跑 200 回合评估策略 / zero / chase，最后导出 defender_0；
checkpoint 从自己的训练日志里找，所以几个 `-Tag` 不同的实例可以并行，但要错开几秒启动，否则 skrl 用同一秒的时间戳建同一个目录）。

评估（基线先跑，再跑策略；`--algorithm` 要和 checkpoint 一致）：

```bat
python -u scripts\evaluate.py --task AS-Defend-v0 --headless --baseline zero --episodes 20 > outputs\eval_zero.log 2>&1
```

```bat
python -u scripts\evaluate.py --task AS-Defend-v0 --headless --baseline chase --episodes 20 > outputs\eval_chase.log 2>&1
```

```bat
python -u scripts\evaluate.py --task AS-Defend-v0 --headless --checkpoint logs\skrl\tello_defend\<run>\checkpoints\agent_8000.pt --algorithm IPPO --episodes 20 > outputs\eval_policy.log 2>&1
```

导出（多智能体 checkpoint 里每个 defender 一套权重，用 `--agent` 选；IPPO 的共享网络和 MAPPO 的分离网络两种布局都认）：

```bat
python -u scripts\export_skrl_policy.py --checkpoint tello_defend --agent defender_0 --out outputs\exports\defend_defender_0.pt
```

录像，两种：策略录像（Replicator 路径，需要 vulkan=false）和 chase 基线的 viewport 截帧：

```bat
python -u scripts\evaluate.py --task AS-Defend-v0 --headless --checkpoint logs\skrl\tello_defend\<run>\checkpoints\agent_8000.pt --algorithm IPPO --episodes 2 --num_envs 4 --video --video_length 200 --video_dir outputs\videos\defend_eval --kit_args=--/app/vulkan=false > outputs\video_defend.log 2>&1
```

```bat
python -u scripts\record_clip.py --frames 150 --out outputs\videos\defend_chase.mp4 --frame_dir outputs\frames_clip > outputs\record_clip.log 2>&1
```

## 4. Approach（SB3 PPO，四机共享一个策略）

任务名：`AS-Tello-Approach-v0`。产物在 `logs\sb3\AS-Tello-Approach-v0\<run>\model.zip`。
出厂策略的三个 `env.*` 覆盖在训练、play、录像时都要带上。

```bat
python -u scripts\sb3\train.py --task AS-Tello-Approach-v0 --headless --num_envs 512 --max_iterations 800 agent.learning_rate=1e-3 env.spawn_azimuth_range=0.0 env.rew_scale_separation=-20.0 env.rew_violation=-100.0 > outputs\train_approach.log 2>&1
```

续训（从出厂策略或自己的 run）：

```bat
python -u scripts\sb3\train.py --task AS-Tello-Approach-v0 --headless --num_envs 512 --max_iterations 200 --checkpoint policies\approach_it5_room_fixed\model.zip agent.learning_rate=1e-3 env.spawn_azimuth_range=0.0 env.rew_scale_separation=-20.0 env.rew_violation=-100.0 > outputs\train_approach_resume.log 2>&1
```

打分（最后一行 `[N episodes] ... success_rate=...`）：

```bat
python -u scripts\sb3\play.py --task AS-Tello-Approach-v0 --headless --num_envs 64 --episodes 256 --checkpoint logs\sb3\AS-Tello-Approach-v0\<run>\model.zip env.spawn_azimuth_range=0.0 env.rew_scale_separation=-20.0 env.rew_violation=-100.0 > outputs\play_approach.log 2>&1
```

导出（VecNormalize + 8 帧堆叠 + 确定性策略 → TorchScript，输入 328 维）：

```bat
python -u scripts\export_sb3_policy.py --checkpoint AS-Tello-Approach-v0 --out outputs\exports\approach.pt
```

录像：

```bat
python -u scripts\record_approach.py --pilot policy --checkpoint logs\sb3\AS-Tello-Approach-v0\<run>\model.zip --spawn_azimuth_range 0.0 --set rew_scale_separation=-20.0 rew_violation=-100.0 --episodes 2 --out outputs\videos\approach.mp4 --frame_dir outputs\frames_approach > outputs\record_approach.log 2>&1
```

## 5. Room-Single / Room（SB3 PPO，真实房间 + AprilTag 感知）

任务名：`AS-Tello-Room-Single-v0`（单机，调奖励用）、`AS-Tello-Room-v0`（四机，一个中央控制器）。
产物在 `logs\sb3\<任务名>\<run>\model.zip`。先跑一次环境检查：

```bat
python -u scripts\check_room_env.py --headless --task AS-Tello-Room-Single-v0 > outputs\check_room_single.log 2>&1
```

```bat
python -u scripts\sb3\train.py --task AS-Tello-Room-Single-v0 --headless --num_envs 512 --max_iterations 1500 > outputs\train_room_single.log 2>&1
```

```bat
python -u scripts\sb3\train.py --task AS-Tello-Room-v0 --headless --num_envs 512 --max_iterations 1500 > outputs\train_room.log 2>&1
```

```bat
python -u scripts\sb3\play.py --task AS-Tello-Room-Single-v0 --headless --num_envs 64 --episodes 64 --checkpoint logs\sb3\AS-Tello-Room-Single-v0\<run>\model.zip > outputs\play_room_single.log 2>&1
```

```bat
python -u scripts\export_sb3_policy.py --checkpoint AS-Tello-Room-v0 --out outputs\exports\room.pt
```

```bat
python -u scripts\record_room.py --task AS-Tello-Room-v0 --checkpoint logs\sb3\AS-Tello-Room-v0\<run>\model.zip --episodes 2 --out outputs\videos\room.mp4 --frame_dir outputs\frames_room > outputs\record_room.log 2>&1
```

训练中途也能录：不给 `--checkpoint` 时取最新 run 里最新的 `model_N_steps.zip`。

## 6. 一键基准（本手册的数据就是它跑出来的）

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\bench.ps1 -Family hover,waypoint,defend,approach,room_single,room
```

每一步是一个独立的低优先级 Isaac Sim 进程，起跑前检查空闲显存（不足 4 GB 跳过），
结果写 `outputs\bench\timeline.tsv` 和每步的 `.log/.err`。汇总成表：

```bat
python scripts\windows\bench_report.py --out outputs\bench\report.md
```

## 7. 本机实测耗时

2026-10-17 凌晨，`bench.ps1` 跑出来的数字，原始表在 `simulation_infra\outputs\bench\report.md`，
每步的日志在 `outputs\bench\<step>.log`。全程 BelowNormal 优先级，隔壁有一个间歇性的 Anaconda GPU 任务
（空闲显存 10.7-15.7 GB），着色器缓存已热。"每轮秒"是 SB3 表里 `time_elapsed / iterations`、
skrl 用 tqdm 稳态速率换算，都不含启动。

### 训练

| 任务 | 算法 | num_envs | 本次轮数 | 每轮秒 | 启动秒 | 吞吐 | Linux 参考（文档） |
|---|---|---|---|---|---|---|---|
| `AS-Tello-Hover-v0` | skrl PPO | 1024 | 40 | **0.48** | 9 | 50 timesteps/s（≈51k transitions/s） | A10G 上 200 轮约 10 分钟（3 s/轮） |
| `AS-Tello-Waypoint-v0` | skrl PPO | 1024 | 40 | **0.55** | 10 | 44 timesteps/s | 无 |
| `AS-Defend-v0` | skrl IPPO | 512 | 30 | **0.68** | 13 | 23.5 timesteps/s | 无 |
| `AS-Defend-v0` | skrl MAPPO | 512 | 15 | **0.59** | 10 | 27 timesteps/s | 无 |
| `AS-Tello-Approach-v0` | SB3 PPO 共享 | 512 | 20 | **17.7** | 19 | 7.4k transitions/s | 4.7 s/轮（RTX 5080，Linux） |
| `AS-Tello-Room-Single-v0` | SB3 PPO | 512 | 6 | **58.8** | 16 | 2.2k transitions/s | 无 |
| `AS-Tello-Room-v0` | SB3 PPO 中央 | 512 | 6 | **112.7** | 14 | 1.2k transitions/s | 27 s/轮（RTX 5080，Linux） |

按这些速率推算文档里的完整训练量（不含启动）：

| 训练量 | 本机预计 | 文档值 |
|---|---|---|
| Hover 200 轮 | 1.6 分钟 | 10 分钟（A10G） |
| Defend IPPO 500 轮 | 6 分钟 | 无 |
| Approach 800 轮 | **3.9 小时** | 1 小时 |
| Room-Single 1500 轮 | **24.5 小时** | 无 |
| Room 1500 轮 | **47 小时** | 11 小时 |

Approach / Room 这两个家族要靠加大 `--num_envs` 或者夜里挂机，见下面的批次对比和第 8 节第 6 条。

### 批次对比（Approach，6 轮）

| num_envs | 优先级 | 每轮秒（首轮 / 稳态） | transitions/s | 每轮样本 | 相对 512 的样本吞吐 |
|---|---|---|---|---|---|
| 512 | BelowNormal | 29 / 17.7 | 7.4k | 131k | 1.0x |
| 512 | Normal | 23 / 19.8 | 6.4k | 131k | 0.9x（**优先级不是原因**） |
| 2048 | BelowNormal | 32 / 23.8 | 20.8k | 524k | **2.8x** |

结论：本机 Approach 用 `--num_envs 2048` 划算得多。出厂配方 800 轮 x 512 envs（1.05 亿样本）在 2048 envs 下
等于约 200 轮，约 1.4 小时；按 `policy_maker.md` 的提醒，批次变大后把 `agent.learning_rate` 降到 3e-4 左右，
并与一次 512 的跑法对比后再信。Room 家族还没做同样的对比，预计同样受益。

### 其它步骤

| 步骤 | 耗时 | 产物 / 结果 |
|---|---|---|
| skrl `play.py --steps 400`，16 envs | 16-26 s | `[play] 48 episodes finished: mean return 9.17` |
| skrl `play.py --video` 200 步，4 envs，D3D12 | 36-43 s | `logs\skrl\<exp>\<run>\videos\play\rl-video-step-0.mp4`，229 KB，3.3 s 1280x720 |
| `evaluate.py` 20 回合，64 envs | 17-30 s | zero 2.4% / 100%，chase 50% / 100%，30 轮策略 0% / 100% |
| `evaluate.py --video` 200 步，4 envs | 36 s | `outputs\videos\bench_defend_eval\rl-video-step-0.mp4`，349 KB |
| `record_clip.py` 100 帧（chase 基线） | 48 s | `outputs\videos\bench_defend_chase.mp4`，243 KB |
| SB3 `play.py` Approach 64 回合，64 envs | 128 s | 20 轮策略：illegal_entry 100%（意料之中） |
| `record_approach.py` 1 回合 | 239 s | `outputs\videos\bench_approach.mp4`，776 KB，374 帧 |
| `check_room_env.py --task AS-Tello-Room-Single-v0` | 257 s | 21 ok，ALL CHECKS PASSED |
| SB3 `play.py` Room-Single 64 回合 | 109 s | 6 轮策略：success 100%，spot_err 0.18 m（见第 8 节第 9、12 条） |
| SB3 `play.py` Room 16 回合 | 90 s | 6 轮策略：collision 100%，10 s 结束（意料之中） |
| `record_room.py` 1 回合（75 s） | 846-928 s | `outputs\videos\bench_room_single.mp4` 2.3 MB，`bench_room.mp4` 2.9 MB，各 1039 帧 |
| `export_sb3_policy.py` / `export_skrl_policy.py` | 5-6 s | `outputs\exports\bench_*.pt`：hover/waypoint 12→4，defend 23→4，approach 328→4，room_single 200→5，room 768→20 |

## 8. 跑基准时发现的问题

按影响大小排列。前五条已经处理（规则或补丁），后面的是需要在第二层（正式训练）开始前核实的事。

1. **`--video` 必须 `--kit_args=--/app/vulkan=false`。** 无头录像加载的 `isaaclab.python.headless.rendering.kit` 写死 `app.vulkan = true`，本机 Vulkan 启动 9 秒后 access violation 崩溃（和 `WINDOWS.md` 里窗口模式的崩溃是同一个根因）。加参数后 skrl play、`evaluate.py --video` 都正常出 1280x720 的 mp4。崩溃会在 `Z:\isaac\env_isaaclab\Lib\site-packages\isaacsim\kit\data\Kit\Isaac-Sim\5.1\` 留下 10 MB 的 `.dmp`，可删。
2. **skrl `play.py` 无头时永不退出。** 上游只在 `--video` 时跳出循环。本机补丁加了 `--steps N`，并在结束时打印 `[play] N episodes finished: mean return ...`（多智能体任务把各 agent 的奖励求和）。
3. **导出。** 仓库只有 SB3 的导出函数（录像器内部用），没有命令行入口，也没有 skrl 的导出。新增 `scripts\export_sb3_policy.py`（VecNormalize + 帧堆叠 + 确定性策略，输出 328 维输入的 TorchScript）和 `scripts\export_skrl_policy.py`（RunningStandardScaler + MLP + 裁剪；IPPO/MAPPO 用 `--agent` 选 defender）。两者都不需要 Kit，CPU 上 5 秒。注意参数里的实验名只接受文件路径或 `logs\` 下的实验目录名：仓库根目录恰好有个 `tello_defend\` 文件夹，不能被当成 checkpoint。
4. **`bench.ps1` 的两个 PowerShell 5.1 坑**：`powershell -File xxx.ps1 -Family a,b` 把数组参数当成一个字符串传入（脚本内按逗号拆）；`Start-Process -PassThru` 后不先读 `$p.Handle`，`WaitForExit` 之后 `ExitCode` 是空的。
5. **Git Bash 的 `tail -F` 会锁住被它读的文件**，PowerShell 往同一个文件 `Add-Content` 会报 "being used by another process"。监控进度要看另一个文件（console 日志），不要看 `timeline.tsv`。
6. **SB3 家族在本机比 Linux 参考慢 2-4 倍，skrl 家族不慢。** Approach 每轮 17 s（Linux 参考 4.7 s），Room-Single 每轮 59 s（Linux 四机版参考 27 s），训练时 GPU 利用率只有 0-20%。原因方向：Hover/Defend 每个策略步只有 2 个物理子步，Approach 是 25 个、Room 是 37 个，每个子步都要在 torch 里跑一遍 Tello 飞控级联，Windows（WDDM）下 CUDA 小核函数的启动开销远高于 Linux，物理子步多的任务就被启动开销卡住。实测排除了进程优先级和隔壁任务抢占（512 envs 正常优先级同样 20 s/轮）；加大 `--num_envs` 能摊薄每步的固定开销：2048 envs 时样本吞吐是 512 的 2.8 倍（第 7 节批次对比）。
7. **Defend 的 chase 基线和 README 对不上：是 chase 代码的 bug，已修。** README 写 chase 100% 拦截 / 0% 突破，本机最初实测 50% / 100%。原因：环境把已击落攻击者的相对位置置零（配套的 alive 标志在观测末尾），而 `evaluate.py` 和 `record_clip.py` 里的 chase 取 `argmin(距离)` 时没有屏蔽击落者，第一次拦截后两架防守机都"追"一个零向量、原地悬停，另一架攻击者必然突破。README 的数字来自改观测（去掉 100 m 哨兵值）之前。两处都改成只在存活攻击者里选最近的；顺便给 `evaluate.py` 加了 `env.<字段>=<值>` 覆盖（它原来不走 Hydra）。修复前后 100 回合的数字（默认难度）：

   | 基线 | 波次 | 拦截率 | 突破率 |
   |---|---|---|---|
   | zero | 开（默认） | 2.5% | 100% |
   | zero | 关 | 1.5% | 99% |
   | chase 修复前 | 开 | 59.5% | 100% |
   | chase 修复后 | 开 | 113%（每回合 2.26 架） | 94% |
   | chase 修复后 | 关 | **86.5%** | **26%** |

   两点说明：开着波次时"interception rate of attackers"是"每回合拦截数 ÷ 攻击者数"，可以超过 100%，而突破率在 20 秒回合里几乎总会到 100%，所以**比较策略请用 `env.respawn_waves=False`**（README 的 100% / 0% 就是这种设置下的理想值；修复后的贪心追击只有 86.5% / 26%，因为两架防守机先追同一架，掉头截第二架时它常常已经太近）。
8. **Defend IPPO 30 轮：奖励 -42.6 → -67.5，评估 0% 拦截**，和 README 说的"PPO 退化成被动"一致。这是课程学习要解决的问题，不是环境坏了。
9. **Room-Single 的 play 和录像结果不一致。** 6 轮训练后 `sb3\play.py` 报 64 回合 100% 成功（均误差 0.18 m），但 `record_room.py` 录的那一回合是 FAIL（waypoints 0/1，75 s 跑满）。play 用 SB3 原生推理，录像器用导出的 TorchScript，二者的观测归一化或帧堆叠可能有差异，第二层要用同一组观测比对两条推理路径的输出。
10. **录像很慢。** `record_room.py` 一回合 75 s 要 14 分钟（1039 帧，每帧约 0.8 s），`record_approach.py` 一回合约 4 分钟，skrl `--video` 200 步约 35-45 秒。要看多回合时先用 play 打分，只给最好的 checkpoint 录像。
11. **隔壁的 Anaconda GPU 任务是间歇性的**（空闲显存在 10.9 GB 和 15.6 GB 之间来回），基准全程以 BelowNormal 优先级运行，耗时表里记录了每步起跑时的空闲显存。
12. **Room-Single 6 轮就 100% 成功**，好得可疑，正式训练时要复核成功判据。
13. Isaac Sim 无头启动到开始训练约 9-13 秒（着色器缓存已热）；`check_room_env.py` 单机 21 项检查本机首次通过，耗时 4.3 分钟。
14. `evaluate.py` 每次运行都会往 `simulation_infra\tello_defend\tello_defend_ippo\` 写一个 88 字节的空 tfevents（skrl 的默认实验目录相对于当前目录），仓库里上游留下的那三个就是这么来的。无害，可删。

## 9. Defend 课程第一阶段（2026-10-17）

README 建议的课程起点：`env.capture_radius=1.0 env.spawn_radius=4.0 env.attacker_speed=[0.2,0.4]`，
512 envs，300 轮（IPPO 4.6 分钟，MAPPO 5.4 分钟）。评估一律在**同一组易设置**下，波次开着（默认），
所以拦截率是"每回合拦截数 ÷ 2"。`bench.ps1 -Family defend_stage0` / `defend_stage0_mappo` 可以整套重跑。

| 对象（易设置，50-64 回合） | 拦截率 | 突破率 | 训练中总奖励均值（首轮 → 末 1/4 平均） |
|---|---|---|---|
| zero（两次） | 12.0%、21.0% | 100% | - |
| chase（修复后，两次） | 363%、414%（每回合 7-8 架） | 26.6%、14.1% | - |
| IPPO 300 轮 | 13.7% | 100% | -135 → -208 |
| MAPPO 300 轮 | 18.0% | 100% | -150 → -223 |

同一基线两次 50 回合的结果能差 9 个百分点（`evaluate.py` 不固定种子），所以两个策略的 13.7% 和 18.0% 与 zero 没有统计差别；要分辨小改进得跑 200 回合以上。

结论：**易设置并没有解决"退化成被动"**。两种算法 300 轮后都和 zero 基线差不多（zero 在 1 m 捕获半径下也能"蹭"到 12%，
因为攻击者会从防守机身边飞过），训练曲线从第一轮起就单调下滑，而 chase 在同样设置下每回合能截 7 架。
所以 README 里"随机探索撞不到 0.35 m 捕获半径、没有正信号"的解释不成立：正信号有（瞬时奖励最大值到 90），
策略仍然选择不动。下一步该查的是奖励结构本身，而不是继续放宽课程：

- `rew_scale_closing`（5.0 x 每步拉近的距离）对**不动的**防守机也付钱，因为攻击者自己在朝基地飞；主动出击反而可能先拉远与另一架的距离。
- 突破 -20 是共享惩罚且只结束回合，不动的局部最优是"等攻击者飞过来蹭一下捕获奖励，然后被突破"；两架同时出击也拦不住第二架，于是没有动机。
- 可试的改法（每次只改一个）：把 closing 改成只在防守机自身速度朝向目标时计分；给每架防守机指派各自的目标（README 也提了）；
  用 `env.respawn_waves=False` 训练让清场成为明确的回合终点；从 chase 做行为克隆热启动（第三层的工具）。

Checkpoint 与导出：IPPO `logs\skrl\tello_defend\2026-10-17_03-16-03_ippo_torch_tello_defend_ippo\checkpoints\agent_4800.pt`（导出 `outputs\exports\defend_stage0_ippo_300.pt`），MAPPO `logs\skrl\tello_defend\2026-10-17_03-25-47_mappo_torch_tello_defend_mappo\checkpoints\`（导出 `outputs\exports\defend_stage0_mappo_300.pt`）。日志在 `outputs\bench\defend_stage0_*.log`、`defend_base_*.log`。

## 10. Defend 奖励 / 终止结构实验（2026-10-17，接第 9 节）

问题：第 9 节的易设置下 IPPO 300 轮仍和 zero 基线一样（13.7% 拦截），训练曲线从第一轮起下滑。
本节按"每次只改一项"的协议把第 9 节末尾列的候选改法逐个量化，并先查清退化策略到底在做什么。

### 10.1 协议

- 训练：`python -u scripts\skrl\train.py --task AS-Defend-v0 --algorithm IPPO --headless --num_envs 512 --max_iterations 300 env.capture_radius=1.0 env.spawn_radius=4.0 env.attacker_speed=[0.2,0.4] <改动>`，曲线上升的再跑 600 轮。
- 评估：同一组易设置 + `env.respawn_waves=False`（清场即回合结束，和 README 的数字可比），策略、`--baseline zero`、`--baseline chase` 各 200 回合：
  `python -u scripts\evaluate.py --task AS-Defend-v0 --headless --checkpoint <ckpt> --algorithm IPPO --episodes 200 env.capture_radius=1.0 env.spawn_radius=4.0 env.attacker_speed=[0.2,0.4] env.respawn_waves=False`。
- 曲线：TensorBoard 的 `Reward / Total reward (mean)`（skrl 把两架防守机的奖励**相加**后按回合累计，所以是两机之和）首值 → 末值。
- 一键：`powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\bench.ps1 -Family defend_s10 -Tag <前缀> -Extra "<改动>" [-Iters 600]`，步骤日志在 `outputs\bench\<前缀>_{train,eval_policy,eval_zero,eval_chase,export}.log`，汇总 `python scripts\windows\bench_report.py`。
  这一家族从自己的训练日志里找 checkpoint，不再取"最新目录"。
- 基线在这组设置下（200 回合，本节共跑了 13 次）：zero 16-23% 拦截 / 92-98% 突破，chase 99-100% / 0-1.5%。同一个 zero 不同次能差 7 个百分点，所以 ±5% 以内的差别不算差别。

### 10.2 先看退化策略在干什么

`scripts\evaluate.py` 只给拦截率。用一个诊断脚本（加载 checkpoint，跑 600 步 x 64 envs，记录策略均值动作、饱和比例、防守机离基地 / 离最近攻击者的距离）看第 9 节的 IPPO 300 轮 checkpoint：

| 时间 | 离最近攻击者 (m) | 离基地水平距离 (m) |
|---|---|---|
| 0 s | 3.6 | 2.5 |
| 2 s | 3.2 | 2.3 |
| 4 s | 8.4 | 8-9 |
| 6 s | 12.6 | 12-14 |

- 策略**不是不动，是以约 3 m/s 逃离基地**，比速度接口的上限（每轴 1 m/s）还快。
- 策略网络输出的均值远超 [-1, 1]：四个通道 |均值| 平均 1.7 / 0.7 / 1.2 / **3.7**（偏航通道），饱和（|a| >= 0.99）比例 0.5 / 0.2 / 0.4 / **0.8**。
- 原因在控制器和 Isaac Lab 的接口：`TelloVelocityController.compute` 按**世界系**速度误差算力（`vel_error = cmd - lin_vel_w`），而 `RigidObject.set_external_force_and_torque` 默认 `is_global=False`，把这个力当**机体系**施加。
  只要机头偏航，水平速度环就跟着机头转：偏 90 度时推力方向和误差方向垂直，偏 180 度时反相，速度环变成正反馈，于是飞出 3 m/s。
  chase / zero 基线偏航通道恒为 0、机头永远朝初始方向，所以完全不受影响（也就是 README 说的"好策略存在"）。
  学习中的高斯策略从第一轮起就以 std ~0.9 随机偏航，观测里又没有航向（只有 projected gravity，不含 yaw），策略根本无法把世界系的相对位置映射成正确的动作；
  飞走后 closing 项持续为负，这就是"训练曲线从第一轮起下滑"。
- Hover / Waypoint 任务没撞上这个问题，是因为它们有 `rew_scale_ang_vel` 惩罚角速度，策略很快学会不偏航。Defend 的 effort 惩罚（-0.01 x a² x step_dt）太小，起不到这个作用。
- 顺带发现：`DirectMARLEnv.step` 先调 `_get_dones` 再调 `_get_rewards`，`_breached` 是上一步算出来的，所以突破那一步和结束那一步各扣一次 -20（每机 -40），不影响结论。

### 10.3 候选改法与结果

| 改动（在易设置之上） | 覆盖参数 | 轮数 | 策略 拦截 / 突破（回合） | zero | chase | 训练总回报均值（两机之和）首 → 末 |
|---|---|---|---|---|---|---|
| 候选 1：训练时波次关 | `env.respawn_waves=False` | 300 | 22.1% / 94.0% (201) | 22.0% / 95.0% | 100.0% / 0.0% | 137 -> -225 (末 1/4 均值 -234, 最高 137) |
| 候选 2：closing 只付自身运动 | `env.closing_mode=own_motion` | 300 | 18.8% / 95.5% (200) | 19.2% / 94.0% | 100.0% / 0.0% | -153 -> -286 (末 1/4 均值 -234, 最高 -102) |
| 候选 3：一对一指派目标 | `env.target_mode=assigned` | 300 | 19.8% / 95.5% (200) | 22.0% / 93.5% | 100.0% / 0.0% | -151 -> -394 (末 1/4 均值 -234, 最高 -105) |
| 备用：capture 100 + entropy 0.01 | `env.rew_scale_capture=100.0 agent.agent.entropy_loss_scale=0.01` | 300 | 11.5% / 97.5% (200) | 16.2% / 97.5% | 100.0% / 0.0% | -143 -> -310 (末 1/4 均值 -274, 最高 -78) |
| 锁偏航 | `env.yaw_action=False` | 300 | 23.2% / 90.5% (200) | 18.2% / 97.0% | 100.0% / 0.0% | -73 -> -33 (末 1/4 均值 -30, 最高 -7) |
| 锁偏航，600 轮 | `env.yaw_action=False` | 600 | 20.5% / 93.0% (200) | 20.8% / 92.5% | 99.8% / 0.5% | -76 -> -46 (末 1/4 均值 -35, 最高 9) |
| 锁偏航 + tanh 限幅均值 | `env.yaw_action=False agent.models.policy.output='tanh(ACTIONS)'` | 300 | 9.7% / 98.0% (201) | 19.8% / 95.5% | 98.8% / 1.5% | -68 -> -85 (末 1/4 均值 -77, 最高 -29) |
| 锁偏航 + own-motion closing | `env.yaw_action=False env.closing_mode=own_motion` | 300 | 33.0% / 91.0% (200) | 22.0% / 94.5% | 99.5% / 0.5% | -71 -> -59 (末 1/4 均值 -54, 最高 58) |
| 锁偏航 + own-motion closing x50 | `env.yaw_action=False env.closing_mode=own_motion env.rew_scale_closing=50.0` | 300 | 14.0% / 99.0% (200) | 21.2% / 95.5% | 99.2% / 1.0% | -117 -> -564 (末 1/4 均值 -508, 最高 -117) |
| **锁偏航 + 初始 std 0.37**（seed 42） | `env.yaw_action=False agent.models.policy.initial_log_std=-1.0` | 300 | 41.8% / 78.5% (200) | 17.2% / 98.0% | 100.0% / 0.0% | -53 -> 45 (末 1/4 均值 23, 最高 94) |
| 同上，600 轮（seed 42） | `env.yaw_action=False agent.models.policy.initial_log_std=-1.0` | 600 | 30.8% / 88.5% (200) | 18.5% / 97.5% | 100.0% / 0.0% | -49 -> 4 (末 1/4 均值 4, 最高 62) |
| 同上，`--seed 1` | `env.yaw_action=False agent.models.policy.initial_log_std=-1.0` | 300 | 10.0% / 98.0% (200) | 18.2% / 97.0% | 100.0% / 0.0% | -48 -> -86 (末 1/4 均值 -61, 最高 -7) |
| 同上，`--seed 2` | `env.yaw_action=False agent.models.policy.initial_log_std=-1.0` | 300 | 11.0% / 95.5% (200) | 20.0% / 96.5% | 100.0% / 0.0% | -50 -> -83 (末 1/4 均值 -73, 最高 -1) |

### 10.4 结论

**任务里列的三个候选（波次关、own-motion closing、一对一指派）没有一个把策略拉到 zero 之上**：200 回合下都在 19-23% 拦截 / 90-96% 突破，和 zero 的 16-23% / 92-98% 在同一噪声带里，训练曲线仍从第一轮起下滑。备用的 `rew_scale_capture=100 + entropy_loss_scale=0.01`（c5）也一样。
原因是 10.2 节的那个：奖励怎么改都没用，因为策略一偏航就失去对水平运动的控制，奖励信号对应不到动作上。

**唯一产生过高于 zero 的策略的改法是 `env.yaw_action=False` + `agent.models.policy.initial_log_std=-1.0`（初始 std 1.0 → 0.37），但它依赖种子**：
seed 42（yaml 默认）300 轮 **41.8% 拦截 / 78.5% 突破**（比 zero 多 20 个百分点、少 15 个百分点突破，离 chase 的 100% / 0% 还有一半多），同一种子 600 轮 30.8% / 88.5%；
`--seed 1` 300 轮只有 10.0% / 98.0%（回报 -48 → -86），`--seed 2` 11.0% / 95.5%（回报 -50 → -83），都和没改一样。
也就是 3 个种子里只有 1 个（seed 42，两次都是）学会了出击。所以这个改法的正确说法是"打开了一扇门"：它是本节十几轮实验里唯一让训练回报变正、上升（-53 → +45）的配置，但 300 轮内能不能走进去看种子，**没有找到可靠地让 IPPO 出击的奖励 / 终止改法**。
seed 42 的策略学到的是"守门"而不是追击：一架防守机停在基地旁 1.7-2.0 m 处等攻击者自己飞过来（诊断脚本里 600 步 x 64 envs 的拦截数 64，其它策略 15-37），追击余弦只有 +0.13，x 通道仍是 ±1 的 bang-bang。
默认难度（捕获半径 0.35、出生半径 6、速度 0.4-0.9，波次关）下它 4.8% / 98.5%（zero 1.5% / 99%，chase 86.5% / 26%）；易设置、波次开（第 9 节的口径）53.0%（每回合 1.06 架），第 9 节的 IPPO 是 13.7%。

单项都不够、拆开看各自的作用：
- 只锁偏航（`yaw_action=False`，c4）：不再飞走（回合回报最低值 -173 而不是 -650），训练回报从 -73 升到 -33 但停在负值，评估 23.2% / 90.5%（600 轮 20.5% / 93.0%），和 zero 没差别。诊断：策略均值仍在 [-1, 1] 之外饱和（某个通道 60-93% 的步数 |a| >= 0.99、符号恒定），每架机沿固定方向漂，不跟踪攻击者（动作与目标方向的余弦 -0.07 / -0.34）。
- 锁偏航 + 把均值用 tanh 限幅（`agent.models.policy.output='tanh(ACTIONS)'`，c8）：9.7% / 98.0%，更差。限幅不解决问题；降低初始探索噪声有时解决：std 0.37 时被裁剪的样本从 32% 降到 0.7%。
- 锁偏航 + own-motion closing（标度 5，c6）：33.0% / 91.0%，第二个高于 zero 的配置（+11 个百分点），但突破率几乎没降，回报 -71 → -59 仍为负（单种子，同样要打折）；标度 50（c9）：14.0% / 99.0%，回报 -117 → -564，策略学会背着目标飞——哪怕奖励只剩 own-motion closing 一项（`outputs\s10\train_pure_closing.log`，100 轮）也是如此（每步均值 0 → -0.8），而只奖励 |a| 的管线检查（`train_pure_effort.log`）正常上升。所以是"状态相关奖励 + 裁剪高斯 + 速度环滞后"这组合把早期梯度带偏，不是奖励符号或 skrl 的问题（own-motion 项在 chase 下每步 +0.64，上限 0.83）。

下一步建议（按优先级）：
1. 用 `is_global=True` 或机体系速度指令 + 观测里加航向来**修掉**控制器的坐标系问题，而不是靠锁偏航绕过（锁偏航等于部署时机头固定，bridge 要保证起飞航向）。修掉之后再回头看这些奖励候选才有意义。
2. 多种子（>= 3）再报任何数：同配置两次 seed 42 差 11 个百分点，换种子差 30 个百分点；seed 42 两次都成、seed 1 / 2 都不成，说明成败在前几轮的初始化就定了。
3. 从 41.8% 的 checkpoint（`outputs\exports\s10_c7.pt`）续训并逐步把课程参数调回默认；或者按第 9 节末尾说的，从 chase 做行为克隆热启动再 PPO 微调——热启动后的 std 也要小。

### 10.5 代码改动（未提交）

- `tello_defend_env_cfg.py` / `tello_defend_env.py`：三个开关 `closing_mode`、`target_mode`、`yaw_action`（第 3 节的表）。默认值下奖励和原来逐位一致：用 HEAD 版本的 `_get_rewards` 在同一状态上对比 400 步 x 32 envs，随机动作和零动作最大差都是 0.0。
- `scripts\evaluate.py`：除 `env.<字段>=<值>` 外也接受 `agent.<键>=<值>`（和 train.py 的 Hydra 路径一致，例如 `agent.models.policy.output='tanh(ACTIONS)'`），按 checkpoint 训练时的模型结构重建网络。
- `scripts\export_skrl_policy.py`：识别 `params\agent.yaml` 里 `models.policy.output` 的 `tanh(ACTIONS)` 之类的输出激活（它没有参数，state dict 里看不出来），导出时加在裁剪之前。
- `scripts\windows\bench.ps1`：`defend_s10` 家族（10.1 节）、`-Tag / -Extra / -Iters` 参数、`Skrl-Ckpt-Of-Log`（从训练日志找自己的 run 目录）。
- 本节的诊断脚本（策略均值 / 饱和 / 追击余弦 / 距离随时间）在会话 scratchpad 里，不在仓库中；日志：`outputs\bench\s10_*.log`（协议步骤）、`outputs\s10\`（诊断、符号检查、纯奖励试验）。

## 11. Defend 控制器坐标系修复，第 10 节协议重跑（2026-10-17，接第 10 节）

第 10.2 节查明的根因：`TelloVelocityController.compute` 按**世界系**速度误差算力，而 `RigidObject.set_external_force_and_torque` 默认 `is_global=False`、把它当**机体系**施加，于是一偏航水平速度环就跟着机头转、180 度时反相，策略以 3 m/s 飞走。
本节把它修掉、验证，再把第 10 节的协议在修好的环境上重跑（**不锁偏航、不改奖励**），看 IPPO 能不能自己学会出击。

### 11.1 改法与理由

- 选的是**世界系速度环、机体系施力**：`control\tello_command.py` 的 `compute` 多一个参数 `quat_w`（机体姿态四元数 w,x,y,z），内部把世界系的力用姿态的逆旋转到机体系，返回 `(force_b, torque_b)` 都在机体系；两个环境（`tello_defend_env.py` 防守 / 攻击两个循环，`tello_hover_env.py`；waypoint 继承 hover）传 `quat_w=drone.data.root_quat_w`，`set_external_force_and_torque` 仍用默认 `is_global=False`。
- **没用任务里提的 `is_global=True`**：查 Isaac Lab 2.3.2 的 `utils\wrench_composer.py` 和 `utils\warp\kernels.py`（`set_forces_and_torques_at_position` → `cast_force_to_link_frame` / `cast_torque_to_link_frame`），`is_global=True` 会把力**和力矩**都用 `quat_rotate_inv` 从世界系转到机体系；调平力矩（roll / pitch）和偏航力矩是机体系量，这样传姿态环就错了。要走 `is_global=True` 得先把力矩转到世界系再让它转回来，等价于现在的做法但多一次旋转。
- **没用"机体系 rc 通道 + 观测加航向"的方案**：Defend 的观测（自身速度、相对位置 / 速度）、chase 基线、奖励全在世界系，策略不需要航向；偏航通道现在只改变机头朝向，对任务没有影响。真机的 `rc` 通道是机体系的，而 `bridge\tello_bridge.py` 目前把动作直接映射成 rc，所以**部署桥要先用定位给的 yaw 把世界系速度旋到机体系**（待办，本节没改 bridge）。
- `env.yaw_action` 保留（默认 `True`），现在只是复现第 10 节用的开关，cfg 里的说明已改。
- 旋转公式是纯 torch 的（控制器模块不引入 isaaclab），用 1000 个随机四元数对比旋转矩阵 R^T v，最大误差 1.3e-15；yaw 180 度时世界 +x 正确变成机体 -x（`outputs\s11\quat_check.py`）。

### 11.2 验证

偏航跑飞测试 `outputs\s11\yaw_test.py`（`--task AS-Defend-v0` / `AS-Tello-Hover-v0`）：16 envs，5 s，动作 `[vx=1, 0, 0, yaw_rate=1]`（世界系 +x 1 m/s，100 deg/s），Defend 里 defender_1 不偏航作对照，攻击者速度置 0 防止突破结束回合；Hover 设 `max_distance=100.0` 防止重置。

| 环境 / 机 | 修前（`outputs\s11\yaw_test_before_*.log`） | 修后（`yaw_test_after_*.log`） |
|---|---|---|
| Defend defender_0（5 s 转了 468 度） | 最高 **4.03 m/s**，末 2 s 平均 1.65 m/s，世界 vx **-0.81**（反向）；t=3 s 时 3.84 m/s | 最高 1.00 m/s，末 2 s 1.00 m/s，世界 vx +1.00 |
| Defend defender_1（不偏航，对照） | 1.00 / 1.00 / +1.00 | 1.00 / 1.00 / +1.00 |
| Hover robot（转了 468 度） | 最高 4.03，末 2 s 1.65，vx -0.81 | 1.00 / 1.00 / +1.00 |

- `scripts\check_approach_env.py --headless --test all`（`outputs\s11\check_approach.log`）：22 个 `[ok]`、0 个 FAIL，zero 0% / scripted 100%，team return -787.7 / +1539.5（approach 用的是另一套 `tello_dynamics` 控制器，本来就是机体系施力，这里只是回归检查）。
- 基线（易设置、波次关、200 回合，`outputs\s11\eval_{chase,zero}_fixed.log`）：chase **100.0% / 0.0%**（拦截用时 2.7 s），zero 20.0% / 92.0%，和第 10 节的带（zero 16-23% / 92-98%，chase 99-100% / 0-1.5%）一致——基线从不偏航，修复对它们是恒等变换。

### 11.3 第 10 节协议重跑（不锁偏航）

`bench.ps1 -Family defend_s10 -Tag s11_<name> -Extra "<改动> --seed <N>"`，300 轮；评估易设置 + `env.respawn_waves=False` 各 200 回合（10.1 节协议，`--seed` 只进训练）。
每行一个种子；zero / chase 是同一轮 bench 里同设置跑的基线；曲线列是 TensorBoard `Reward / Total reward (mean)`（两机之和）。

| 改动（在易设置之上） | 覆盖参数 | 种子 | 策略 拦截 / 突破（回合） | zero | chase | 训练总回报均值（两机之和）首 → 末 |
|---|---|---|---|---|---|---|
| 不改（修复后的原始奖励） | 无 | 1 | **79.2% / 39.0%** (200) | 18.7% / 97.0% | 100.0% / 0.0% | -45 -> 120 (末 1/4 均值 119, 最高 283) |
| 同上 | 无 | 2 | 5.2% / 98.5% (200) | 20.2% / 94.0% | 100.0% / 0.0% | -97 -> -102 (末 1/4 均值 -101, 最高 -79) |
| 同上 | 无 | 3 | 11.8% / 98.5% (200) | 20.2% / 94.5% | 100.0% / 0.0% | -75 -> -86 (末 1/4 均值 -103, 最高 -56) |
| 初始 std 0.37 | `agent.models.policy.initial_log_std=-1.0` | 1 | 10.8% / 99.5% (200) | 18.2% / 96.5% | 100.0% / 0.0% | -41 -> -69 (末 1/4 均值 -66, 最高 -11) |
| 同上 | 同上 | 2 | **43.0% / 78.5%** (200) | 20.8% / 94.0% | 100.0% / 0.0% | -48 -> -5 (末 1/4 均值 -17, 最高 60) |
| 同上 | 同上 | 3 | **31.2% / 88.5%** (200) | 17.4% / 96.0% | 100.0% / 0.0% | -44 -> -35 (末 1/4 均值 -23, 最高 96) |

对照第 10 节：修复前"不改"（第 9 节 / 10.3 候选 1-3）全在 zero 带里（13-23% / 94-100%），曲线从第一轮起下滑；修复前最好的是锁偏航 + std 0.37 的 seed 42，41.8% / 78.5%。

### 11.4 结论

**修好坐标系之后，IPPO 不锁偏航、不改奖励也能学会出击，但还不稳定。** 6 次训练里 3 次明显高于 zero：不改奖励的 seed 1 得 **79.2% 拦截 / 39.0% 突破**，是目前最好的策略（比第 10 节锁偏航 + std 0.37 的 41.8% 高近一倍，突破率减半），训练回报从 -45 升到 +120；初始 std 0.37 的 seed 2（43.0% / 78.5%）和 seed 3（31.2% / 88.5%）也高于 zero。另外 3 次停在 zero 以下（5-12% 拦截、98-99% 突破，训练回报全程为负），离 chase 的 100% / 0% 都还有距离。
也就是说，坐标系 bug 是"必要条件"而不是"充分条件"：修掉它后不再需要 `env.yaw_action=False`（偏航只改机头朝向，不再扰动速度环），奖励信号终于对应得上动作，第一次出现了不靠任何绕行就能学会出击的种子；但 300 轮内能否出击仍取决于种子 / 初始化，和第 10 节"成败在前几轮就定了"的观察一致。初始 std 0.37 把高于 zero 的种子从 1/3 提到 2/3，却把 seed 1 从 79% 压到 11%，3 个种子分不出它和"不改"谁更好。
下一步（按优先级）：(1) 多种子 / 600 轮，看出击种子的比例是否稳定、不出击的种子多训练会不会翻过来；(2) 从 seed 1 的 checkpoint 续训并把课程参数逐步调回默认（`logs\skrl\tello_defend\2026-10-17_12-40-52_ippo_torch_tello_defend_ippo\checkpoints\agent_4800.pt`，导出 `outputs\exports\s11_base_seed1.pt`）；(3) 现在回头试第 10 节的奖励候选（own-motion closing、一对一指派）才有意义——它们之前被 bug 掩盖；(4) 部署桥加 yaw 旋转（11.1）。

### 11.5 代码改动（未提交）

- `source\...\control\tello_command.py`：`compute(actions, lin_vel_w, quat_w, ang_vel_b, projected_gravity_b, dt)` 返回机体系 `(force_b, torque_b)`；模块 docstring 写明坐标系约定和为什么不用 `is_global=True`；`_rotate_inverse` 纯 torch。
- `tasks\direct\tello_defend\tello_defend_env.py`（防守 / 攻击两个循环）、`tasks\direct\tello_hover\tello_hover_env.py`：传 `quat_w=drone.data.root_quat_w`；`tello_defend_env_cfg.py`：`yaw_action` 的说明改成"第 10 节的绕行开关"。
- `scripts\windows\bench.ps1`：`Family-DefendS10` 里 `$extra` 改名 `$extraArgs`。PowerShell 变量名不分大小写，原来那行等于把拆开的列表赋回 `[string]$Extra` 参数、重新拼成一个字符串，于是 `--seed N` 跟着 `agent.models.*` 一起被转给 `evaluate.py`（exit 2；第 10 节 `s10_c7_seed1` 的评估失败也是这个原因）。本节 std 0.37 三组的评估因此由 `outputs\s11\run_s11_evals.ps1` 补跑（同名日志，timeline 追加行）。
- 测试 / 工具都在 `outputs\s11\`（不在仓库）：`yaw_test.py`、`quat_check.py`、`apply_fix.py`（打补丁用）、`run_s11.cmd`（6 次 bench 串行）、`run_s11_evals.ps1`、`tb_summary.py`（曲线首 / 末 / 末 1/4 / 最高）。日志 `outputs\bench\s11_*.log`、`outputs\s11\*.log`；checkpoint 由各训练日志里的 run 名指向（`logs\skrl\tello_defend\2026-10-17_{12-40-52,12-47-15,12-53-23,12-59-28,13-03-11,13-06-46}_ippo_torch_tello_defend_ippo`），导出 `outputs\exports\s11_{base,std037}_seed{1,2,3}.pt`。

## 12. Defend 行为克隆热启动（2026-10-23，接第 11 节；训练机换到 Linux 盒子）

训练从这一节起全部跑在远程 Linux 机 `<user>@<remote-linux-box>`（RTX 5080，驱动 580.178，`source ~/isaac/activate.sh`，仓库 `~/Action_Space` 分支 `bc-warm-start`）。
它跑 Defend IPPO 300 轮只要 **61 秒**（本机 0.68 s/轮 ≈ 3.4 分钟），没有 Vulkan 问题，也没有别的 GPU 任务。代码同步：本机 `git bundle create` → scp → 远程 `git pull --ff-only /tmp/x.bundle main`。长任务放 tmux，日志在 `simulation_infra/outputs/<dir>/`。

### 12.1 动机

第 11 节 6 条曲线按 50 轮分段看：出击的 seed 1 在 150-200 轮已平台（146 → 95 → 118），不出击的种子 300 轮全程为负且无上升趋势。
加轮数救不回不出击的种子，所以改走 chase 热启动：chase 在易设置下 100% / 0%，任务可解，缺的只是让网络一开始就"知道要追"。

### 12.2 工具

- `tasks/direct/tello_defend/chase.py`：`chase_actions` 从 evaluate.py 抽出来共用。
- `scripts/bc_warm_start.py`：用 chase + 高斯噪声（DART，σ=0.3）跑 512 envs × 200 步采样，标签是干净的 chase 动作。拟合观测标准化器、策略均值，
  以及 **value 头**（蒙特卡洛回报经 value 标准化器；`models.separate: False` 是共享主干，两头一起拟合），策略 log_std 置 -1.0（std 0.37）。
  输出是标准 skrl IPPO 检查点，`skrl/train.py --checkpoint` 和 `evaluate.py --checkpoint` 直接可读。

### 12.3 发现

- **BC 检查点本身**（v1，只拟合策略；易设置、波次关、200 回合）：**96.2% 拦截 / 4.0% 突破**，用时 3.1 s。不做任何 RL 已超过第 11 节最好的 IPPO（79.2% / 39.0%）。
- **v1 续训被 PPO 拆掉**：起点回报 +132（两机之和，任何纯 IPPO 都没到过的起点），之后单调下滑到 -38，300 轮后评估 19.5% / 93.5%（= zero）。
  原因是 value 头没训练过（起始 value loss 1.1），共享主干让它的梯度直接搅坏策略。v2 在 BC 阶段同时拟合 value 头（见 12.4）。
- **`train.py --checkpoint` 续训与 `--seed` 无关**：从任何检查点（BC 的、纯 IPPO 的、去掉 optimizer 状态的）续训，seed 1 和 2 的权重 hash、回报曲线逐位相同；
  不带 checkpoint 时 seed 正常起效。原因未查。绕法：在 `bc_warm_start.py --seed` 上变化得到不同起点，每个起点续训一次即可。
- 流水线脚本里 `set -f`（防 `[0.2,0.4]` 被 glob）会连 `ls logs/.../*ippo*` 一起关掉，取训练目录要用 `find`。
- 默认难度基线（远程，波次关，200 回合）：chase **88.8% / 22.0%**，zero 4.0% / 99.5%。第 9 节在本机测的 86.5% / 26% 同一量级。

### 12.4 为什么 PPO 会把一个 96% 的策略拆掉（远程，2026-10-23 18:20-19:00）

每个 300 轮训练在远程只要 1 分钟，所以用一组网格排查。协议同 10.1（易设置，评估 200 回合、波次关），`outputs/grid*/timeline.log` 有每条的原始数字。

**排除掉的解释**（续训后全部塌到 11-14% / 95-98%，训练回报一律单调下滑）：

| 改动 | BC 起点 | 续训 300 轮后 |
|---|---|---|
| v2：BC 阶段同时拟合 value 头（共享主干，系数 0.5），3 个 seed | 73-80% / 23-30%（value 拟合拖低策略质量） | 10-14% / 97-98% |
| 同上 + 学习率 1e-4、KL 阈值 0.008 | | 11% / 98.5% |
| B：chase 目标缩到 0.7（均值离开 ±1 裁剪边界） | 84.3% / 20.4% | 11.2% / 97.5% |
| C：策略 std 0.2 | 87.5% / 13.5% | 11.0% / 97.5% |
| D：B + C | 84.3% / 20.4% | 12.0% / 97.5% |
| E：D + value 系数 0.1 | 76.2% / 30.5% | 12.5% / 95.0% |
| G/H/J：训练时 `env.respawn_waves=False`（清场即回合结束），BC v1 / D / B 起点 | | 16% / 9% / 14% |
| 对照：纯 IPPO，waves 关，seed 1/2/3 | | 41.0% / 49.8% / 7.0% |

**决定性的测量**（`evaluate.py` 新增每机回报、回合长度、指令模长、`--stochastic`；易设置，200 回合）：

| 策略 | waves 关：拦截 / 突破 / 每机回报 / 回合长 | waves 开：拦截 / 突破 / 每机回报 |
|---|---|---|
| chase | 100% / 0% / **+80** / 3.6 s | 439% / 0.4% / +371 |
| BC v1 均值 | 96.2% / 4.0% / +77 / 4.2 s | 343% / 19% / +284 |
| BC v1 采样（std 0.37，训练时的动作方式） | 96.8% / 4.0% / +78 / 4.5 s | 309% / 27% / +251 |
| zero | 23.5% / 93% / **-12** / 8.1 s | 20% / 99.5% / -16 |
| G（BC 起点 waves 关续训 300 轮后） | 16% / 93.5% / -25 和 -46 / 8.1 s，**指令模长 2.68** | |

结论：(1) 奖励函数没问题，追击比不动高 90 分，waves 开着也是；(2) 采样噪声没问题，采样策略和均值一样好；(3) 塌掉的策略**不是学会了不动，而是均值跑飞到 ±1 之外**（前三通道模长 2.68，裁剪后等于沿某个固定方向全速飞）。
策略输出层是线性的（`output: ACTIONS`），skrl 把采样动作裁剪到 ±1 再用裁剪后的值算 log_prob，均值一旦越界梯度就失真，而 BC 起点的均值本来就贴着边界。这解释了为什么降学习率、缩 std、缩目标、拟合 critic、改 waves 都只是放慢下滑而不能阻止。
下一组（12.5）测 `agent.models.policy.output='tanh(ACTIONS)'`（均值有界）和 MAPPO。

### 12.5 杂项发现

- 学习率 0 的对照（A）在第 33 轮出现 NaN 权重：lr=0 时 Adam 的 0 × NaN 梯度仍然污染权重，说明有 NaN 梯度出现过；别的 300 轮训练没出 NaN，暂未深究。
- tmux 的 `has-session -t grid` 按**前缀**匹配，会被同机的 `grid2` 匹配上，等待循环永远不退出；会话名不要互为前缀。
- `train.py` 续训第一轮记录的 "Total reward" 偏高（+136，纯 IPPO 从零开始也是），因为第一个记录窗口里先结束的都是快速清场的短回合，不是真实水平。
- 本机 `Z:\Action_Space` 的 main 在 18:29 被另一个会话 `filter-branch` 改写、18:40 合并了 "team2 template" 历史，hash 全变。与远程分支同步改用 `git bundle` + `git cherry-pick`，远程需先 `git config user.email/name`。

### 12.6 tanh 有界输出 + MAPPO（grid3，19:05-19:18）

`agent.models.policy.output='tanh(ACTIONS)'`（train.py 走 Hydra，引号必须原样传到 Hydra；bash 里写 `"agent.models.policy.output='tanh(ACTIONS)'"`，不能经 `eval`）。waves 关，易设置，300 轮。

| 运行 | 起点 | 终点 拦截 / 突破 | 指令模长 | 回报四段 |
|---|---|---|---|---|
| K1 IPPO tanh，BC 起点（BC 本身 90.2% / 10.5%） | +69 | **32.8% / 85.5%** | 1.42 | 17, -55, -17, -7, **49**（最后一段回升） |
| P1 IPPO tanh，从零 | | 17.8% / 95.0% | 0.94 | 1, -68, -62, -58 |
| N1 MAPPO tanh，BC 起点（BC 93.0% / 7.5%） | +74 | 13.5% / 96.5% | 1.36 | 16, -61, -29, -43 |
| M1 MAPPO tanh，从零 | | 3.0% / 100% | 1.66 | -24, -70, -77, -73 |
| L1/L2 MAPPO 线性输出，从零 | | 16.8% / 94.5%，35.0% / 87.0% | **2.95 / 2.31** | 均值同样跑飞 |
| 对照：第 11 节式"出击"的纯 IPPO seed 1（63% / 48.5%） | | | **2.50** | 也是跑飞，只是饱和方向碰巧对 |

结论：均值跑飞是所有 Defend 训练（IPPO、MAPPO、从零、BC 起点）的通病，tanh 能限住它，但 BC 起点仍先跌后慢升，所以跑飞是症状不是根因。
MAPPO 不比 IPPO 好。下一组（12.7）拆开策略 / value 网络（`agent.models.separate=True`）并在 BC 阶段把 critic 一起拟合，另测冻结观测标准化器。

### 12.7 实机规格落地：3 防守 vs 2 攻击，脚本教师 + DAgger 克隆（2026-10-23 19:40-19:47，远程）

实机规格（19:40 公布）：**3 架 defender、2 架 attacker**，允许直接把 attacker 撞下来，要求"拦截失败就派新机继续"。
对应改动（commit 039ac04）：`tello_defend_env_cfg.py` `NUM_DEFENDERS = 3`（观测维度不变 = 9 + 7×2 = 23；`target_mode="assigned"` 在 3 > 2 时自动退回 nearest）；捕获半径保持 0.35 m 当作撞击接触。

**脚本教师**（`tasks/direct/tello_defend/chase.py`，`evaluate.py --baseline <name>`）：
- `smart`：一对一分配（总距离最小的排列）+ 领先追击（解 |r + v_a t| = v_max t 的拦截点，v_max = 1 m/s；无解退回纯追击）。2v2 默认难度 **100% / 0%**（chase 88.8% / 22%）。但分配用到另一架防守机的距离，单机观测里没有，**克隆不出来**（离线 BC 只有 32-47%）。
- `smart2`：分配只用自身观测：按攻击者绕基地的方位角排序，defender_i 取第 i 个；多出来的防守机当**守门员**，追离基地最近的攻击者（= "拦截失败就有新机顶上"）。两机看到的攻击者一样，排序一致，不需要通信。

**DAgger**（`bc_warm_start.py --dagger_rounds K`）：round0 用教师 + 噪声采样；之后每轮用学生均值动作跑环境、教师标注、累加数据、重拟合标准化器和策略。
0.35 m 捕获半径下纯离线 BC 的复合误差太大，DAgger 两轮后 holdout mse 从 0.0146 降到 0.0066，方向余弦从 0.974 到 **0.992**（54 万样本/机）。

| 3v2，默认难度，200 回合 | waves 关：拦截 / 突破 / 每机回报 / 用时 | waves 开：拦截 / 突破 |
|---|---|---|
| zero | 6.2% / 100% / -12.6 | |
| chase | 96.5% / 7.0% / +78.2 / 3.5 s | |
| **smart2 教师** | **100% / 0%** / +83.1 / 3.7 s | 411% / 0.8% |
| **DAgger 克隆（3 机各一网络，均值动作）** | **98.8% / 1.5%** / +85.3 / 3.9 s | 355% / 14.5% |
| 同上，采样动作（std 0.37） | **100% / 0%** / +85.8 / 4.2 s | |

| **DAgger 4 轮（`--steps 800 --dagger_rounds 4`，192 万样本/机，余弦 0.995）** | **100% / 0%** / +81.9 / 3.7 s；采样 99.8% / 0.5% | **404.7% / 1.2%**（≈ 教师） |

4 轮版本已和教师持平，**仿真里的目标达成**。交付在 `simulation_infra/policies/defend_3v2_smart2_dagger4/`：skrl 检查点 `agent_skrl_ippo.pt` + 三机 TorchScript `defender_{0,1,2}.pt`（`export_skrl_policy.py --agent`）+ 克隆日志，README 里有评估命令。
部署仍需桥接层把世界系速度按 yaw 旋转再发 `rc`（11.1 节）。

### 12.8 撞停规则：捕获半径收紧到接触级（2026-10-23 20:40，本机）

主办方要求攻击机必须被物理撞停。把捕获半径从 0.35 m 收到接触级重评（3v2，默认难度，波次关，200 回合）：

| 捕获半径 | smart2 教师 | DAgger 克隆 | chase |
|---|---|---|---|
| 0.35 m（训练值） | 100% / 0% | 100% / 0% | 96.5% / 7.0% |
| 0.20 m | 100% / 0% | 100% / 0% | 89.2% / 20.5% |
| 0.15 m | 100% / 0% | 99.0% / 1.0% | 83.5% / 28.5% |

领先追击的末端精度在仿真里够用，克隆在 0.15 m 只掉 1 个点；chase 随半径收紧明显掉。
仿真里缺的是：撞击后防守机应耗尽（现在还活着）、守门员应待命而非一直追、拦截点应偏上 0.15 m 避开下洗、接触相对速度要够。这些要改环境和教师后重新克隆（见 12.7 建议）。

### 12.9 撞停规则下重新克隆（2026-10-23 21:55-22:00，远程）

环境改动（commit 之前一条）：`defender_spent_on_capture=True`（撞到就耗尽，指令置零、不能再捕获，新波次/新回合恢复）、`outcome_credit` 记每架防守机的捕获数、`evaluate.py` 打印 `captures/episode`。
教师 smart2：守门员在基地上方 0.5 m 待命，只有攻击者进入 0.6×spawn_radius 才出击追离基地最近的；拦截点偏上 0.15 m。捕获半径 0.20 m。

| 3v2，默认难度，耗尽规则，200 回合 | 拦截 / 突破 | 每回合捕获 d0 / d1 / 守门员 |
|---|---|---|
| chase，0.20 m | 86.8% / 23.5% | 0.56 / 0.58 / 0.61（三机抢同一目标，耗尽后无人补位） |
| smart2 教师，0.20 m | **100% / 0%** | 1.00 / 0.99 / 0.01 |
| 教师，0.20 m，waves 开 | 330% / 1.6% | 3.14 / 3.01 / 0.48 |
| **DAgger 4 轮克隆，0.20 m** | **99.5% / 0.5%** | 0.98 / 0.97 / 0.04 |
| 克隆，0.15 m | 93.2% / 11.5% | 0.90 / 0.82 / **0.14**（首撞失手时守门员补上） |
| 克隆，0.20 m，waves 开 | 332% / 1.6% | 3.16 / 3.05 / 0.44 |

交付 `simulation_infra/policies/defend_3v2_ram_dagger4/`。未定项：速度。仿真防守 1.0 m/s、攻击 0.4-0.9 m/s 是 kit 假设；主办方真机日志里水平 rc 上限 15（fast 35），实测 rc 10-19 只有 0.16 m/s 中位；社区实测 rc 100 约 2 m/s。几何只看速度比，精度看绝对速度（捕获半径、延迟、定位误差都是绝对量）。现场测 rc→m/s 后按实测值重克隆。

### 12.10 防守机互撞：队友观测、排斥与高度分层（2026-10-23 22:30 - 10-24 00:15，远程）

问题：观测里没有队友，策略不可能避让；环境也没有互撞统计。加了三样东西：
- 观测末尾追加每个队友 4 维 [相对位置 xyz, active 标志]（23 → 31 维；active=0 表示已撞击耗尽，真机上即已落地、失联或 ToF 高度 < 30 cm），教师和克隆对 inactive 队友不做任何规则。
- `outcome_min_sep`：每回合防守机两两最小距离（忽略包含耗尽机的机对），`evaluate.py` 打印 p5 / 中位数和 < 0.30 m 的回合比例。
- 教师 smart2：队友排斥（`teacher_sep_radius / gain / vertical`）；"最后一架只由更近的去撞"的让路开关做过但默认关（不连续，克隆学得很差：余弦 0.95，拦截 96.6%）。

**出生点**：原来三架在 2.5 m 圆上独立随机角、高度 1-2 m 随机，两架一出生就在 0.3 m 内约 6%，和教师的互撞率一致 → 改为 120° 均匀 ±0.3 rad，高度按编号 1.0 / 1.5 / 2.0 m（真机也这样摆）。

**排斥强度扫描**（教师，1000 回合，0.20 m）：水平排斥越强互撞越少但拦截掉得快：0.8/1.5 → 100% / 0%、近距 2.5%；0.9/2.0 → 100% / 0.1%、0.4%；1.0/2.0 → 99.0%；1.2/3.0 → 93.3%；1.5/4.0 → 86.5%、0%。
原因：两架攻击机从相近方位进来时，水平排斥把追击机推离目标。

**高度分层**（分层出生 + 偏垂直排斥：编号小往下、编号大往上，离地 < 0.6 m 改为向上；巡航高度偏置 ±0.2 m 测过略有害已关）：

| 配置 | 拦截 / 突破 | 近距 (<0.3 m) | p5 间距 |
|---|---|---|---|
| 排斥 0.9/2.0、水平、随机出生（12.9 版） | 99.7% / 0.5% | 0.5% | 0.45 m |
| 分层出生 + 偏垂直 0.9/2.0 | 100% / 0.1% | 0.5% | 0.41 m |
| 分层出生 + 偏垂直 0.9/2.0，无巡航偏置 | 100% / 0.0% | 0.2% | 0.41 m |
| **分层出生 + 偏垂直 1.0/2.5，无巡航偏置（定稿默认）** | **100% / 0.1%** | **0.1%** | **0.52 m** |
| 分层出生 + 水平 0.9/2.0 | 99.9% / 0.2% | 0.0% | 0.49 m |
| 对照 chase | 93.8% / 12.0% | 26.2% | 0.09 m |

克隆（31 维，10 轮 DAgger，排斥 0.9/2.0 水平，未分层）：98.5% / 2.2%、近距 0.9%——排斥的代价转嫁到克隆上。定稿配置的克隆在跑（12.11）。
其它经验：3 个 10 轮 DAgger 并行会 CUDA OOM（每个 5-9 GB），最多 2 个；5 个 Kit 同开会 "Failed to create simulation view backend"；`git am` 在远程偶尔卡住，改用 scp 文件 + 本地 commit 同步。

### 12.11 定稿克隆（2026-10-24 00:18-00:36，远程）→ `policies/defend_3v2_ram6_s2/`

定稿教师（分层出生 1.0/1.5/2.0 m、偏垂直排斥 1.0/2.5、无巡航偏置、守门员待命 1.0 m）1000 回合：**99.9% / 0.1%**，近距 0.3%，p5 间距 0.51 m。
10 轮 DAgger、seed 2（seed 1 与它并行时 CUDA OOM；两个 10 轮并行是上限，评估别再叠）：拟合 holdout mse 0.0015 / 0.0032 / 0.0021，余弦 0.998 / 0.994 / 0.995，是目前最好的拟合。

| 克隆 ram6_s2 | 拦截 / 突破 | 近距 (<0.3 m) | p5 间距 |
|---|---|---|---|
| 0.20 m，1000 回合 | **99.6% / 0.5%** | **0.3%** | 0.50 m |
| 0.15 m，400 回合 | 93.4% / 11.2% | 6.0% | 0.29 m |
| 对照 ram4_A（29 维、弱排斥） 0.20 m | 99.8% / 0.3% | 7.8% | 0.26 m |

取舍：为把近距回合从 7.8% 压到 0.3%、p5 间距从 0.26 m 拉到 0.50 m，拦截率付出 0.2 个点。实机上互撞是不可逆事故，这个取舍值得。
交付给部署运行器（31 维观测、inactive 标志、分离守卫 0.6/0.8 m）。未定项仍是真机速度标定（12.9）。

### 12.12 冲 100 / 0：按拦截时间分配（2026-10-24 00:50-01:00，远程）

教师的最后 0.1% 突破来自**方位角分配不看防守机在哪**：`evaluate.py --diag` 打印的 3 个突破回合全是两架攻击机方位差 14-18°、其中一架 0.8-0.9 m/s，
被派去的 defender 在基地另一侧，横穿 5 m 追不上 4.5 m 外 0.9 m/s 的攻击者。
现在观测里有队友位置，改为 `teacher_assign="time"`：每架机用自己的观测重建 3×2 的领先追击拦截时间表，取总时间最小的分配（耗尽机、已落攻击者代价 1e3），
没拿到任务的那架当守门员。输入相同、算法确定，三架机不用通信就得到同一个分配。

| 教师，2000 回合，0.20 m | 拦截 / 突破 | 每回合捕获 d0 / d1 / d2 | 拦截时间 | 最近间距 min / p5 | 近距回合 |
|---|---|---|---|---|---|
| 方位角分配 | 99.9% / 0.1% | 1.00 / 1.00 / 0.00 | 4.5 s | 0.20 / 0.50 m | 0.1% |
| **拦截时间分配（默认）** | **100.0% / 0.0%** | 0.65 / 0.68 / 0.67 | **3.1 s** | **0.63 / 1.74 m** | **0.0%** |

副作用全是正面的：守门员变成动态角色（谁来不及谁留守），三机负载均衡，间距翻三倍。克隆（10 轮 DAgger，seed 2）在跑 → `policies/defend_3v2_ram7_s2/`。

### 12.13 100 / 0 达成：分配特征进观测（2026-10-24 01:05-01:25，远程）→ `policies/defend_3v2_ram8_s2/`

直接克隆拦截时间分配的教师（ram7_s2）只有 98.2% / 3.4%、余弦 0.95：分配是 6 种排列取最小值，哪个目标归我随小扰动翻转，网络学不好离散决策。
拆开：`chase.assignment_from_obs` 从单机观测算出 [我负责的攻击者 one-hot (2), 守门员标志 (1)]，环境把它追加到观测（31 → 34 维），
网络只学连续的追击、排斥、待命；真机运行器用同一算法（numpy 移植，与 torch 参考在 400 个随机观测上逐一相等）算同一特征。

| 2000 回合，0.20 m，耗尽规则 | 拦截 / 突破 | 捕获 d0 / d1 / d2 | 拦截时间 | 间距 min / p5 | 近距 |
|---|---|---|---|---|---|
| 教师（34 维观测） | 100.0% / 0.0% | 0.66 / 0.67 / 0.68 | 3.1 s | 0.65 / 1.81 m | 0.0% |
| **克隆 ram8_s2（10 轮 DAgger，余弦 0.996-0.998）** | **100.0% / 0.0%** | 0.67 / 0.65 / 0.68 | 3.1 s | **0.65 / 1.72 m** | **0.0%** |
| 克隆，0.15 m，400 回合 | 96.0% / 8.0% | | 3.7 s | 0.07 / 0.18 m | 10.8% |
| 克隆，waves 开，400 回合 | 467% / 1.8% | 3.18 / 3.02 / 3.15 | | 0.05 / 0.36 m | 3.6% |

仿真目标（100 / 0、零互撞）达成。整条路线：chase → BC（96%）→ PPO 全部退化（约 40 次实验，根因未查）→ 脚本教师 smart/smart2 → DAgger →
3v2 + 撞停耗尽 → 队友观测 + 排斥 → 分层出生 + 偏垂直排斥 → 拦截时间分配 → 分配进观测。
仍未定：真机速度标定（12.9）、遥测沿墙镜像歧义、真机干跑。

### 12.14 错开放飞 + 守门员分环（2026-10-24 01:35-01:56，远程）→ `policies/defend_3v2_ram9_s2/`

部署会话在离线演示里发现：只有一架攻击机在场且无人耗尽时，两架守门员同时飞向基地上方同一待命点（最近 0.30 m）。仿真训练里没有这个情形（两架攻击机同时出现，捕获者即耗尽），但真机上主办方错开放飞或攻击机自己坠落就会出现。
改动：`attacker_launch_delay=[0,3]` 第二架攻击机随机延迟放飞，延迟期间 pending（不可见、不可捕获、不算突破、不触发清场）；守门员 i 在基地周围 0.8 m 环上 2πi/3 方位待命（`teacher_standby_ring`）。

| 0.20 m，耗尽规则 | 拦截 / 突破 | 间距 min / p5 | 近距 |
|---|---|---|---|
| 教师，同时放飞，1000 回合 | 100% / 0% | 0.76 / 1.81 m | 0.0% |
| 教师，错开 0-3 s，2000 回合 | 100% / 0% | 0.25 / 0.51 m | 1.3% |
| **克隆 ram9_s2，同时放飞，400 回合** | **100% / 0%** | 0.67 / 1.86 m | 0.0% |
| **克隆 ram9_s2，错开，2000 回合** | **100% / 0%** | 0.16 / 0.48 m | 2.2% |
| 克隆，waves 开，400 回合 | 359% / 0.2% | 0.06 / 0.33 m | 3.6% |

拦截在两种放飞方式下都保持 100 / 0；错开放飞下仍有 2% 的近距回合（教师 1.3%），交给运行器 0.6 m 守卫兜底。ram9_s2 为最终交付。
