# Benchmark one or more task families on this Windows machine: short train -> play/score -> export ->
# video, each step as its own low-priority (BelowNormal) Isaac Sim process, timed and logged under
# outputs\bench\. Usage (any shell, no activation needed):
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\bench.ps1 -Family hover,waypoint
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\bench.ps1 -Family all
#
# Families: hover, waypoint (skrl PPO), defend (skrl IPPO/MAPPO), approach (SB3), room_single, room (SB3 central).
# Every step appends a line to outputs\bench\timeline.tsv; stdout/stderr go to outputs\bench\<step>.log/.err.
# Summarise with: python scripts\windows\bench_report.py
param(
    [string[]]$Family = @("all"),
    [int]$MinFreeVramMiB = 4096,
    # defend_s10 only (WINDOWS_TRAINING.md section 10): step-name prefix, extra train overrides
    # (space-separated, e.g. "env.respawn_waves=False env.closing_mode=own_motion"), iterations
    [string]$Tag = "s10",
    [string]$Extra = "",
    [int]$Iters = 300
)
$ErrorActionPreference = "Continue"
. Z:\isaac\activate_isaac.ps1
$PY = "Z:\isaac\env_isaaclab\Scripts\python.exe"
$ROOT = (Get-Location).Path
$BENCH = Join-Path $ROOT "outputs\bench"
New-Item -ItemType Directory -Force $BENCH | Out-Null
$TIMELINE = Join-Path $BENCH "timeline.tsv"
$TAB = [string][char]9
if (-not (Test-Path $TIMELINE)) {
    (@("step", "start", "wall_s", "exit", "free_vram_mib_before", "command") -join $TAB) | Set-Content -Encoding utf8 $TIMELINE
}

function Free-VRAM { [int]((nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) | Select-Object -First 1) }

function Run-Step([string]$Name, [string[]]$PyArgs, [string]$Priority = "BelowNormal") {
    $log = Join-Path $BENCH "$Name.log"
    $err = Join-Path $BENCH "$Name.err"
    $free = Free-VRAM
    $cmd = "python -u " + ($PyArgs -join " ")
    $stamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    if ($free -lt $MinFreeVramMiB) {
        Write-Host "[$Name] SKIP: only $free MiB VRAM free"
        (@($Name, $stamp, "0", "SKIP", "$free", $cmd) -join $TAB) | Add-Content -Encoding utf8 $TIMELINE
        return $false
    }
    $t0 = Get-Date
    Write-Host ("[{0}] {1} start (free VRAM {2} MiB)" -f $t0.ToString("HH:mm:ss"), $Name, $free)
    Write-Host "    $cmd"
    $p = Start-Process -FilePath $PY -ArgumentList (@("-u") + $PyArgs) -WorkingDirectory $ROOT -NoNewWindow -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
    $null = $p.Handle  # cache the handle, otherwise $p.ExitCode is empty after WaitForExit (PowerShell 5.1 quirk)
    try { $p.PriorityClass = $Priority } catch { Write-Host "    (could not set priority: $_)" }
    $p.WaitForExit()
    $sec = [int]((Get-Date) - $t0).TotalSeconds
    (@($Name, $stamp, "$sec", "$($p.ExitCode)", "$free", $cmd) -join $TAB) | Add-Content -Encoding utf8 $TIMELINE
    Write-Host ("[{0}] {1} done in {2}s exit={3}" -f (Get-Date).ToString("HH:mm:ss"), $Name, $sec, $p.ExitCode)
    return ($p.ExitCode -eq 0)
}

function Newest-Run([string]$Parent) {
    Get-ChildItem $Parent -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
}
function Skrl-Ckpt-Of-Log([string]$Experiment, [string]$TrainLog) {
    # the run dir named in a train.py log ("Exact experiment name requested from command line: <stamp>"), so
    # concurrent runs cannot pick each other's checkpoint; falls back to the newest run
    $m = Select-String -Path $TrainLog -Pattern "Exact experiment name requested from command line:\s*(\S+)" | Select-Object -Last 1
    if ($m) {
        $stamp = $m.Matches[0].Groups[1].Value
        $run = Get-ChildItem (Join-Path $ROOT "logs\skrl\$Experiment") -Directory -Filter "$stamp*" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($run) {
            $c = Get-ChildItem (Join-Path $run.FullName "checkpoints\agent_*.pt") -ErrorAction SilentlyContinue |
                Sort-Object { [int]($_.BaseName -replace "agent_", "") } | Select-Object -Last 1
            if ($c) { return $c.FullName }
        }
    }
    return (Skrl-Ckpt $Experiment)
}
function Skrl-Ckpt([string]$Experiment) {
    # logs\skrl\<experiment>\<newest run>\checkpoints\agent_<max N>.pt (best_agent.pt only appears once a reward improves)
    $run = Newest-Run (Join-Path $ROOT "logs\skrl\$Experiment")
    if (-not $run) { return $null }
    $c = Get-ChildItem (Join-Path $run.FullName "checkpoints\agent_*.pt") -ErrorAction SilentlyContinue |
        Sort-Object { [int]($_.BaseName -replace "agent_", "") } | Select-Object -Last 1
    if ($c) { return $c.FullName } else { return $null }
}
function Sb3-Ckpt([string]$Task) {
    $run = Newest-Run (Join-Path $ROOT "logs\sb3\$Task")
    if ($run -and (Test-Path (Join-Path $run.FullName "model.zip"))) { return (Join-Path $run.FullName "model.zip") }
    return $null
}

function Family-SkrlSingle([string]$Tag, [string]$Task, [string]$Experiment, [int]$Iters) {
    Run-Step "${Tag}_train" @("scripts\skrl\train.py", "--task", $Task, "--headless", "--num_envs", "1024", "--max_iterations", "$Iters") | Out-Null
    $ck = Skrl-Ckpt $Experiment
    if (-not $ck) { Write-Host "[$Tag] no checkpoint found, skipping play/video"; return }
    Write-Host "[$Tag] checkpoint: $ck"
    Run-Step "${Tag}_play"  @("scripts\skrl\play.py", "--task", $Task, "--headless", "--num_envs", "16", "--checkpoint", $ck, "--steps", "400") | Out-Null
    Run-Step "${Tag}_export" @("scripts\export_skrl_policy.py", "--checkpoint", $ck, "--out", "outputs\exports\bench_$Tag.pt") | Out-Null
    # --video loads isaaclab.python.headless.rendering.kit, which sets app.vulkan=true: Vulkan crashes Kit on this
    # GPU/driver (access violation ~9 s in), so force D3D12 like the windowed runs (see WINDOWS.md)
    Run-Step "${Tag}_video" @("scripts\skrl\play.py", "--task", $Task, "--headless", "--num_envs", "4", "--checkpoint", $ck, "--video", "--video_length", "200", "--kit_args=--/app/vulkan=false") | Out-Null
}

function Family-Defend {
    Run-Step "defend_ippo_train" @("scripts\skrl\train.py", "--task", "AS-Defend-v0", "--algorithm", "IPPO", "--headless", "--num_envs", "512", "--max_iterations", "30") | Out-Null
    $ck = Skrl-Ckpt "tello_defend"
    Run-Step "defend_eval_zero"  @("scripts\evaluate.py", "--task", "AS-Defend-v0", "--headless", "--baseline", "zero", "--episodes", "20") | Out-Null
    Run-Step "defend_eval_chase" @("scripts\evaluate.py", "--task", "AS-Defend-v0", "--headless", "--baseline", "chase", "--episodes", "20") | Out-Null
    if ($ck) {
        Write-Host "[defend] checkpoint: $ck"
        Run-Step "defend_eval_policy" @("scripts\evaluate.py", "--task", "AS-Defend-v0", "--headless", "--checkpoint", $ck, "--algorithm", "IPPO", "--episodes", "20") | Out-Null
        Run-Step "defend_export" @("scripts\export_skrl_policy.py", "--checkpoint", $ck, "--agent", "defender_0", "--out", "outputs\exports\bench_defend_ippo_defender_0.pt") | Out-Null
        Run-Step "defend_video_policy" @("scripts\evaluate.py", "--task", "AS-Defend-v0", "--headless", "--checkpoint", $ck, "--algorithm", "IPPO", "--episodes", "2", "--num_envs", "4", "--video", "--video_length", "200", "--video_dir", "outputs\videos\bench_defend_eval", "--kit_args=--/app/vulkan=false") | Out-Null
    }
    Run-Step "defend_video_clip" @("scripts\record_clip.py", "--frames", "100", "--out", "outputs\videos\bench_defend_chase.mp4", "--frame_dir", "outputs\frames_clip") | Out-Null
    Run-Step "defend_mappo_train" @("scripts\skrl\train.py", "--task", "AS-Defend-v0", "--algorithm", "MAPPO", "--headless", "--num_envs", "512", "--max_iterations", "15") | Out-Null
}

function Family-DefendBaselines {
    # The scripted baselines, 100 episodes each, with the default wave respawn and with it off (README numbers)
    foreach ($b in @("zero", "chase")) {
        Run-Step "defend_base_${b}_waves" @("scripts\evaluate.py", "--task", "AS-Defend-v0", "--headless", "--baseline", $b, "--episodes", "100") | Out-Null
        Run-Step "defend_base_${b}_nowaves" @("scripts\evaluate.py", "--task", "AS-Defend-v0", "--headless", "--baseline", $b, "--episodes", "100", "env.respawn_waves=False") | Out-Null
    }
}

function Family-DefendStage0([int]$Iters = 300, [string]$Algo = "IPPO") {
    # README's curriculum stage 0 (bigger capture radius, closer and slower attackers): train, then score the
    # checkpoint and the chase baseline under the SAME overrides, then export defender_0
    $easy = @("env.capture_radius=1.0", "env.spawn_radius=4.0", "env.attacker_speed=[0.2,0.4]")
    $tag = "defend_stage0_" + $Algo.ToLower() + "_$Iters"
    Run-Step "${tag}_train" (@("scripts\skrl\train.py", "--task", "AS-Defend-v0", "--algorithm", $Algo, "--headless", "--num_envs", "512", "--max_iterations", "$Iters") + $easy) | Out-Null
    $ck = Skrl-Ckpt "tello_defend"
    if (-not $ck) { Write-Host "[$tag] no checkpoint found, skipping"; return }
    Write-Host "[$tag] checkpoint: $ck"
    Run-Step "${tag}_eval_policy" (@("scripts\evaluate.py", "--task", "AS-Defend-v0", "--headless", "--checkpoint", $ck, "--algorithm", $Algo, "--episodes", "50") + $easy) | Out-Null
    Run-Step "${tag}_eval_chase"  (@("scripts\evaluate.py", "--task", "AS-Defend-v0", "--headless", "--baseline", "chase", "--episodes", "50") + $easy) | Out-Null
    Run-Step "${tag}_eval_zero"   (@("scripts\evaluate.py", "--task", "AS-Defend-v0", "--headless", "--baseline", "zero", "--episodes", "50") + $easy) | Out-Null
    Run-Step "${tag}_export" @("scripts\export_skrl_policy.py", "--checkpoint", $ck, "--agent", "defender_0", "--out", "outputs\exports\$tag.pt") | Out-Null
}

function Family-DefendS10([string]$Tag, [string]$Extra, [int]$Iters = 300) {
    # WINDOWS_TRAINING.md section 10 protocol: IPPO on the easy settings plus ONE change ($Extra, space-separated
    # Hydra overrides), then policy / zero / chase scored with waves OFF (README-comparable), 200 episodes each,
    # under the same env overrides, then export defender_0. Usage:
    #   bench.ps1 -Family defend_s10 -Tag s10_c1 -Extra "env.respawn_waves=False" [-Iters 600]
    $easy = @("env.capture_radius=1.0", "env.spawn_radius=4.0", "env.attacker_speed=[0.2,0.4]")
    # not "$extra": PowerShell variable names are case-insensitive, so that would assign the split list back to the
    # [string] parameter $Extra and re-join it into ONE token (seen: "--seed 1" forwarded to evaluate.py, exit 2)
    $extraArgs = @($Extra -split "\s+" | Where-Object { $_ })
    # evaluate.py takes env.* and agent.models.* (the latter so the checkpoint's model is rebuilt the same way)
    $extraEnv = @($extraArgs | Where-Object { ($_ -match '^(env|agent[.]models)[.]') -and ($_ -notmatch '^env[.]respawn_waves=') })
    $evalArgs = $easy + @("env.respawn_waves=False") + $extraEnv
    Run-Step "${Tag}_train" (@("scripts\skrl\train.py", "--task", "AS-Defend-v0", "--algorithm", "IPPO", "--headless", "--num_envs", "512", "--max_iterations", "$Iters") + $easy + $extraArgs) | Out-Null
    $ck = Skrl-Ckpt-Of-Log "tello_defend" (Join-Path $BENCH "${Tag}_train.log")
    if (-not $ck) { Write-Host "[$Tag] no checkpoint found, skipping"; return }
    Write-Host "[$Tag] checkpoint: $ck"
    Run-Step "${Tag}_eval_policy" (@("scripts\evaluate.py", "--task", "AS-Defend-v0", "--headless", "--checkpoint", $ck, "--algorithm", "IPPO", "--episodes", "200") + $evalArgs) | Out-Null
    Run-Step "${Tag}_eval_zero"   (@("scripts\evaluate.py", "--task", "AS-Defend-v0", "--headless", "--baseline", "zero", "--episodes", "200") + $evalArgs) | Out-Null
    Run-Step "${Tag}_eval_chase"  (@("scripts\evaluate.py", "--task", "AS-Defend-v0", "--headless", "--baseline", "chase", "--episodes", "200") + $evalArgs) | Out-Null
    Run-Step "${Tag}_export" @("scripts\export_skrl_policy.py", "--checkpoint", $ck, "--agent", "defender_0", "--out", "outputs\exports\$Tag.pt") | Out-Null
}

function Family-Approach {
    $envArgs = @("env.spawn_azimuth_range=0.0", "env.rew_scale_separation=-20.0", "env.rew_violation=-100.0")
    Run-Step "approach_train" (@("scripts\sb3\train.py", "--task", "AS-Tello-Approach-v0", "--headless", "--num_envs", "512", "--max_iterations", "20", "agent.learning_rate=1e-3") + $envArgs) | Out-Null
    $ck = Sb3-Ckpt "AS-Tello-Approach-v0"
    if (-not $ck) { Write-Host "[approach] no checkpoint found, skipping"; return }
    Write-Host "[approach] checkpoint: $ck"
    Run-Step "approach_play" (@("scripts\sb3\play.py", "--task", "AS-Tello-Approach-v0", "--headless", "--num_envs", "64", "--episodes", "64", "--checkpoint", $ck) + $envArgs) | Out-Null
    Run-Step "approach_export" @("scripts\export_sb3_policy.py", "--checkpoint", $ck, "--out", "outputs\exports\bench_approach.pt") | Out-Null
    Run-Step "approach_video" @("scripts\record_approach.py", "--pilot", "policy", "--checkpoint", $ck, "--spawn_azimuth_range", "0.0", "--set", "rew_scale_separation=-20.0", "rew_violation=-100.0", "--episodes", "1", "--out", "outputs\videos\bench_approach.mp4", "--frame_dir", "outputs\frames_approach") | Out-Null
}

function Family-ApproachScale {
    # How much the per-iteration cost is fixed overhead: the same 6 iterations at 512 / 2048 envs, and 512 envs
    # at Normal priority (is the slowness contention with the other GPU job, or Windows itself?)
    $envArgs = @("env.spawn_azimuth_range=0.0", "env.rew_scale_separation=-20.0", "env.rew_violation=-100.0")
    Run-Step "approach_scale_2048" (@("scripts\sb3\train.py", "--task", "AS-Tello-Approach-v0", "--headless", "--num_envs", "2048", "--max_iterations", "6", "agent.learning_rate=1e-3") + $envArgs) | Out-Null
    Run-Step "approach_scale_512_normalprio" (@("scripts\sb3\train.py", "--task", "AS-Tello-Approach-v0", "--headless", "--num_envs", "512", "--max_iterations", "6", "agent.learning_rate=1e-3") + $envArgs) "Normal" | Out-Null
}

function Family-Room([string]$Tag, [string]$Task, [int]$Iters, [int]$PlayEpisodes) {
    if ($Tag -eq "room_single") {
        Run-Step "${Tag}_check" @("scripts\check_room_env.py", "--headless", "--task", $Task) | Out-Null
    }
    Run-Step "${Tag}_train" @("scripts\sb3\train.py", "--task", $Task, "--headless", "--num_envs", "512", "--max_iterations", "$Iters") | Out-Null
    $ck = Sb3-Ckpt $Task
    if (-not $ck) { Write-Host "[$Tag] no checkpoint found, skipping"; return }
    Write-Host "[$Tag] checkpoint: $ck"
    Run-Step "${Tag}_play"   @("scripts\sb3\play.py", "--task", $Task, "--headless", "--num_envs", "64", "--episodes", "$PlayEpisodes", "--checkpoint", $ck) | Out-Null
    Run-Step "${Tag}_export" @("scripts\export_sb3_policy.py", "--checkpoint", $ck, "--out", "outputs\exports\bench_$Tag.pt") | Out-Null
    Run-Step "${Tag}_video"  @("scripts\record_room.py", "--task", $Task, "--checkpoint", $ck, "--episodes", "1", "--out", "outputs\videos\bench_$Tag.mp4", "--frame_dir", "outputs\frames_room") | Out-Null
}

# "-Family hover,waypoint" arrives as one string when launched with powershell -File: split on commas
$Family = @($Family | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$fams = if ($Family -contains "all") { @("hover", "waypoint", "defend", "approach", "room_single", "room") } else { $Family }
Write-Host ("== bench start {0}  families: {1}" -f (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"), ($fams -join ","))
foreach ($f in $fams) {
    switch ($f) {
        "hover"       { Family-SkrlSingle "hover" "AS-Tello-Hover-v0" "tello_hover" 40 }
        "waypoint"    { Family-SkrlSingle "waypoint" "AS-Tello-Waypoint-v0" "tello_waypoint" 40 }
        "defend"      { Family-Defend }
        "approach"    { Family-Approach }
        "approach_scale" { Family-ApproachScale }
        "defend_baselines" { Family-DefendBaselines }
        "defend_stage0"    { Family-DefendStage0 300 "IPPO" }
        "defend_stage0_600" { Family-DefendStage0 600 "IPPO" }
        "defend_stage0_mappo" { Family-DefendStage0 300 "MAPPO" }
        "defend_s10"       { Family-DefendS10 $Tag $Extra $Iters }
        "room_single" { Family-Room "room_single" "AS-Tello-Room-Single-v0" 6 64 }
        "room"        { Family-Room "room" "AS-Tello-Room-v0" 6 16 }
        default       { Write-Host "unknown family: $f" }
    }
}
Write-Host ("== bench done {0}" -f (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"))
