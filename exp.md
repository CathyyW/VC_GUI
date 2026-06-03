
(base) PS C:\WINDOWS\system32> & "$env:LOCALAPPDATA\Android\Sdk\emulator\emulator.exe" -avd AndroidWorld -port 5554 -grpc 8554 -no-snapshot


Windows 本地停掉旧 forwarder
& "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" -s emulator-5554 shell settings put secure enabled_accessibility_services ""
& "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" -s emulator-5554 shell settings put secure accessibility_enabled 0
& "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" -s emulator-5554 shell am force-stop com.google.androidenv.accessibilityforwarder



ssh -F NUL -p 32786 -N -v -o ExitOnForwardFailure=yes -R 15554:127.0.0.1:5554 -R 15555:127.0.0.1:5555 -R 18554:127.0.0.1:8554 -L 20000:127.0.0.1:20000 root@connect.westd.seetacloud.com

ssh -F NUL -p 32786 -N -v -o ExitOnForwardFailure=yes -R 15554:127.0.0.1:5554 -R 15555:127.0.0.1:5555 -R 18554:127.0.0.1:8554 root@connect.westd.seetacloud.com


netstat -ano | findstr LISTENING | findstr 8554

tasklist /FI "PID eq 48184"

(android_world) root@autodl-container-7j2mj0bz0c-71f0dc3f:~/autodl-tmp/V-Droid# /root/autodl-tmp/android-sdk/platform-tools/adb connect 127.0.0.1:15555
connected to 127.0.0.1:15555
(android_world) root@autodl-container-7j2mj0bz0c-71f0dc3f:~/autodl-tmp/V-Droid# /root/autodl-tmp/android-sdk/platform-tools/adb devices

### 会陷入死循环 同一个动作的score一直很高


ssh -F NUL -p 18563 -N -v -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 `
  -R 15554:127.0.0.1:5554 `
  -R 15555:127.0.0.1:5555 `
  -R 18554:127.0.0.1:8554 `
  root@connect.westc.seetacloud.com

# 20000 可以直接手动开，也可以用下面的 watchdog 自动重启
ssh -F NUL -p 32786 -N -v -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 `
  -L 20000:127.0.0.1:20000 `
  root@connect.westd.seetacloud.com

# 推荐：在 Windows PowerShell 中运行 watchdog
powershell -ExecutionPolicy Bypass -File .\a11y-20000-watchdog.ps1

# 如果不想每次重启 20000 都输入密码，先在 Windows 配 SSH key
ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\autodl_vdroid" -C "vdroid-a11y"
Get-Content "$env:USERPROFILE\.ssh\autodl_vdroid.pub" | ssh -F NUL -p 32786 root@connect.westd.seetacloud.com "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"
ssh -F NUL -p 32786 -i "$env:USERPROFILE\.ssh\autodl_vdroid" -o BatchMode=yes root@connect.westd.seetacloud.com "echo key-login-ok"