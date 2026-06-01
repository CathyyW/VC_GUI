
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

### 动态ui 一直不采取新动作
### 对语义理解不够好？特定文件名字 没有掌握
### 会陷入死循环 wait的分数一直很高
