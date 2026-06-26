## 模拟器部署在本地的反向SSH步骤
### windows powershell 本地启动模拟器
```powershell
 & "$env:LOCALAPPDATA\Android\Sdk\emulator\emulator.exe" -avd AndroidWorld -port 5554 -grpc 8554 -no-snapshot
```


### windows powershell 本地启动ssh隧道

```powershell
ssh -F NUL -p <PORT NUMBER> -N -v -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 `
  -R 15554:127.0.0.1:5554 `
  -R 15555:127.0.0.1:5555 `
  -R 18554:127.0.0.1:8554 `
  root@connect.westc.seetacloud.com
```

### windows powershell 本地检查连接
```powershell
netstat -ano | findstr LISTENING | findstr 8554
```
```powershell
tasklist /FI "PID eq 48184"
```
### 远程 server 检查连接
```bash
/root/autodl-tmp/android-sdk/platform-tools/adb connect 127.0.0.1:15555
```
```bash
/root/autodl-tmp/android-sdk/platform-tools/adb devices
```


### 在 Windows PowerShell 中运行 watchdog
```powershell
powershell -ExecutionPolicy Bypass -File .\a11y-20000-watchdog.ps1
```


#### 如果不想每次重启 20000 都输入密码，先在 Windows 配 SSH key
ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\autodl_vdroid" -C "vdroid-a11y"
Get-Content "$env:USERPROFILE\.ssh\autodl_vdroid.pub" | ssh -F NUL -p 32786 root@connect.westd.seetacloud.com "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"
ssh -F NUL -p 32786 -i "$env:USERPROFILE\.ssh\autodl_vdroid" -o BatchMode=yes root@connect.westd.seetacloud.com "echo key-login-ok"
