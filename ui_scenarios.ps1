param(
  [Parameter(Mandatory = $true)]
  [ValidateSet("initial", "2x2", "2x2_custom", "controls", "animation")]
  [string]$Scenario
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms

Add-Type @'
using System;
using System.Text;
using System.Runtime.InteropServices;

public static class Win32 {
  public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);
  [DllImport("user32.dll")] public static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int maxCount);
  [DllImport("user32.dll")] public static extern int GetWindowTextLength(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int X, int Y);
  [DllImport("user32.dll")] public static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint dwData, UIntPtr dwExtraInfo);
  public const uint MOUSEEVENTF_LEFTDOWN = 0x0002;
  public const uint MOUSEEVENTF_LEFTUP = 0x0004;
  public const uint MOUSEEVENTF_WHEEL = 0x0800;
}

public struct RECT {
  public int Left;
  public int Top;
  public int Right;
  public int Bottom;
}
'@

$artifactDir = Join-Path (Get-Location) "ui_test_artifacts"
New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null

$windowTitle = "TransformLab: Animated Linear Transformation Visualizer"

function Get-WindowHandle {
  param([string]$Title = $windowTitle)
  $deadline = (Get-Date).AddSeconds(20)
  while ((Get-Date) -lt $deadline) {
    $script:target = [IntPtr]::Zero
    $callback = [Win32+EnumWindowsProc]{
      param($hWnd, $lParam)
      $len = [Win32]::GetWindowTextLength($hWnd)
      if ($len -gt 0 -and [Win32]::IsWindowVisible($hWnd)) {
        $sb = New-Object System.Text.StringBuilder ($len + 1)
        [void][Win32]::GetWindowText($hWnd, $sb, $sb.Capacity)
        if ($sb.ToString() -eq $Title) {
          $script:target = $hWnd
          return $false
        }
      }
      return $true
    }
    [Win32]::EnumWindows($callback, [IntPtr]::Zero) | Out-Null
    if ($script:target -ne [IntPtr]::Zero) {
      return $script:target
    }
    Start-Sleep -Milliseconds 250
  }
  throw "App window not found."
}

function Capture-Window {
  param([IntPtr]$Handle, [string]$Path)
  $rect = New-Object RECT
  [void][Win32]::GetWindowRect($Handle, [ref]$rect)
  $width = $rect.Right - $rect.Left
  $height = $rect.Bottom - $rect.Top
  $bmp = New-Object System.Drawing.Bitmap $width, $height
  $graphics = [System.Drawing.Graphics]::FromImage($bmp)
  $graphics.CopyFromScreen(
    (New-Object System.Drawing.Point($rect.Left, $rect.Top)),
    [System.Drawing.Point]::Empty,
    (New-Object System.Drawing.Size($width, $height))
  )
  $bmp.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
  $graphics.Dispose()
  $bmp.Dispose()
}

function Click-Point {
  param([int]$X, [int]$Y)
  [void][Win32]::SetCursorPos($X, $Y)
  Start-Sleep -Milliseconds 70
  [Win32]::mouse_event([Win32]::MOUSEEVENTF_LEFTDOWN, 0, 0, 0, [UIntPtr]::Zero)
  Start-Sleep -Milliseconds 30
  [Win32]::mouse_event([Win32]::MOUSEEVENTF_LEFTUP, 0, 0, 0, [UIntPtr]::Zero)
}

function Find-Control {
  param(
    [System.Windows.Automation.AutomationElement]$Root,
    [string]$Name,
    [System.Windows.Automation.ControlType]$ControlType,
    [int]$Index = 0
  )
  $condition = New-Object System.Windows.Automation.AndCondition(
    (New-Object System.Windows.Automation.PropertyCondition(
      [System.Windows.Automation.AutomationElement]::NameProperty,
      $Name
    )),
    (New-Object System.Windows.Automation.PropertyCondition(
      [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
      $ControlType
    ))
  )
  $matches = $Root.FindAll([System.Windows.Automation.TreeScope]::Descendants, $condition)
  if ($matches.Count -le $Index) {
    throw "Missing control '$Name' index $Index"
  }
  return $matches[$Index]
}

function Find-InvokableControl {
  param(
    [System.Windows.Automation.AutomationElement]$Root,
    [string]$Name
  )
  foreach ($controlType in @(
    [System.Windows.Automation.ControlType]::Button,
    [System.Windows.Automation.ControlType]::CheckBox
  )) {
    try {
      return Find-Control -Root $Root -Name $Name -ControlType $controlType
    } catch {
    }
  }
  throw "Missing invokable control '$Name'"
}

function Invoke-Control {
  param([System.Windows.Automation.AutomationElement]$Element)
  $invokePattern = $null
  if ($Element.TryGetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern, [ref]$invokePattern)) {
    $invokePattern.Invoke()
    return
  }
  $togglePattern = $null
  if ($Element.TryGetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern, [ref]$togglePattern)) {
    $togglePattern.Toggle()
    return
  }
  throw "Control '$($Element.Current.Name)' is not invokable."
}

function Set-SpinnerValue {
  param([System.Windows.Automation.AutomationElement]$Spinner, [double]$Value)
  $rangePattern = $null
  if (-not $Spinner.TryGetCurrentPattern([System.Windows.Automation.RangeValuePattern]::Pattern, [ref]$rangePattern)) {
    throw "Spinner does not support range values."
  }
  $rangePattern.SetValue($Value)
}

function Get-Spinners {
  param([System.Windows.Automation.AutomationElement]$Root)
  $condition = New-Object System.Windows.Automation.PropertyCondition(
    [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
    [System.Windows.Automation.ControlType]::Spinner
  )
  return $Root.FindAll([System.Windows.Automation.TreeScope]::Descendants, $condition)
}

function Switch-To-2x2 {
  param(
    [IntPtr]$Handle,
    [System.Windows.Automation.AutomationElement]$Window
  )
  $combo = Find-Control -Root $Window -Name "Matrix Size" -ControlType ([System.Windows.Automation.ControlType]::ComboBox)
  $rect = $combo.Current.BoundingRectangle
  $centerX = [int]($rect.Left + ($rect.Width * 0.50))
  $centerY = [int]($rect.Top + ($rect.Height * 0.52))

  for ($attempt = 0; $attempt -lt 5; $attempt++) {
    Click-Point -X $centerX -Y $centerY
    Start-Sleep -Milliseconds 180
    [System.Windows.Forms.SendKeys]::SendWait("%{DOWN}")
    Start-Sleep -Milliseconds 180
    [System.Windows.Forms.SendKeys]::SendWait("{UP}{ENTER}")
    Start-Sleep -Milliseconds 500

    try {
      $group = Find-Control -Root $Window -Name "Custom 2 x 2 Matrix" -ControlType ([System.Windows.Automation.ControlType]::Group) -Index 0
      if ($null -ne $group) {
        return
      }
    } catch {
    }

    [void][Win32]::SetForegroundWindow($Handle)
    Start-Sleep -Milliseconds 150
  }

  throw "Could not switch matrix size to 2x2."
}

$process = Start-Process ".\\.venv\\Scripts\\python.exe" -ArgumentList "main.py" -PassThru
try {
  Start-Sleep -Seconds 8
  $handle = Get-WindowHandle
  [void][Win32]::SetForegroundWindow($handle)
  Start-Sleep -Milliseconds 250
  $window = [System.Windows.Automation.AutomationElement]::FromHandle($handle)

  switch ($Scenario) {
    "initial" {
      Capture-Window -Handle $handle -Path (Join-Path $artifactDir "scenario_initial.png")
      Write-Output "PASS initial"
    }
    "2x2" {
      Switch-To-2x2 -Handle $handle -Window $window
      Capture-Window -Handle $handle -Path (Join-Path $artifactDir "scenario_2x2.png")
      Write-Output "PASS 2x2"
    }
    "2x2_custom" {
      Switch-To-2x2 -Handle $handle -Window $window
      $rotation = Find-InvokableControl -Root $window -Name "Rotation"
      Invoke-Control -Element $rotation
      Start-Sleep -Milliseconds 500
      $spinners = Get-Spinners -Root $window
      Set-SpinnerValue -Spinner $spinners[0] -Value 1.4
      Set-SpinnerValue -Spinner $spinners[1] -Value 0.3
      Set-SpinnerValue -Spinner $spinners[2] -Value -0.2
      Set-SpinnerValue -Spinner $spinners[3] -Value 0.9
      $apply = Find-Control -Root $window -Name "Apply" -ControlType ([System.Windows.Automation.ControlType]::Button)
      Invoke-Control -Element $apply
      Start-Sleep -Milliseconds 500
      Capture-Window -Handle $handle -Path (Join-Path $artifactDir "scenario_2x2_custom.png")
      Write-Output "PASS 2x2_custom"
    }
    "controls" {
      $shear = Find-InvokableControl -Root $window -Name "Shear"
      Invoke-Control -Element $shear
      Start-Sleep -Milliseconds 500
      Capture-Window -Handle $handle -Path (Join-Path $artifactDir "scenario_controls_shear.png")
      $rect = New-Object RECT
      [void][Win32]::GetWindowRect($handle, [ref]$rect)
      $startX = $rect.Left + 380
      $startY = $rect.Top + 390
      $endX = $startX + 200
      $endY = $startY - 110
      Click-Point -X $startX -Y $startY
      [Win32]::mouse_event([Win32]::MOUSEEVENTF_LEFTDOWN, 0, 0, 0, [UIntPtr]::Zero)
      Start-Sleep -Milliseconds 50
      [void][Win32]::SetCursorPos($endX, $endY)
      Start-Sleep -Milliseconds 100
      [Win32]::mouse_event([Win32]::MOUSEEVENTF_LEFTUP, 0, 0, 0, [UIntPtr]::Zero)
      Start-Sleep -Milliseconds 100
      [Win32]::mouse_event([Win32]::MOUSEEVENTF_WHEEL, 0, 0, [uint32]240, [UIntPtr]::Zero)
      Start-Sleep -Milliseconds 300
      Capture-Window -Handle $handle -Path (Join-Path $artifactDir "scenario_controls_moved.png")
      $homeButton = Find-Control -Root $window -Name "Home View" -ControlType ([System.Windows.Automation.ControlType]::Button)
      Invoke-Control -Element $homeButton
      Start-Sleep -Milliseconds 500
      Capture-Window -Handle $handle -Path (Join-Path $artifactDir "scenario_controls_home.png")
      $reset = Find-Control -Root $window -Name "Reset All" -ControlType ([System.Windows.Automation.ControlType]::Button)
      Invoke-Control -Element $reset
      Start-Sleep -Milliseconds 700
      Capture-Window -Handle $handle -Path (Join-Path $artifactDir "scenario_controls_reset.png")
      Write-Output "PASS controls"
    }
    "animation" {
      $shear = Find-InvokableControl -Root $window -Name "Shear"
      Invoke-Control -Element $shear
      Start-Sleep -Milliseconds 500
      $animate = Find-Control -Root $window -Name "Animate" -ControlType ([System.Windows.Automation.ControlType]::Button)
      Invoke-Control -Element $animate
      Start-Sleep -Milliseconds 280
      Capture-Window -Handle $handle -Path (Join-Path $artifactDir "scenario_animation_mid.png")
      Start-Sleep -Milliseconds 1050
      Capture-Window -Handle $handle -Path (Join-Path $artifactDir "scenario_animation_end.png")
      Write-Output "PASS animation"
    }
  }
} finally {
  if ($process -and -not $process.HasExited) {
    Stop-Process -Id $process.Id
  }
}
